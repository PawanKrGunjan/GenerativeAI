"""
Database operations: schema setup, document indexing, chunking, semantic search
"""

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from .llm import embed_text

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)


# ────────────────────────────────────────────────
# Schema setup
# ────────────────────────────────────────────────


def setup_facts_table(
    conn: sqlite3.Connection,
    log: Optional[logging.Logger] = None,
) -> None:
    """Create user_facts table if not exists"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_facts (
            user_id     TEXT NOT NULL,
            key         TEXT NOT NULL,
            value       TEXT NOT NULL,
            confidence  REAL NOT NULL,
            source      TEXT,
            updated_at  TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        )
    """)
    conn.commit()
    if log:
        log.info("Ensured table: user_facts")


def setup_documents_table(
    conn: sqlite3.Connection,
    log: Optional[logging.Logger] = None,
) -> None:
    """Create documents metadata table"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            file_path   TEXT PRIMARY KEY,
            file_hash   TEXT NOT NULL,
            content     TEXT NOT NULL,
            indexed_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    if log:
        log.info("Ensured table: documents")


def setup_document_chunks_table(
    conn: sqlite3.Connection,
    log: Optional[logging.Logger] = None,
) -> None:
    """Create vector chunks table"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path   TEXT NOT NULL,
            chunk_text  TEXT NOT NULL,
            embedding   BLOB NOT NULL,
            chunk_index INTEGER NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON document_chunks(file_path)"
    )
    conn.commit()
    if log:
        log.info("Ensured table: document_chunks")


# ────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────


def compute_file_hash(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file"""
    filepath = Path(filepath)
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_document_content(filepath: Path | str) -> str:
    """Load content from supported file types"""
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(filepath))
            docs = loader.load()
            return "\n".join(doc.page_content for doc in docs)

        if ext == ".csv":
            loader = CSVLoader(str(filepath))
            docs = loader.load()
            return "\n".join(doc.page_content for doc in docs)

        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
            return df.to_string(index=False)

        if ext in (".txt", ".md", ".text"):
            return filepath.read_text(encoding="utf-8", errors="replace")

        logger.warning("Unsupported file type: %s", ext)
        return ""

    except Exception:
        logger.exception("Failed to load document %s", filepath.name)
        return ""


# ────────────────────────────────────────────────
# Indexing
# ────────────────────────────────────────────────


def index_document_chunks(
    conn: sqlite3.Connection,
    log: Optional[logging.Logger],
    file_path: Path | str,
    content: str,
) -> None:
    """Split content → embed chunks → store"""
    file_path = Path(file_path)
    conn.execute(
        "DELETE FROM document_chunks WHERE file_path = ?",
        (str(file_path),),
    )

    chunks = text_splitter.split_text(content)
    if not chunks:
        if log:
            log.warning("No chunks produced for %s", file_path.name)
        return

    batch = []
    for i, chunk in enumerate(chunks):
        enriched = f"Source: {file_path.name}\n\n{chunk}"
        try:
            embedding = embed_text(enriched)
            batch.append(
                (
                    str(file_path),
                    enriched,
                    embedding.tobytes(),
                    i,
                    datetime.now(timezone.utc).isoformat(),
                )
            )
        except Exception as e:
            if log:
                log.error(
                    "Embedding failed for chunk %d of %s: %s",
                    i,
                    file_path.name,
                    e,
                )

    if batch:
        conn.executemany(
            """
            INSERT INTO document_chunks
            (file_path, chunk_text, embedding, chunk_index, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()
        if log:
            log.info("Indexed %d chunks from %s", len(batch), file_path.name)


def sync_documents_to_db(
    conn: sqlite3.Connection,
    log: Optional[logging.Logger],
    folder: Path | str = DATA_DIR,
) -> None:
    """Scan folder, hash check, index new/changed documents"""
    folder = Path(folder)
    if not folder.is_dir():
        if log:
            log.warning("Data directory not found: %s", folder)
        return

    files = [p for p in folder.iterdir() if p.is_file()]
    if not files:
        return

    for filepath in tqdm(files, desc="Indexing documents"):
        abs_path = str(filepath.resolve())
        current_hash = compute_file_hash(filepath)

        row = conn.execute(
            "SELECT file_hash FROM documents WHERE file_path = ?",
            (abs_path,),
        ).fetchone()

        if row and row[0] == current_hash:
            continue

        content = load_document_content(filepath)
        if not content.strip():
            continue

        try:
            ts = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO documents (file_path, file_hash, content, indexed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    content = excluded.content,
                    indexed_at = excluded.indexed_at
                """,
                (abs_path, current_hash, content, ts),
            )

            index_document_chunks(conn, log, abs_path, content)

        except Exception:
            if log:
                log.exception("Failed to process %s", filepath.name)

    conn.commit()


# ────────────────────────────────────────────────
# Querying
# ────────────────────────────────────────────────


def semantic_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 7,
) -> List[str]:
    """In-DB cosine similarity search"""
    if not query.strip():
        return []

    try:
        query_emb = embed_text(query.strip())
    except Exception:
        return []

    rows = conn.execute("SELECT chunk_text, embedding FROM document_chunks").fetchall()

    if not rows:
        return []

    scored = []
    q_norm = np.linalg.norm(query_emb)

    for text, emb_blob in rows:
        try:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            denom = q_norm * np.linalg.norm(emb)
            sim = float(np.dot(query_emb, emb) / denom) if denom else 0.0
            scored.append((sim, text))
        except Exception:
            pass

    scored.sort(reverse=True, key=lambda x: x[0])
    return [text for _, text in scored[:top_k]]


def get_indexed_documents(conn: sqlite3.Connection) -> List[str]:
    """Return list of indexed file paths"""
    rows = conn.execute(
        "SELECT file_path FROM documents ORDER BY indexed_at DESC"
    ).fetchall()
    return [r[0] for r in rows]
