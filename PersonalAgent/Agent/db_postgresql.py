"""
PostgreSQL + pgvector + nomic-embed-text (768 dims)
psycopg 3.x compatible
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm

from .document_reader import text_splitter, compute_file_hash, load_document_content
from .config import DATA_DIR, DB_CONFIG, EMBEDDING_DIM
from .llm import embed_text


# ────────────────────────────────────────────────
# Connection
# ────────────────────────────────────────────────

def get_pg_connection(log: logging.Logger) -> psycopg.Connection:
    """Create PostgreSQL connection (psycopg3)."""
    try:
        conn = psycopg.connect(**DB_CONFIG, row_factory=dict_row)
        log.info("✅ Connected to PostgreSQL")
        return conn
    except Exception:
        log.exception("❌ PostgreSQL connection failed")
        raise


# ────────────────────────────────────────────────
# Schema Setup
# ────────────────────────────────────────────────
def setup_documents_table(conn, log=None):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    file_path   TEXT PRIMARY KEY,
                    file_hash   TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
        conn.commit()

        if log:
            log.info("Ensured table: documents")

    except Exception as e:
        if log:
            log.exception("setup_documents_table failed: %s", e)
        conn.rollback()
        raise

def setup_document_chunks_table(conn, log=None):
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id          BIGSERIAL PRIMARY KEY,
                    file_path   TEXT NOT NULL,
                    chunk_text  TEXT NOT NULL,
                    embedding   VECTOR(768) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
        conn.commit()

        if log:
            log.info("Ensured table: document_chunks")

    except Exception as e:
        if log:
            log.exception("setup_document_chunks_table failed: %s", e)
        conn.rollback()
        raise

def setup_chat_history_table(conn: psycopg.Connection, log: logging.Logger) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id            BIGSERIAL PRIMARY KEY,
                    user_id       VARCHAR(100) NOT NULL,
                    role          VARCHAR(50) NOT NULL,
                    title         VARCHAR(200),
                    human_vector  VECTOR({EMBEDDING_DIM}) NOT NULL,
                    ai_vector     VECTOR({EMBEDDING_DIM}) NOT NULL,
                    human_text    TEXT NOT NULL,
                    ai_text       TEXT NOT NULL,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_user_role ON chat_history(user_id, role)"
            )
        conn.commit()
        log.info("✅ chat_history table ready")

        # Concurrent vector indexes
        old_autocommit = conn.autocommit
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chat_human_vec
                    ON chat_history USING ivfflat (human_vector vector_cosine_ops)
                    WITH (lists=100)
                """)
                cur.execute(f"""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chat_ai_vec
                    ON chat_history USING ivfflat (ai_vector vector_cosine_ops)
                    WITH (lists=100)
                """)
            log.info("✅ chat_history vector indexes ready")
        finally:
            conn.autocommit = old_autocommit

    except Exception:
        conn.rollback()
        log.exception("❌ chat_history setup failed")
        raise


# ────────────────────────────────────────────────
# Sync
# ────────────────────────────────────────────────

def sync_chat_history(
    conn: psycopg.Connection,
    user_id: str,
    role: str,
    human_msg: str,
    ai_response: str,
    embeddings,
    log: logging.Logger,
) -> bool:
    try:
        title = human_msg.split(".")[0][:200]

        human_vec = embeddings.embed_query(f"[ROLE: {role}] {human_msg}")
        ai_vec = embeddings.embed_query(f"[ROLE: {role}] {ai_response}")

        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO chat_history
                (user_id, role, title, human_vector, ai_vector, human_text, ai_text)
                VALUES (%s, %s, %s, %s::vector({EMBEDDING_DIM}),
                        %s::vector({EMBEDDING_DIM}), %s, %s)
                """,
                (user_id, role, title, human_vec, ai_vec, human_msg, ai_response),
            )

        conn.commit()
        log.info("💾 Synced chat | user=%s | role=%s | title=%s",
                 user_id, role, title[:50])
        return True

    except Exception:
        conn.rollback()
        log.exception("❌ Failed to sync chat")
        return False

# Indexing
def index_document_chunks(
    conn: 'psycopg.Connection',
    file_path: Path | str,
    content: str,
    log: Optional[logging.Logger]
) -> None:
    file_path = Path(file_path)
    
    with conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE file_path = %s", (str(file_path),))

    chunks = text_splitter.split_text(content)
    if not chunks:
        log.error("No chunks: %s", file_path.name)
        return

    batch = []
    for i, chunk in enumerate(chunks):
        enriched = f"Source: {file_path.name}\n\n{chunk}"
        try:
            batch.append((
                str(file_path),
                enriched,
                embed_text(enriched).tolist(),  # List or numpy array OK
                i,
                datetime.now(timezone.utc).isoformat(),
            ))
        except Exception as e:
            log.error("Embed failed %d/%s: %s", i, file_path.name, e)

    if batch:
        try:
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO document_chunks
                    (file_path, chunk_text, embedding, chunk_index, created_at)
                    VALUES (%s, %s, %s, %s, %s)  -- No cast needed!
                """, batch)
            conn.commit()
            log.info("Indexed %d chunks: %s", len(batch), file_path.name)
        except Exception as e:
            conn.rollback()
            log.error("Insert failed %s: %s", file_path.name, e)
            raise("Insert failed %s: %s", file_path.name, e)


def sync_documents_to_db(
    conn: 'psycopg.Connection',
    log: Optional[logging.Logger],
    folder: Path | str = DATA_DIR,
) -> None:
    folder = Path(folder)
    if not folder.is_dir():
        if log: log.warning("No data dir: %s", folder)
        return

    files = [p for p in folder.iterdir() if p.is_file()]
    if not files: return

    for filepath in tqdm(files, desc="Indexing"):
        abs_path = str(filepath.resolve())
        current_hash = compute_file_hash(filepath, log)

        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("SELECT file_hash FROM documents WHERE file_path = %s", (abs_path,))
            row = cur.fetchone()

            if row and row['file_hash'] == current_hash: continue
            content = load_document_content(filepath,log)
            if not content.strip(): continue

            try:
                ts = datetime.now(timezone.utc).isoformat()
                cur.execute("""
                    INSERT INTO documents (file_path, file_hash, content, indexed_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(file_path) DO UPDATE SET
                        file_hash = EXCLUDED.file_hash,
                        content = EXCLUDED.content,
                        indexed_at = EXCLUDED.indexed_at
                """, (abs_path, current_hash, content, ts))
                index_document_chunks(conn, abs_path, content,log)
            except Exception as e:
                if log: log.exception("Process failed %s: %s", filepath.name, e)
                conn.rollback()
                log.exception("Sync failed")
                raise
    conn.commit()
    if log: log.info("Sync complete")


# ────────────────────────────────────────────────
# Semantic Search (Documents)
# ────────────────────────────────────────────────

def semantic_search(
    conn: psycopg.Connection,
    log: logging.Logger,
    query: str,
    top_k: int = 3,
) -> List[str]:

    if not query.strip():
        log.debug("Empty document search query")
        return []

    try:
        query_emb = embed_text(query.strip()).tolist()
    except Exception:
        log.exception("❌ Document embedding failed")
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_text
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_emb, top_k),
            )
            rows = cur.fetchall()

        log.info("📄 Document search returned %d results", len(rows))
        return [r["chunk_text"] for r in rows]

    except Exception:
        log.exception("❌ Document semantic search failed")
        return []


# ────────────────────────────────────────────────
# Semantic Search (Chat, Role Filtered)
# ────────────────────────────────────────────────

def semantic_search_chat(
    conn: psycopg.Connection,
    log: logging.Logger,
    query: str,
    user_id: str,
    role: str,
    embeddings,
    top_k: int = 3,
) -> List[Dict]:

    if not query.strip():
        log.debug("Empty chat search query")
        return []

    try:
        query_emb = embeddings.embed_query(f"[ROLE: {role}] {query}")
    except Exception:
        log.exception("❌ Chat embedding failed")
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT title, human_text, ai_text,
                       human_vector <=> %s::vector AS distance
                FROM chat_history
                WHERE user_id = %s
                  AND role = %s
                ORDER BY human_vector <=> %s::vector
                LIMIT %s
                """,
                (query_emb, user_id, role, query_emb, top_k),
            )
            rows = cur.fetchall()

        log.info(
            "🧠 Chat search | user=%s | role=%s | results=%d",
            user_id,
            role,
            len(rows),
        )
        return rows

    except Exception:
        log.exception("❌ Chat semantic search failed")
        return []
    


def get_indexed_documents(
    conn,
    log: Optional[logging.Logger] = None
) -> List[str]:
    """
    Return all indexed document file paths ordered by latest.
    """
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT file_path FROM documents ORDER BY indexed_at DESC"
            )
            rows = cur.fetchall()

        if log:
            log.info("Retrieved %d indexed documents", len(rows))

        return [row["file_path"] for row in rows]

    except Exception as e:
        if log:
            log.exception("Failed to fetch indexed documents: %s", e)
        return []