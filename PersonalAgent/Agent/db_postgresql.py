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
import traceback
from tqdm import tqdm
from .document_reader import text_splitter, compute_file_hash, load_document_content
from .config import DATA_DIR, DB_CONFIG, EMBEDDING_DIM
from .llm import embed_text
from .logger import setup_logger

logger = setup_logger("")


# ────────────────────────────────────────────────
# psycopg 3.x Connection & Schema
# ────────────────────────────────────────────────
def get_pg_connection() -> "psycopg.Connection":
    """Connection using psycopg 3.x"""
    # print(f"Connecting PostgreSQL database {DB_CONFIG}")
    return psycopg.connect(
        **DB_CONFIG, row_factory=psycopg.rows.dict_row  # dict-like rows
    )


def setup_facts_table(
    conn: "psycopg.Connection", log: Optional[logging.Logger]
) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                user_id     TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                confidence  REAL NOT NULL,
                source      TEXT,
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (user_id, key)
            )
        """)
        conn.commit()
    log.info("Ensured table: user_facts")


def setup_chat_history_table(
    conn: "psycopg.Connection", log: Optional[logging.Logger]
) -> None:
    """Setup chat_history table + pgvector (psycopg3 safe)."""
    try:
        # 1. Tables/extensions (normal transaction)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id          BIGSERIAL PRIMARY KEY,
                    user_id     VARCHAR(100) NOT NULL,
                    title       VARCHAR(200),
                    human_vector VECTOR(768) NOT NULL,
                    ai_vector   VECTOR(768) NOT NULL,
                    human_text  TEXT NOT NULL,
                    ai_text     TEXT NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id)"
            )
        conn.commit()

        # 2. Concurrent indexes (autocommit mode)
        was_autocommit = conn.autocommit  # Save original
        conn.autocommit = True  # KEY FIX: psycopg3 property

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chat_human_vec 
                    ON chat_history USING ivfflat (human_vector vector_cosine_ops) WITH (lists=100)
                """)
                cur.execute("""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chat_ai_vec 
                    ON chat_history USING ivfflat (ai_vector vector_cosine_ops) WITH (lists=100)
                """)
        finally:
            conn.autocommit = was_autocommit  # Restore

        if log:
            log.info("✅ chat_history + concurrent indexes ready (768 dims)")

    except Exception as e:
        if log:
            log.error(f"chat_history setup failed: {e}")
        raise RuntimeError(f"chat_history failed: {e}")


def setup_documents_table(
    conn: "psycopg.Connection", log: Optional[logging.Logger]
) -> None:
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
    log.info("Ensured table: documents")


def setup_document_chunks_table(
    conn: "psycopg.Connection", log: Optional[logging.Logger]
) -> None:
    try:
        # Table + extension (transactional)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id          BIGSERIAL PRIMARY KEY,
                    file_path   TEXT NOT NULL,
                    chunk_text  TEXT NOT NULL,
                    embedding   VECTOR({EMBEDDING_DIM}) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
        conn.commit()
    except Exception as e:
        conn.rollback()
        log.error(f"Table creation failed: {e}")
        raise (f"Table creation failed: {e}")

    # Indexes with autocommit (psycopg3 syntax)
    try:
        conn.autocommit = True  # psycopg3: direct attribute
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_file_path 
                ON document_chunks(file_path)
            """)
            cur.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_embedding 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100)
            """)
            cur.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_created_at 
                ON document_chunks(created_at)
            """)
    finally:
        conn.autocommit = False  # Reset for caller

    log.info("Ensured document_chunks (VECTOR(768)) + indexes")


# ────────────────────────────────────────────────
# Indexing
# ────────────────────────────────────────────────
# Indexing
def index_document_chunks(
    conn: "psycopg.Connection",
    file_path: Path | str,
    content: str,
    log: Optional[logging.Logger],
) -> None:
    file_path = Path(file_path)

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_chunks WHERE file_path = %s", (str(file_path),)
        )

    chunks = text_splitter.split_text(content)
    if not chunks:
        log.error("No chunks: %s", file_path.name)
        return

    batch = []
    for i, chunk in enumerate(chunks):
        enriched = f"Source: {file_path.name}\n\n{chunk}"
        try:
            batch.append(
                (
                    str(file_path),
                    enriched,
                    embed_text(enriched).tolist(),  # List or numpy array OK
                    i,
                    datetime.now(timezone.utc).isoformat(),
                )
            )
        except Exception as e:
            log.error("Embed failed %d/%s: %s", i, file_path.name, e)

    if batch:
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO document_chunks
                    (file_path, chunk_text, embedding, chunk_index, created_at)
                    VALUES (%s, %s, %s, %s, %s)  -- No cast needed!
                """,
                    batch,
                )
            conn.commit()
            log.info("Indexed %d chunks: %s", len(batch), file_path.name)
        except Exception as e:
            conn.rollback()
            log.error("Insert failed %s: %s", file_path.name, e)
            raise ("Insert failed %s: %s", file_path.name, e)


def sync_documents_to_db(
    conn: "psycopg.Connection",
    log: Optional[logging.Logger],
    folder: Path | str = DATA_DIR,
) -> None:
    folder = Path(folder)
    if not folder.is_dir():
        if log:
            log.warning("No data dir: %s", folder)
        return

    files = [p for p in folder.iterdir() if p.is_file()]
    if not files:
        return

    for filepath in tqdm(files, desc="Indexing"):
        abs_path = str(filepath.resolve())
        current_hash = compute_file_hash(filepath, log)

        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                "SELECT file_hash FROM documents WHERE file_path = %s", (abs_path,)
            )
            row = cur.fetchone()

            if row and row["file_hash"] == current_hash:
                continue
            content = load_document_content(filepath, log)
            if not content.strip():
                continue

            try:
                ts = datetime.now(timezone.utc).isoformat()
                cur.execute(
                    """
                    INSERT INTO documents (file_path, file_hash, content, indexed_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(file_path) DO UPDATE SET
                        file_hash = EXCLUDED.file_hash,
                        content = EXCLUDED.content,
                        indexed_at = EXCLUDED.indexed_at
                """,
                    (abs_path, current_hash, content, ts),
                )
                index_document_chunks(conn, abs_path, content, log)
            except Exception as e:
                if log:
                    log.exception("Process failed %s: %s", filepath.name, e)
                conn.rollback()
                log.exception("Sync failed")
                raise
    conn.commit()
    if log:
        log.info("Sync complete")


def sync_chat_history(conn, user_id, human_msg, ai_response, embeddings, logger=None):
    title = human_msg.split(".")[0][:200]
    human_vec = embeddings.embed_query(human_msg)  # 768 dims
    ai_vec = embeddings.embed_query(ai_response)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_history (user_id, title, human_vector, ai_vector, human_text, ai_text)
            VALUES (%s, %s, %s::vector(768), %s::vector(768), %s, %s)
        """,
            (user_id, title, human_vec, ai_vec, human_msg, ai_response),
        )
    conn.commit()
    logger.info(f"Synced: {title[:50]}...")
    return True


# ────────────────────────────────────────────────
# Querying
# ────────────────────────────────────────────────
def get_indexed_documents(
    conn: "psycopg.Connection", log: Optional[logging.Logger]
) -> List[str]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT file_path FROM documents ORDER BY indexed_at DESC")
        rows = cur.fetchall()
        log.info(f" {len(rows)} indexed documents")
        return [row["file_path"] for row in rows]


def semantic_search(
    conn: "psycopg.Connection",
    log: Optional[logging.Logger],
    query: str,
    top_k: int = 3,
) -> List[str]:
    if not query.strip():
        return []

    try:
        query_emb = embed_text(query.strip()).tolist()
    except Exception as e:
        log.error("Query embed failed: %s", e)
        return []

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT chunk_text FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """,
            (query_emb, top_k),
        )
        rows = cur.fetchall()
        log.info(f"{len(rows)} result")
        return [row["chunk_text"] for row in rows]


def semantic_search_chat(
    conn: "psycopg.Connection",
    log: Optional[logging.Logger],
    query: str,
    user_id: str,
    top_k: int = 3,
) -> List[Dict]:
    """Semantic search past chats like documents."""
    if not query.strip():
        return []

    try:
        query_emb = embed_text(query.strip()).tolist()
    except Exception as e:
        log.error("Chat query embed failed: %s", e)
        return []

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT title, human_text, ai_text, human_vector <=> %s::vector AS distance
            FROM chat_history 
            WHERE user_id = %s
            ORDER BY human_vector <=> %s::vector
            LIMIT %s
        """,
            (query_emb, user_id, query_emb, top_k),
        )

        rows = cur.fetchall()
        log.info(f"Found {len(rows)} similar chats")
        return rows  # Returns [{'title':..., 'human_text':..., 'ai_text':..., 'distance':0.12}, ...]


def get_user_facts(
    conn, log: Optional[logging.Logger], user_id: str, limit: int = 5
) -> List[tuple]:
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT key, value, confidence 
                FROM user_facts 
                WHERE user_id = %s 
                ORDER BY updated_at DESC LIMIT %s
            """,
                (user_id, limit),
            )
            rows = cur.fetchall()
            log.info(f"{len(rows)} user fetched")
            return [
                (
                    r["key"],
                    r["value"],
                    r["confidence"],
                )  # Use dict_row consistently, no hasattr checks
                for r in rows
            ]
    except Exception:  # Broader catch for simplicity; import specific error if needed
        conn.rollback()
        return []
