"""
app/db_connect.py

Optimized PostgreSQL + pgvector database layer
Production-ready for LangGraph / AI agents
"""

import logging
from typing import List, Optional, Any, Dict
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from utils.config import DB_CONFIG, NIFTY_50_FILE


# ────────────────────────────────────────────────
# CONNECTION POOL (FAST)
# ────────────────────────────────────────────────

POOL = ConnectionPool(
    conninfo=DB_CONFIG["dsn"],
    min_size=1,
    max_size=10,
    kwargs={"row_factory": dict_row},
)


@contextmanager
def get_connection(log: logging.Logger):
    """
    Fetch connection from pool.
    Much faster than creating new connections.
    """
    try:
        with POOL.connection() as conn:
            register_vector(conn)
            yield conn
    except Exception:
        log.exception("Database connection error")
        raise


# ────────────────────────────────────────────────
# DATABASE INITIALIZATION
# ────────────────────────────────────────────────

def initialize_database(log: logging.Logger):
    """Create tables + indexes safely."""

    with get_connection(log) as conn:
        with conn.cursor() as cur:

            # Extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

            # NSE MASTER TABLE
            cur.execute("""
            CREATE TABLE IF NOT EXISTS nse_stocks (
                symbol TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """)

            # NIFTY 50 TABLE
            cur.execute("""
            CREATE TABLE IF NOT EXISTS nifty50_stocks (
                symbol TEXT PRIMARY KEY,
                industry TEXT,
                series TEXT,
                isin_code TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY(symbol)
                REFERENCES nse_stocks(symbol)
                ON DELETE CASCADE
            );
            """)

            # EMBEDDINGS TABLE
            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_embeddings (
                symbol TEXT PRIMARY KEY,
                summary TEXT,
                embedding VECTOR(768),
                created_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY(symbol)
                REFERENCES nse_stocks(symbol)
                ON DELETE CASCADE
            );
            """)

            # PRICE HISTORY TABLE
            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                trade_date DATE NOT NULL,
                open NUMERIC,
                high NUMERIC,
                low NUMERIC,
                close NUMERIC,
                volume BIGINT,
                created_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY(symbol)
                REFERENCES nse_stocks(symbol)
                ON DELETE CASCADE,
                UNIQUE(symbol, trade_date)
            );
            """)

            # INDEXES

            # vector search
            cur.execute("""
            CREATE INDEX IF NOT EXISTS stock_embedding_idx
            ON stock_embeddings
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """)

            # price analytics
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_symbol_date
            ON stock_prices(symbol, trade_date DESC);
            """)

            # fuzzy company search
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_company_name_trgm
            ON nse_stocks
            USING gin (company_name gin_trgm_ops);
            """)

        conn.commit()

    log.info("Database initialized successfully")


# ────────────────────────────────────────────────
# UPSERT MASTER STOCK
# ────────────────────────────────────────────────

def upsert_nse_stock(symbol: str, company_name: str, log: logging.Logger):

    with get_connection(log) as conn:
        with conn.cursor() as cur:

            cur.execute("""
            INSERT INTO nse_stocks (symbol, company_name)
            VALUES (%s, %s)
            ON CONFLICT (symbol)
            DO UPDATE SET company_name = EXCLUDED.company_name;
            """, (symbol, company_name))

        conn.commit()


# ────────────────────────────────────────────────
# UPSERT NIFTY 50
# ────────────────────────────────────────────────

def upsert_nifty50_stock(
    symbol: str,
    industry: str,
    series: str,
    isin_code: str,
    log: logging.Logger
):

    with get_connection(log) as conn:
        with conn.cursor() as cur:

            cur.execute("""
            INSERT INTO nifty50_stocks
            (symbol, industry, series, isin_code)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT(symbol)
            DO UPDATE SET
                industry = EXCLUDED.industry,
                series = EXCLUDED.series,
                isin_code = EXCLUDED.isin_code;
            """, (symbol, industry, series, isin_code))

        conn.commit()


# ────────────────────────────────────────────────
# UPSERT EMBEDDINGS
# ────────────────────────────────────────────────

def upsert_stock_embedding(
    symbol: str,
    summary: str,
    embedding: List[float],
    log: logging.Logger
):

    with get_connection(log) as conn:
        with conn.cursor() as cur:

            cur.execute(
                "SELECT 1 FROM nse_stocks WHERE symbol = %s;",
                (symbol,)
            )

            if cur.fetchone() is None:
                raise ValueError(f"Symbol {symbol} not found")

            cur.execute("""
            INSERT INTO stock_embeddings(symbol, summary, embedding)
            VALUES (%s,%s,%s)
            ON CONFLICT(symbol)
            DO UPDATE SET
                summary = EXCLUDED.summary,
                embedding = EXCLUDED.embedding,
                created_at = NOW();
            """, (symbol, summary, embedding))

        conn.commit()


# ────────────────────────────────────────────────
# FETCH STOCK WITH METADATA
# ────────────────────────────────────────────────

def get_stock_by_symbol(
    symbol: str,
    log: logging.Logger
) -> Optional[Dict[str, Any]]:

    with get_connection(log) as conn:
        with conn.cursor() as cur:

            cur.execute("""
            SELECT
                e.symbol,
                n.company_name,
                nf.industry,
                nf.series,
                nf.isin_code,
                e.summary,
                e.created_at
            FROM stock_embeddings e
            JOIN nse_stocks n
                ON e.symbol = n.symbol
            LEFT JOIN nifty50_stocks nf
                ON e.symbol = nf.symbol
            WHERE e.symbol = %s
            ORDER BY e.created_at DESC
            LIMIT 1;
            """, (symbol,))

            return cur.fetchone()


# ────────────────────────────────────────────────
# MARKET ANALYTICS
# ────────────────────────────────────────────────

def get_top_losers(log: logging.Logger, limit: int = 10):

    with get_connection(log) as conn:
        with conn.cursor() as cur:

            cur.execute("""
            SELECT
                s.symbol,
                s.close,
                p.close AS prev_close,
                ROUND(((s.close - p.close) / p.close) * 100, 2) AS pct_loss
            FROM stock_prices s
            JOIN stock_prices p
                ON s.symbol = p.symbol
                AND p.trade_date = s.trade_date - INTERVAL '1 day'
            WHERE s.trade_date = (
                SELECT MAX(trade_date)
                FROM stock_prices
            )
            ORDER BY pct_loss ASC
            LIMIT %s;
            """, (limit,))

            return cur.fetchall()


# ────────────────────────────────────────────────
# ANALYZE (IMPORTANT FOR VECTOR INDEX)
# ────────────────────────────────────────────────

def analyze_embeddings_table(log: logging.Logger):

    with get_connection(log) as conn:
        with conn.cursor() as cur:
            cur.execute("ANALYZE stock_embeddings;")

        conn.commit()


# ────────────────────────────────────────────────
# GENERIC QUERY
# ────────────────────────────────────────────────

def execute_query(
    query: str,
    params: Optional[tuple],
    log: logging.Logger
) -> List[Dict[str, Any]]:

    with get_connection(log) as conn:
        with conn.cursor() as cur:

            cur.execute(query, params)

            if cur.description:
                return cur.fetchall()

            conn.commit()
            return []


# ────────────────────────────────────────────────
# BASIC STATS
# ────────────────────────────────────────────────

def count_nse_stocks(log: logging.Logger) -> int:

    with get_connection(log) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM nse_stocks;")
            return cur.fetchone()["count"]


def count_nifty50_stocks(log: logging.Logger) -> int:

    with get_connection(log) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM nifty50_stocks;")
            return cur.fetchone()["count"]