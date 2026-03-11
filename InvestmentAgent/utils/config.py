"""
app/config.py
Central configuration — paths, model names, constants
Loads from .env file (using python-dotenv)
"""

import os
from pathlib import Path
from typing import Final
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()  # looks for .env in current working directory (usually project root)

# ── LLM & Embedding ──────────────────────────────────────────────
MODEL_NAME: Final[str] = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBEDDING_MODEL: Final[str] = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
EMBEDDING_DIM: Final[int] = int(os.getenv("EMBEDDING_DIM", "768"))

# ── Directories (relative to project root) ───────────────────────
ROOT_DIR: Final[Path] = Path(os.getenv("ROOT_DIR", ".")).resolve()
DATA_DIR: Final[Path] = ROOT_DIR / "data"
LOG_DIR: Final[Path] = ROOT_DIR / "logs"
GRAPH_DIR: Final[Path] = ROOT_DIR / "graphs"
#HISTORY_DIR: Final[Path] = ROOT_DIR / "history"

# Important data files
NIFTY_50_FILE: Final[Path] = DATA_DIR / "ind_nifty50list.csv"
ALL_NSE_SYMBOLS: Final[Path] = DATA_DIR / "all_nse_stocks.json"

# ── Chunking & Retrieval settings ────────────────────────────────
CHUNK_SIZE: Final[int] = 800
CHUNK_OVERLAP: Final[int] = 150
TOP_K: Final[int] = 7
MAX_SOURCE_LENGTH: Final[int] = 250

# Create directories if they don't exist
for directory in (DATA_DIR, LOG_DIR, GRAPH_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ── Database configuration ───────────────────────────────────────
# Recommended: use full connection string (preferred by psycopg & SQLAlchemy)
DATABASE_URL: Final[str] = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/investment"
)

# If your code still needs separate kwargs (e.g. old psycopg.connect style)
DB_CONFIG = {
    "host": os.getenv("POSTGRE_HOST", "localhost"),
    "port": int(os.getenv("POSTGRE_PORT", "5433")),
    "dbname": os.getenv("POSTGRE_DB", "investment"),
    "user": os.getenv("POSTGRE_USER", "postgres"),
    "password": os.getenv("POSTGRE_PASSWORD", ""),
    "connect_timeout": 15,
}

# Build DSN string for psycopg3
DB_CONFIG["dsn"] = (
    f"host={DB_CONFIG['host']} "
    f"port={DB_CONFIG['port']} "
    f"dbname={DB_CONFIG['dbname']} "
    f"user={DB_CONFIG['user']} "
    f"password={DB_CONFIG['password']}"
)

# ── Ollama / LLM base URL (if needed) ────────────────────────────
OLLAMA_BASE_URL: Final[str] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


IST = ZoneInfo("Asia/Kolkata")

# Optional: quick debug print when file is run directly
if __name__ == "__main__":
    print("Configuration loaded:")
    print(f"  ROOT_DIR         : {ROOT_DIR}")
    print(f"  LLM_MODEL        : {MODEL_NAME}")
    print(f"  DATABASE_URL     : {DATABASE_URL}")
    print(f"  DB_CONFIG (host) : {DB_CONFIG['host']}:{DB_CONFIG['port']}")