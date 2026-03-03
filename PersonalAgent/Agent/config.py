"""
Agent/config.py
Central configuration — paths, model names, constants
"""

from pathlib import Path
from typing import Final

# ── LLM & Embedding ──────────────────────────────────────────────
MODEL_NAME: Final[str] = "llama3.2:3b"
EMBEDDING_MODEL: Final[str] = "nomic-embed-text:latest"
EMBEDDING_DIM = 768
# ── Directories (relative to project root) ───────────────────────
ROOT_DIR: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = ROOT_DIR / "data"
LOG_DIR: Final[Path] = ROOT_DIR / "logs"
GRAPH_DIR: Final[Path] = ROOT_DIR / "graphs"
HISTORY_DIR: Final[Path] = ROOT_DIR / "history"

# ── Chunking & Retrieval settings ────────────────────────────────
CHUNK_SIZE: Final[int] = 800
CHUNK_OVERLAP: Final[int] = 150
TOP_K: Final[int] = 7
MAX_SOURCE_LENGTH: Final[int] = 250  # used in fact extraction

# Ensure all directories exist
for directory in (DATA_DIR, LOG_DIR, GRAPH_DIR, HISTORY_DIR):
    directory.mkdir(parents=True, exist_ok=True)


DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "personalagent",
    "user": "postgres",
    "password": "Ganesh123",
}

DEFAULT_MAX_ITERATIONS=3
ENABLE_RESPONSE_RATING = True  # Set to False in production
ENABLE_RESPONSE_CORRECTION = True # Disable in Productions