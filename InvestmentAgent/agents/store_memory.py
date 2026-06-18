"""
app/agents/memory.py

Persistent per-symbol memory with:
• Startup caching
• Company name DB preload
• Safe multi-symbol support
• Cache-first reads
• Disk persistence
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Union
from datetime import datetime
import threading

from psycopg.rows import dict_row

from utils.db_connect import get_connection
from utils.config import DATA_DIR, MEMORY_SAVE_DIR
from utils.logger import LOGGER
# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

MEMORY_DIR = MEMORY_SAVE_DIR / "Saved_Memory" / "symbols"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Global caches
# ─────────────────────────────────────────────

# symbol → memory object
MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}

# symbol → company name
COMPANY_NAMES: Dict[str, str] = {}

# thread safety
CACHE_LOCK = threading.Lock()

# ─────────────────────────────────────────────
# Startup Loaders
# ─────────────────────────────────────────────

def _load_company_names_from_db():
    """Load symbol → company name mapping."""
    global COMPANY_NAMES

    try:
        with get_connection(LOGGER) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT symbol, company_name FROM nse_stocks")

                rows = cur.fetchall()

                COMPANY_NAMES = {
                    r["symbol"].upper(): r["company_name"].strip()
                    for r in rows
                }

        LOGGER.info(f"Loaded {len(COMPANY_NAMES):,} company names from DB")

    except Exception as e:
        LOGGER.warning(f"Company name preload failed: {e}")
        COMPANY_NAMES = {}


def _load_all_memories_on_startup():
    """Preload existing memory JSON files."""
    count = 0

    for file in MEMORY_DIR.glob("*.json"):
        try:
            symbol = file.stem.upper()

            with open(file, "r", encoding="utf-8") as f:
                mem = json.load(f)

            MEMORY_CACHE[symbol] = mem
            count += 1

        except Exception as e:
            LOGGER.warning(f"Failed loading {file}: {e}")

    LOGGER.info(f"Pre-loaded {count} existing symbol memories into cache")


# Run once on import
_load_company_names_from_db()
_load_all_memories_on_startup()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _normalize_symbols(symbols: Union[str, List[str]]) -> List[str]:
    """Accept single or list of symbols."""
    if isinstance(symbols, str):
        return [symbols.upper()]

    return [s.upper() for s in symbols]


def _get_memory_path(symbol: str) -> Path:
    return MEMORY_DIR / f"{symbol}.json"


def _create_new_memory(symbol: str) -> Dict[str, Any]:
    """Create fresh memory structure."""

    company = COMPANY_NAMES.get(symbol, "Unknown Company")

    return {
        "symbol": symbol,
        "company_name": company,
        "created_at": datetime.utcnow().isoformat(),
        "last_updated": None,
        "reflections": [],
        "key_facts": {},
        "last_signals": {},
        "notes": []
    }

# ─────────────────────────────────────────────
# Core Memory Functions
# ─────────────────────────────────────────────

def load_symbol_memory(symbol: Union[str, List[str]]) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Load memory for one or multiple symbols.

    Returns
    -------
    dict
        if single symbol

    dict[symbol → memory]
        if multiple symbols
    """

    symbols = _normalize_symbols(symbol)

    results = {}

    with CACHE_LOCK:

        for sym in symbols:

            # Cache hit
            if sym in MEMORY_CACHE:
                results[sym] = MEMORY_CACHE[sym]
                continue

            path = _get_memory_path(sym)

            # Disk load
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        mem = json.load(f)

                    MEMORY_CACHE[sym] = mem
                    results[sym] = mem
                    continue

                except Exception as e:
                    LOGGER.warning(f"Failed loading memory for {sym}: {e}")

            # Create new
            mem = _create_new_memory(sym)
            MEMORY_CACHE[sym] = mem
            _save_memory(sym, mem)

            results[sym] = mem

    if len(symbols) == 1:
        return results[symbols[0]]

    return results


def _save_memory(symbol: str, memory: Dict[str, Any]):
    """Internal save."""

    path = _get_memory_path(symbol)

    memory["last_updated"] = datetime.utcnow().isoformat()

    if not memory.get("company_name"):
        memory["company_name"] = COMPANY_NAMES.get(symbol, "Unknown Company")

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)

        MEMORY_CACHE[symbol] = memory

    except Exception as e:
        LOGGER.error(f"Failed saving memory for {symbol}: {e}")


def save_symbol_memory(symbol: str, memory: Dict[str, Any]):
    """Public save wrapper."""
    symbol = symbol.upper()

    with CACHE_LOCK:
        _save_memory(symbol, memory)


# ─────────────────────────────────────────────
# Memory Update Utilities
# ─────────────────────────────────────────────

def update_reflection(symbol: str, reflection: str):

    mem = load_symbol_memory(symbol)

    mem["reflections"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "reflection": reflection.strip()
    })

    save_symbol_memory(symbol, mem)


def add_key_fact(symbol: str, key: str, value: Any):

    mem = load_symbol_memory(symbol)

    mem["key_facts"][key.strip()] = value

    save_symbol_memory(symbol, mem)


def add_note(symbol: str, note: str):

    mem = load_symbol_memory(symbol)

    mem["notes"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "note": note.strip()
    })

    save_symbol_memory(symbol, mem)


# ─────────────────────────────────────────────
# Fast Lookups
# ─────────────────────────────────────────────

def get_company_name(symbol: str) -> str:

    sym = symbol.upper()

    if sym in COMPANY_NAMES:
        return COMPANY_NAMES[sym]

    mem = load_symbol_memory(sym)

    return mem.get("company_name", "Unknown Company")


# ─────────────────────────────────────────────
# Conversation History
# ─────────────────────────────────────────────

CONVERSATION_DIR = MEMORY_SAVE_DIR / "Saved_Memory" / "conversations"
CONVERSATION_DIR.mkdir(parents=True, exist_ok=True)

def _get_conversation_path(user_id: str):
    return CONVERSATION_DIR / f"{user_id}.json"

def load_conversation_history(user_id: str):
    path = _get_conversation_path(user_id)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)  # list of {"role": "human/ai", "content": "..."}
    except Exception as e:
        LOGGER.warning(f"Failed to load conversation history for {user_id}: {e}")
        return []

def save_conversation_history(user_id: str, messages: list):
    path = _get_conversation_path(user_id)
    serializable = []
    for m in messages:
        role = "human" if m.__class__.__name__ == "HumanMessage" else "ai"
        serializable.append({"role": role, "content": m.content})
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    except Exception as e:
        LOGGER.error(f"Failed to save conversation history for {user_id}: {e}")

TOOL_RESULT_DIR = MEMORY_SAVE_DIR / "Saved_Memory" / "tools"
TOOL_RESULT_DIR.mkdir(parents=True, exist_ok=True)

def _get_tool_path(symbol: str, tool_name: str):
    return TOOL_RESULT_DIR / f"{symbol}_{tool_name}.json"

def save_tool_result(symbol: str, tool_name: str, data: dict):
    path = _get_tool_path(symbol, tool_name)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        LOGGER.error(f"Failed to save tool result {tool_name} for {symbol}: {e}")

def load_tool_result(symbol: str, tool_name: str):
    path = _get_tool_path(symbol, tool_name)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        LOGGER.warning(f"Failed to load tool result {tool_name} for {symbol}: {e}")
        return None