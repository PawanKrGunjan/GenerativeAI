from difflib import get_close_matches
from typing import Dict, Any, List

from utils.db_connect import get_connection
from psycopg.rows import dict_row

STOCK_CACHE = None


def load_stock_cache(logger) -> Dict[str, Any]:
    global STOCK_CACHE

    if STOCK_CACHE:
        return STOCK_CACHE

    with get_connection(logger) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, company_name
                FROM nse_stocks
            """)
            rows = cur.fetchall()

    symbol_index = {r["symbol"].upper(): r for r in rows}
    name_index = {r["company_name"].lower(): r for r in rows}
    names_pool = list(name_index.keys())

    STOCK_CACHE = {
        "rows": rows,
        "symbol_index": symbol_index,
        "name_index": name_index,
        "names_pool": names_pool,
    }

    logger.info("Loaded %d stocks into cache", len(rows))

    return STOCK_CACHE
