"""
Tools for stock discovery and lookup (NSE-focused).
Provides functionality to map company names to NSE/BSE trading symbols.
"""

from typing import List, Dict
from langchain_core.tools import tool
from utils.db_connect import get_connection
from utils.logger import LOGGER
import re
import unicodedata


def normalize_company_name(name: str) -> str:
    """
    Normalize a company name for reliable fuzzy database matching.

    This function handles common issues in company names:
    - Unicode artifacts from LLMs or copy-paste
    - Ampersand and "and" variants
    - Excessive punctuation and whitespace

    Args:
        name: Raw company name string (may contain typos, unicode issues, etc.)

    Returns:
        Cleaned and normalized company name (empty string if input is empty/invalid)

    Examples:
        >>> normalize_company_name("Larsen Æ Toubro Ltd.")
        'Larsen & Toubro Ltd'
        >>> normalize_company_name("Tata Motors  limited")
        'Tata Motors limited'
        >>> normalize_company_name("H D F C   Bank")
        'H D F C Bank'
        >>> normalize_company_name("")
        ''
    """
    if not name or not name.strip():
        return ""

    # Decompose combined unicode characters
    name = unicodedata.normalize("NFKD", name)

    # Replace common variants and corruptions
    replacements = {
        "Æ": "&", "æ": "&",
        "＆": "&",
        " & ": " & ",
        " and ": " & ",
        " AND ": " & ",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)

    # Retain only alphanumeric + safe separators
    name = re.sub(r"[^a-zA-Z0-9\s&.-]", " ", name)

    # Collapse multiple spaces and trim
    name = re.sub(r"\s{2,}", " ", name).strip()

    return name


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a user-provided stock symbol to NSE format.

    Examples
    --------
    RELIANCE -> RELIANCE.NS
    INFY.NS  -> INFY.NS
    ^NSEI    -> ^NSEI
    """

    if not symbol:
        return symbol

    symbol = symbol.strip().upper()

    if symbol.startswith("^"):
        return symbol

    if symbol.endswith(".NS"):
        return symbol

    return f"{symbol}.NS"

@tool
def lookup_stock_symbol(company_name: str) -> List[Dict[str, str]]:
    """
    Look up NSE stock symbol(s) from a company name (or partial name).

    This is a **critical first step** in the investment agent workflow.
    It should be called whenever the user refers to a company by name rather
    than providing a ticker/symbol directly.

    The function uses fuzzy matching (ILIKE + trigram indexes) to find matches
    and returns results ordered by relevance (exact > prefix > partial).

    Args:
        company_name: Company name or fragment (e.g. "Reliance", "HDFC Bank", "L&T")

    Returns:
        List of matching companies. Each item is a dictionary with:
            - "company_name": Full official company name from database
            - "symbol": NSE trading symbol (e.g. "RELIANCE", "HDFCBANK")

        Return formats:
        - Successful matches: [{"company_name": "...", "symbol": "..."}, ...]
        - No matches: []
        - Database error (rare): [{"status": "error", "message": "..."}]

    Examples:
        >>> lookup_stock_symbol("Larsen & Toubro")
        [{'company_name': 'Larsen & Toubro Ltd.', 'symbol': 'LT'}]

        >>> lookup_stock_symbol("infosys")
        [
            {'company_name': 'Infosys Ltd.', 'symbol': 'INFY'},
            {'company_name': 'HCL Infosystems Limited', 'symbol': 'HCL-INSYS'}
        ]

        >>> lookup_stock_symbol("xyz non existent company")
        []

    Important:
        - Always use the returned symbol(s) in subsequent tools (price, history, etc.)
        - Do NOT pass raw company names directly to market data tools
    """
    if not company_name or not company_name.strip():
        LOGGER.warning("lookup_stock_symbol called with empty/blank company name")
        return []

    normalized = normalize_company_name(company_name)
    LOGGER.info("lookup_stock_symbol | original=%r normalized=%r", company_name, normalized)

    if len(normalized) < 2:
        LOGGER.debug("Query too short after normalization: %s", normalized)
        return []

    try:
        sql = """
            SELECT symbol, company_name
            FROM nse_stocks
            WHERE company_name ILIKE %s
               OR symbol      ILIKE %s
            ORDER BY
                CASE WHEN company_name ILIKE %s THEN 0
                     WHEN company_name ILIKE %s THEN 1
                     ELSE 2 END,
                LENGTH(company_name),
                company_name
            LIMIT 6
        """

        pattern = f"%{normalized}%"
        prefix = f"{normalized}%"

        with get_connection(LOGGER) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (pattern, pattern, normalized, prefix))
                rows = cur.fetchall()

        if not rows:
            LOGGER.debug("No matches found for normalized query: %s", normalized)
            return []

        matches = [
            {"company_name": row["company_name"], "symbol": normalize_symbol(symbol= row["symbol"])}
            for row in rows
        ]

        LOGGER.debug("Found %d matches for '%s'", len(matches), normalized)
        return matches

    except Exception as e:
        LOGGER.exception("lookup_stock_symbol failed | query=%r", company_name)
        return [{"status": "error", "message": str(e)}]


# ────────────────────────────────────────────────
# Quick local test (can be removed or moved to tests/)
# ────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        "Larsen Æ Toubro",
        "Tata motors ltd",
        "infosys",
        "hdfC bAnk",
        "reliance",
        "l&t",
        "xyzabc nonexistent",
        "   ",
    ]

    for name in test_cases:
        print(f"\nQuery: {name!r}")
        result = lookup_stock_symbol.invoke({"company_name": name})
        print(result)