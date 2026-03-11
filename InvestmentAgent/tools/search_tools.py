"""
Tools for stock discovery and lookup.
"""

from typing import Dict, Any
from difflib import get_close_matches
from langchain_core.tools import tool

from utils.stock_cache import load_stock_cache
from utils.logger import LOGGER


@tool
def lookup_stock_symbol(company_name_or_keyword: str) -> Dict[str, Any]:
    """
    Lookup NSE stock symbol using company name or keyword.

    This tool searches cached NSE stock data using:
    - Exact symbol match
    - Substring company match
    - Fuzzy company match

    Args:
        company_name_or_keyword: Company name or keyword.

    Returns:
        Dict containing:
            status: success | error
            matches: list of possible stocks
    """

    LOGGER.info("lookup_stock_symbol called | query=%s", company_name_or_keyword)

    try:
        cache = load_stock_cache(LOGGER)

        keyword = company_name_or_keyword.lower().strip()

        rows = cache["rows"]
        names_pool = cache["names_pool"]

        matches = []

        for r in rows:
            if keyword in r["company_name"].lower():
                matches.append(r)

        if not matches:
            close = get_close_matches(keyword, names_pool, n=5)

            for name in close:
                matches.append(cache["name_index"][name])

        return {
            "status": "success",
            "matches": matches[:5]
        }

    except Exception as e:
        LOGGER.exception("lookup_stock_symbol failed")

        return {
            "status": "error",
            "message": str(e)
        }