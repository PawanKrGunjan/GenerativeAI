from langchain_core.tools import tool
from typing import Dict, Any
from utils.logger import LOGGER


@tool
def get_index_symbol(index_name: str) -> Dict[str, Any]:
    """
    Resolve common Indian index names to Yahoo Finance tickers.

    Args:
        index_name: Example: "Nifty 50", "Sensex", "Bank Nifty"

    Returns:
        Dict with ticker if found.
    """

    LOGGER.info("get_index_symbol | index_name=%s", index_name)

    try:

        key = (
            index_name.lower()
            .strip()
            .replace("-", "")
            .replace("_", "")
        )

        key = "".join(key.split())

        mapping = {
            "nifty": "^NSEI",
            "nifty50": "^NSEI",
            "niftyfifty": "^NSEI",

            "sensex": "^BSESN",
            "bsesensex": "^BSESN",

            "banknifty": "^NSEBANK",
            "niftybank": "^NSEBANK",

            "niftymidcap": "^NSEMDCP50",
            "niftymidcap50": "^NSEMDCP50",

            "niftynext50": "^NSMIDCP",

            "niftysmallcap": "^CNXSMALL",
        }

        ticker = mapping.get(key)

        if ticker:
            return {
                "status": "found",
                "ticker": ticker,
                "requested": index_name
            }

        return {
            "status": "not_found",
            "ticker": None,
            "requested": index_name
        }

    except Exception as e:

        LOGGER.exception("get_index_symbol failed")

        return {
            "status": "error",
            "message": str(e)
        }