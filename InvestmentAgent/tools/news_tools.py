"""
Financial news search tools.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from typing import Dict, Any

from utils.logger import LOGGER


SEARCH = DuckDuckGoSearchRun()


@tool
def search_recent_news(query: str) -> Dict[str, Any]:
    """
    Search recent financial news.

    Args:
        query: Topic or company.

    Returns:
        List of news results.
    """

    LOGGER.info("search_recent_news | query=%s", query)

    try:

        results = SEARCH.invoke(query)

        return {
            "status": "success",
            "results": results
        }

    except Exception as e:

        LOGGER.exception("search_recent_news failed")

        return {"status": "error", "message": str(e)}