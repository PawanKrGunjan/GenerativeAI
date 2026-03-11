"""
Central tool registry for LangGraph agent system
"""

# -----------------------------
# Search Tools
# -----------------------------
from tools.search_tools import lookup_stock_symbol
from tools.get_index_symbol import get_index_symbol

# -----------------------------
# Market Tools
# -----------------------------
from tools.market_tools import (
    get_current_stock_price,
    download_price_history
)

from tools.get_52_week_range import get_52_week_range
from tools.get_market_sentiment import get_market_sentiment
from tools.get_sector_performance import get_sector_performance

# -----------------------------
# Analysis Tools
# -----------------------------
from tools.analysis_tools import (
    compute_technical_indicators,
    get_top_movers,
    compare_stock_returns,
    predict_stock_trend,
    market_breadth
)

# -----------------------------
# Portfolio Tools
# -----------------------------
from tools.portfolio_tools import get_user_portfolio

# -----------------------------
# News Tools
# -----------------------------
from tools.news_tools import search_recent_news


# ======================================================
# TOOL GROUPS
# ======================================================

SEARCH_TOOLS = [
    lookup_stock_symbol,
    get_index_symbol,
]

MARKET_TOOLS = [
    get_current_stock_price,
    download_price_history,
    get_52_week_range,
    get_market_sentiment,
    get_sector_performance,
]

ANALYSIS_TOOLS = [
    compute_technical_indicators,
    get_top_movers,
    compare_stock_returns,
    predict_stock_trend,
    market_breadth,
]

PORTFOLIO_TOOLS = [
    get_user_portfolio,
]

NEWS_TOOLS = [
    search_recent_news,
]


# ======================================================
# AGENT TOOL ACCESS
# ======================================================

AGENT_TOOL_MAP = {
    "search_agent": SEARCH_TOOLS,
    "market_agent": MARKET_TOOLS,
    "analysis_agent": ANALYSIS_TOOLS,
    "portfolio_agent": PORTFOLIO_TOOLS,
    "news_agent": NEWS_TOOLS,
}
# ======================================================
# ALL TOOLS (used by LangGraph agent)
# ======================================================

TOOLS = (
    SEARCH_TOOLS
    + MARKET_TOOLS
    + ANALYSIS_TOOLS
    + PORTFOLIO_TOOLS
    + NEWS_TOOLS
)