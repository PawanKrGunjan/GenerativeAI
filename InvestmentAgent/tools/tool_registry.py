"""
agents/tool_registry.py
Central tool registry for LangGraph agent system
"""

# ======================================================
# SEARCH TOOLS
# ======================================================

from tools.search_tools import lookup_stock_symbol
from tools.get_index_symbol import get_index_symbol


# ======================================================
# MARKET TOOLS
# ======================================================

from tools.market_tools import (
    get_stock_info,
    get_price_history,
)

from tools.get_market_sentiment import get_nifty50_market_sentiment
from tools.get_sector_performance import get_sector_performance


# ======================================================
# ANALYSIS TOOLS
# ======================================================

from tools.analysis_tools import (
    compute_technical_indicators,
    get_top_movers,
    compare_stock_returns,
    predict_stock_trend,
    market_breadth,
)


# ======================================================
# PORTFOLIO TOOLS
# ======================================================

from tools.portfolio_tools import get_user_portfolio


# ======================================================
# NEWS TOOLS
# ======================================================

from tools.news_tools import search_recent_news


# ======================================================
# TOOL GROUPS
# ======================================================

SEARCH_TOOLS = [
    lookup_stock_symbol,
    get_index_symbol,
]

MARKET_TOOLS = [
    get_stock_info,
    get_price_history,
    get_nifty50_market_sentiment,
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
# ALL TOOLS
# ======================================================

TOOLS = (
    SEARCH_TOOLS
    + MARKET_TOOLS
    + ANALYSIS_TOOLS
    + PORTFOLIO_TOOLS
    + NEWS_TOOLS
)


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
# TOOL REGISTRY
# ======================================================

class ToolRegistry:
    """
    Central registry for managing tools by name and category.
    """

    def __init__(self):
        self.tools = {}
        self.categories = {}

    def register(self, category: str, tool):

        self.tools[tool.name] = tool

        if category not in self.categories:
            self.categories[category] = []

        self.categories[category].append(tool)

    def register_many(self, category: str, tools):

        for tool in tools:
            self.register(category, tool)

    def get_tool(self, name: str):
        return self.tools.get(name)

    def get_tools_by_category(self, category: str):
        return self.categories.get(category, [])

    def get_all_tools(self):
        return list(self.tools.values())


# ======================================================
# INITIALIZE REGISTRY
# ======================================================

ToolRegister = ToolRegistry()

ToolRegister.register_many("search", SEARCH_TOOLS)
ToolRegister.register_many("market", MARKET_TOOLS)
ToolRegister.register_many("analysis", ANALYSIS_TOOLS)
ToolRegister.register_many("portfolio", PORTFOLIO_TOOLS)
ToolRegister.register_many("news", NEWS_TOOLS)