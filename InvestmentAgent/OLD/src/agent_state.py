# src/agent_state.py
from __future__ import annotations

from typing import Optional, Annotated, Any
from typing_extensions import TypedDict, NotRequired

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


class StockState(TypedDict):
    symbol: str
    company: str

    price: Optional[float]
    change: Optional[float]
    p_change: Optional[float]
    previous_close: Optional[float]
    open: Optional[float]
    day_high: Optional[float]
    day_low: Optional[float]
    vwap: Optional[float]
    w52_high: Optional[float]
    w52_low: Optional[float]

    pe: NotRequired[Optional[float]]
    sector_pe: NotRequired[Optional[float]]
    industry: NotRequired[str]
    last_update: NotRequired[str]

    raw: NotRequired[dict[str, Any]]
    error: NotRequired[str]


class InvestState(StockState):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str

    news: NotRequired[list[dict[str, Any]]]
    final: NotRequired[dict[str, Any]]

    resolved_from: NotRequired[str]
    symbol_candidates: NotRequired[list[str]]

    intent: NotRequired[str]                     # "single", "compare", "industry"
    detail_level: NotRequired[str]               # "normal" | "detailed"
    symbols: NotRequired[list[str]]              # for compare
    quotes: NotRequired[dict[str, dict[str, Any]]]  # symbol -> quote dict
    news_map: NotRequired[dict[str, list[dict[str, Any]]]]


def init_state(query: str) -> InvestState:
    return {
        "query": query,
        "messages": [HumanMessage(content=query)],

        "symbol": "",
        "company": "N/A",

        "price": None,
        "change": None,
        "p_change": None,
        "previous_close": None,
        "open": None,
        "day_high": None,
        "day_low": None,
        "vwap": None,
        "w52_high": None,
        "w52_low": None,
    }
