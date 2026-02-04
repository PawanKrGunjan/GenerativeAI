# src/graph.py
from __future__ import annotations

import sqlite3
from typing import Callable, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent_state import InvestState


def route_after_stock(state: InvestState) -> Literal["fetch_extras", "write_answer"]:
    # Only run extras for compare/industry (you can also gate on detail_level).
    return "fetch_extras" if state.get("intent") in ("compare", "industry") else "write_answer"


def build_graph(
    *,
    checkpoint_db: str,
    resolve_symbol_node: Callable[[InvestState], dict],
    resolve_intent_node: Callable[[InvestState], dict],
    fetch_stock_node: Callable[[InvestState], dict],
    fetch_extras_node: Callable[[InvestState], dict],
    compute_metrics_node: Callable[[InvestState], dict],
    fetch_news_node: Callable[[InvestState], dict],
    write_answer_node: Callable[[InvestState], dict],
):
    builder = StateGraph(InvestState)

    builder.add_node("resolve_symbol", resolve_symbol_node)
    builder.add_node("resolve_intent", resolve_intent_node)
    builder.add_node("fetch_stock", fetch_stock_node)
    builder.add_node("fetch_extras", fetch_extras_node)
    builder.add_node("compute_metrics", compute_metrics_node)
    builder.add_node("fetch_news", fetch_news_node)
    builder.add_node("write_answer", write_answer_node)

    builder.add_edge(START, "resolve_symbol")
    builder.add_edge("resolve_symbol", "resolve_intent")
    builder.add_edge("resolve_intent", "fetch_stock")

    builder.add_conditional_edges(
        "fetch_stock",
        route_after_stock,
        {
            "fetch_extras": "fetch_extras",
            "write_answer": "write_answer",
        },
    )

    builder.add_edge("fetch_extras", "compute_metrics")
    builder.add_edge("compute_metrics", "fetch_news")
    builder.add_edge("fetch_news", "write_answer")
    builder.add_edge("write_answer", END)

    conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)
