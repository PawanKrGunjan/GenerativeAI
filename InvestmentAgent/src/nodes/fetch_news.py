# src/nodes/fetch_news.py
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent_state import InvestState


def make_fetch_news_node(
    *,
    search_tool: Any,                              # tool-like: .invoke({"query": q}) -> list|str|dict
    logger: Any | None = None,
    query_builder: Callable[[str, str], str] | None = None,
    max_items: int = 8,
) -> Callable[["InvestState"], dict[str, Any]]:
    def _default_query_builder(sym: str, company: str) -> str:
        return f"{sym} {company} latest news India stock"

    qb = query_builder or _default_query_builder

    def _normalize_news(x: Any) -> list[dict[str, Any]]:
        # already a list of dicts
        if isinstance(x, list):
            out: list[dict[str, Any]] = []
            for item in x:
                if isinstance(item, dict):
                    out.append(item)
                else:
                    out.append({"title": str(item)})
            return out[:max_items]

        # dict -> single item
        if isinstance(x, dict):
            return [x]

        # string or other -> single item
        if x:
            return [{"title": str(x)}]
        return []

    def fetch_news_node(state: "InvestState") -> dict[str, Any]:
        if state.get("error"):
            return {}

        sym = state.get("symbol") or ""
        company = state.get("company") or ""
        q = qb(sym, company)

        if logger:
            logger.info("Searching news: %s", q)

        raw = search_tool.invoke({"query": q})
        return {"news": _normalize_news(raw)}

    return fetch_news_node
