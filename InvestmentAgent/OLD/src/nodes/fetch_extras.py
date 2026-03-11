# src/nodes/fetch_extras.py
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent_state import InvestState


def make_fetch_extras_node(
    *,
    search_tool: Any,                 # tool-like: .invoke({"query": q})
    logger: Any | None = None,
) -> Callable[["InvestState"], dict[str, Any]]:
    def _safe_invoke(query: str) -> Any:
        try:
            if logger:
                logger.info("Extras search: %s", query)
            return search_tool.invoke({"query": query})
        except Exception as e:
            # don’t fail the whole graph for “extras”
            return {"error": f"extras_search_failed: {type(e).__name__}: {e}", "query": query}

    def fetch_extras_node(state: "InvestState") -> dict[str, Any]:
        if state.get("error"):
            return {}

        company = state.get("company", "") or ""

        bench = [
            _safe_invoke("NIFTY 50 today change percent"),
            _safe_invoke("SENSEX today change percent"),
        ]

        peers_web = _safe_invoke(f"{company} listed peers competitors India")

        return {
            "benchmarks": {"raw": bench},
            "peers": [],
            "raw": {"peers_web": peers_web},
        }

    return fetch_extras_node
