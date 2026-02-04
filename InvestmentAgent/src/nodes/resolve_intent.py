# src/nodes/resolve_intent.py
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent_state import InvestState


def resolve_intent_node(state: "InvestState") -> dict[str, Any]:
    q = (state.get("query") or "").lower()

    # intent
    if any(k in q for k in ("industry", "sector", "peers")):
        intent = "industry"
    elif any(k in q for k in ("compare", "vs", "versus")) or ("nifty" in q) or ("sensex" in q):
        intent = "compare"
    else:
        intent = "single"

    # detail level
    detail_level = "detailed" if any(k in q for k in ("detailed", "deep", "full", "in-depth")) else "normal"

    return {"intent": intent, "detail_level": detail_level}
