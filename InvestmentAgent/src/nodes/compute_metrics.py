# src/nodes/compute_metrics.py
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent_state import InvestState


def compute_metrics_node(state: "InvestState") -> dict[str, Any]:
    price = state.get("price")
    vwap = state.get("vwap")
    w52h = state.get("w52_high")
    w52l = state.get("w52_low")
    pe = state.get("pe")
    spe = state.get("sector_pe")

    metrics: dict[str, Any] = {}

    if price is not None and vwap is not None and vwap != 0:
        metrics["pct_from_vwap"] = (price - vwap) / vwap * 100

    if price is not None and w52h is not None and w52h != 0:
        metrics["pct_from_52w_high"] = (price - w52h) / w52h * 100

    if price is not None and w52l is not None and w52l != 0:
        metrics["pct_from_52w_low"] = (price - w52l) / w52l * 100

    if pe is not None and spe is not None and spe != 0:
        metrics["pe_premium_pct"] = (pe - spe) / spe * 100

    score = 0
    if metrics.get("pct_from_vwap", 0) > 0:
        score += 1
    if metrics.get("pe_premium_pct", 0) <= 10:
        score += 1
    if metrics.get("pct_from_52w_high", 0) < -5:
        score += 1
    metrics["score_simple_0_to_3"] = score

    return {"metrics": metrics}
