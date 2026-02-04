# src/analysis/fallback.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent_state import InvestState


def python_fallback_analysis(state: "InvestState") -> list[str]:
    price = state.get("price")
    vwap = state.get("vwap")
    pe = state.get("pe")
    sector_pe = state.get("sector_pe")
    w52h = state.get("w52_high")
    w52l = state.get("w52_low")
    day_high = state.get("day_high")
    day_low = state.get("day_low")
    prev_close = state.get("previous_close")

    bullets: list[str] = []

    if price is not None and prev_close is not None and prev_close != 0:
        bullets.append(
            f"Price action: Up {((price - prev_close) / prev_close * 100):.2f}% vs previous close (short-term sentiment check)."
        )

    if price is not None and vwap is not None and vwap != 0:
        bullets.append(
            f"Intraday strength: Price is {((price - vwap) / vwap * 100):.2f}% vs VWAP (above VWAP = stronger intraday demand)."
        )

    if (
        day_high is not None
        and day_low is not None
        and price is not None
        and day_high > day_low
    ):
        pos = (price - day_low) / (day_high - day_low) * 100
        bullets.append(
            f"Day range position: Trading around {pos:.0f}% of today’s range (near 100% = closer to day high)."
        )

    if price is not None and w52h is not None and w52h != 0:
        bullets.append(
            f"52W context: {((price - w52h) / w52h * 100):.2f}% from 52-week high (near 0% = near yearly high, momentum but higher pullback risk)."
        )

    if price is not None and w52l is not None and w52l != 0:
        bullets.append(
            f"Downside context: {((price - w52l) / w52l * 100):.2f}% above 52-week low (bigger distance usually means better long-term cushion)."
        )

    if pe is not None and sector_pe is not None and sector_pe != 0:
        prem = (pe - sector_pe) / sector_pe * 100
        tag = "premium" if prem > 0 else "discount"
        bullets.append(
            f"Valuation quick check: PE {pe:.2f} vs sector PE {sector_pe:.2f} ({abs(prem):.1f}% {tag}); premium often implies higher growth/quality expectations."
        )

    if not bullets:
        bullets.append(
            "Data quality: Not enough fields to compute derived signals; verify data source or try again later."
        )

    bullets.append("This is for education/information, not a buy/sell recommendation.")
    return bullets
