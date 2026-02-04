# src/analysis/compare.py
from __future__ import annotations

from typing import Any


def python_compare_analysis(quotes: dict[str, dict[str, Any]]) -> list[str]:
    syms = [s for s in quotes.keys() if isinstance(quotes.get(s), dict)]
    if len(syms) < 2:
        return ["Comparison requested but less than 2 valid quotes were available."]

    a, b = syms[0], syms[1]
    qa, qb = quotes[a], quotes[b]

    def pct_from(price: Any, ref: Any) -> float | None:
        if price is None or ref is None:
            return None
        try:
            ref_f = float(ref)
            price_f = float(price)
        except (TypeError, ValueError):
            return None
        if ref_f == 0:
            return None
        return (price_f - ref_f) / ref_f * 100

    bullets: list[str] = []
    bullets.append(f"{a} vs {b}: Quick comparison (snapshot, not a recommendation).")

    pa, pb = qa.get("price"), qb.get("price")
    bullets.append(f"Daily move: {a} {qa.get('p_change')}% vs {b} {qb.get('p_change')}%.")

    avwap = pct_from(pa, qa.get("vwap"))
    bvwap = pct_from(pb, qb.get("vwap"))
    if avwap is not None and bvwap is not None:
        leader = a if avwap > bvwap else b
        bullets.append(
            f"Intraday strength vs VWAP: {a} {avwap:.2f}% vs {b} {bvwap:.2f}% (stronger: {leader})."
        )

    a52 = pct_from(pa, qa.get("w52_high"))
    b52 = pct_from(pb, qb.get("w52_high"))
    if a52 is not None and b52 is not None:
        closer = a if abs(a52) < abs(b52) else b
        bullets.append(f"Near 52W high: {a} {a52:.2f}% vs {b} {b52:.2f}% (closer to high: {closer}).")

    ape, bpe = qa.get("pe"), qb.get("pe")
    if ape is not None and bpe is not None:
        try:
            ape_f, bpe_f = float(ape), float(bpe)
            cheaper = a if ape_f < bpe_f else b
            bullets.append(f"Valuation (PE): {a} {ape_f:.2f} vs {b} {bpe_f:.2f} (lower PE: {cheaper}).")
        except (TypeError, ValueError):
            pass

    bullets.append("Next step: Ask 'Compare with NIFTY' or 'Add peers' if you want benchmark/sector context.")
    return bullets
