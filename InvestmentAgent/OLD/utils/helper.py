# utils/helper.py
from __future__ import annotations

import io
import json
import re
import contextlib
import difflib
from typing import Any, Optional, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    # Only for type hints; avoids runtime circular imports. [web:267]
    from typing import Mapping
    from stock_agent import InvestState  # or: from .types import InvestState (recommended)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def safe_json_loads(text: str) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def quiet_call(fn: Callable[..., Any], *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def has_any_price(d: dict[str, Any]) -> bool:
    return any(d.get(k) is not None for k in ("price", "open", "day_high", "day_low"))


def rank_symbols(
    company: str,
    name_index: Iterable[tuple[str, str]],         # [(SYMBOL, companyNameLower), ...]
    symbol_index: dict[str, dict[str, str]],       # {SYMBOL: {"companyName": ...}, ...}
    limit: int = 8,
) -> list[tuple[str, float, str]]:
    q = (company or "").strip().lower()
    if not q:
        return []

    scored: list[tuple[str, float, str]] = []
    sm = difflib.SequenceMatcher()
    sm.set_seq2(q)

    for sym, name_lower in name_index:
        sm.set_seq1(name_lower)
        score = sm.ratio()
        scored.append((sym, score, symbol_index[sym]["companyName"]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def ensure_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def extract_symbols_from_query(
    q: str,
    symbol_index: dict[str, dict[str, str]],
    *,
    company_ranker: Optional[Callable[[str], list[tuple[str, float, str]]]] = None,
) -> list[str]:
    if not q:
        return []

    text = q.upper()

    # Normalize L&T variants first
    text = re.sub(r"\bL\s*&\s*T\b", " LT ", text)
    text = re.sub(r"\bL\s+AND\s+T\b", " LT ", text)

    found: list[str] = []
    for tok in re.findall(r"[A-Z0-9]+", text):
        if tok in symbol_index and tok not in found:
            found.append(tok)

    if len(found) >= 2:
        return found

    if company_ranker is None:
        return found

    parts = re.split(r"\bVS\b|\bVERSUS\b|,|&|\band\b", text, flags=re.IGNORECASE)
    for part in parts:
        part = part.strip()
        if not part or part in {"COMPARE", "ANALYZE", "ANALYSIS"}:
            continue

        if part in symbol_index and part not in found:
            found.append(part)
            continue

        ranked = company_ranker(part)
        if ranked:
            sym = ranked[0][0]
            if sym not in found:
                found.append(sym)

    return found


def build_final_error(state: "InvestState", message: str) -> dict[str, Any]:
    # Keep this here if you want, but it depends on InvestState shape.
    return {
        "input": state.get("query", ""),
        "tool_called": [],
        "output": {
            "symbol": state.get("symbol", ""),
            "company": state.get("company", "N/A"),
            "market_data": {
                "symbol": state.get("symbol", ""),
                "company": state.get("company", "N/A"),
                "error": message,
            },
            "news_summary": [],
            "analysis": [],
            "risks": [],
            "action_items": [],
            "disclaimers": [
                "This is for education/information, not a recommendation to buy/sell.",
                "Markets are risky; verify from official sources before acting.",
            ],
        },
    }
