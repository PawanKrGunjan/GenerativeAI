# src/nodes/fetch_stock.py
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent_state import InvestState


def make_fetch_stock_node(
    *,
    get_nse_stock_data: Any,                     # tool-like: .invoke({"symbol": sym}) -> dict
    has_any_price: Callable[[dict[str, Any]], bool],
    logger: Any,                                 # logger-like: .info(...)
) -> Callable[["InvestState"], dict[str, Any]]:
    def fetch_stock_node(state: "InvestState") -> dict[str, Any]:
        if state.get("error"):
            return {}

        symbols = state.get("symbols") or []
        if len(symbols) >= 2:
            quotes: dict[str, dict[str, Any]] = {}

            for sym in symbols:
                sym = (sym or "").strip().upper()
                if not sym:
                    continue

                logger.info("Fetching NSE data for symbol=%s", sym)
                data = get_nse_stock_data.invoke({"symbol": sym}) or {}

                if data.get("error") or not has_any_price(data):
                    quotes[sym] = {
                        "symbol": sym,
                        "error": data.get("error") or "Missing price fields",
                    }
                    continue

                # Normalize 52W key variants
                data["w52_high"] = data.get("w52_high") or data.get("52w_high")
                data["w52_low"] = data.get("w52_low") or data.get("52w_low")
                quotes[sym] = data

            # Populate "primary" fields from the first symbol (backward compat)
            first = (symbols[0] or "").strip().upper()
            first_q = quotes.get(first) or {}

            return {
                "quotes": quotes,
                "symbol": first_q.get("symbol", first),
                "company": first_q.get("company", state.get("company", "N/A")),
                "price": first_q.get("price"),
                "change": first_q.get("change"),
                "p_change": first_q.get("p_change"),
                "previous_close": first_q.get("previous_close"),
                "open": first_q.get("open"),
                "day_high": first_q.get("day_high"),
                "day_low": first_q.get("day_low"),
                "vwap": first_q.get("vwap"),
                "w52_high": first_q.get("w52_high"),
                "w52_low": first_q.get("w52_low"),
                "pe": first_q.get("pe"),
                "sector_pe": first_q.get("sector_pe"),
                "industry": first_q.get("industry"),
                "last_update": first_q.get("last_update"),
                "raw": quotes,
            }

        # ---- Single-symbol flow ----
        sym = (state.get("symbol") or "").strip().upper()
        if not sym:
            return {"error": "Missing symbol after resolution."}

        logger.info("Fetching NSE data for symbol=%s", sym)
        data = get_nse_stock_data.invoke({"symbol": sym}) or {}

        if data.get("error"):
            return {"error": data["error"], "raw": data.get("raw", data)}
        if not has_any_price(data):
            return {"error": f"NSE quote missing price fields for {sym}.", "raw": data.get("raw", data)}

        return {
            "symbol": data.get("symbol", sym),
            "company": data.get("company", state.get("company", "N/A")),
            "price": data.get("price"),
            "change": data.get("change"),
            "p_change": data.get("p_change"),
            "previous_close": data.get("previous_close"),
            "open": data.get("open"),
            "day_high": data.get("day_high"),
            "day_low": data.get("day_low"),
            "vwap": data.get("vwap"),
            "w52_high": data.get("w52_high") or data.get("52w_high"),
            "w52_low": data.get("w52_low") or data.get("52w_low"),
            "pe": data.get("pe"),
            "sector_pe": data.get("sector_pe"),
            "industry": data.get("industry"),
            "last_update": data.get("last_update"),
            "raw": data.get("raw", data),
        }

    return fetch_stock_node
