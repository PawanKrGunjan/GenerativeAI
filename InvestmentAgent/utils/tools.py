# utils/tools.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Any, Callable

import nsepython
from langchain_core.tools import tool
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper


@dataclass(frozen=True)
class ToolBundle:
    tools: list
    tool_map: dict
    llm_with_tools: Any
    llm_choose_symbol: Callable[[str, str, list[dict]], Optional[str]]


def build_tools(
    *,
    llm: Any,
    ddg_api: DuckDuckGoSearchAPIWrapper,
    ddg_results: int,
    safe_json_loads: Callable[[str], Optional[dict]],
    safe_float: Callable[[Any], Optional[float]],
    quiet_call: Callable[..., Any],
    logger: Any = None,
) -> ToolBundle:
    """Create tool objects and return them with a name->tool map and helper functions."""

    @tool
    def search_tool(query: str) -> list[dict]:
        """Search the web using DuckDuckGo and return structured results (title/link/snippet)."""
        return ddg_api.results(query, max_results=ddg_results)

    @tool
    def llm_extract_company(query: str) -> dict:
        """Extract company mention + intent. Returns JSON: {company, intent}."""
        prompt = (
            "Extract the Indian listed company mention from the user query.\n"
            "Return ONLY JSON: {\"company\":\"...\",\"intent\":\"price|analysis|news|unknown\"}.\n"
            "Rules: company should be short (e.g., 'Larsen and Toubro', 'TCS'); "
            "exclude words like price/stock/share/today/lastprice.\n"
            f"User query: {query}"
        )
        resp = llm.invoke(prompt)
        obj = safe_json_loads(resp.content) or {}
        return {
            "company": (obj.get("company") or "").strip(),
            "intent": (obj.get("intent") or "unknown").strip(),
        }

    def llm_choose_symbol(user_query: str, extracted_company: str, candidates: list[dict]) -> Optional[str]:
        """Pick a symbol from candidates only; returns symbol or None."""
        prompt = (
            "Pick the single best matching company for the user's intent.\n"
            "You MUST choose from the provided candidates only.\n"
            "Return ONLY JSON: {\"symbol\":\"...\"} or {\"symbol\":null}.\n\n"
            f"User query: {user_query}\n"
            f"Extracted company mention: {extracted_company}\n\n"
            f"Candidates JSON:\n{json.dumps(candidates, ensure_ascii=False)}\n"
        )
        resp = llm.invoke(prompt)
        obj = safe_json_loads(resp.content) or {}
        sym = obj.get("symbol")
        allowed = {c.get("symbol") for c in candidates if isinstance(c, dict)}
        if isinstance(sym, str) and sym.upper() in allowed:
            return sym.upper()
        return None

    @tool
    def get_nse_stock_data(symbol: str) -> dict:
        """Fetch NSE equity quote via nsepython (nse_eq primary, fallback to nse_fno, then secfno lastPrice)."""
        symbol = (symbol or "").upper().strip()
        if not symbol:
            return {"error": "Empty symbol.", "symbol": symbol}

        # Optional: validate symbol against NSE equity symbols list
        try:
            eq_syms = quiet_call(nsepython.nse_eq_symbols)
            if isinstance(eq_syms, list) and eq_syms and symbol not in set(map(str.upper, eq_syms)):
                # Not necessarily fatal (could be SME / debt / etc.), but usually helps
                return {"error": f"Unknown/unsupported equity symbol: {symbol}", "symbol": symbol}
        except Exception:
            # Don't hard-fail if symbol list fetch fails
            pass

        # 1) Prefer equity quote
        data = None
        try:
            data = quiet_call(nsepython.nse_eq, symbol)
        except Exception:
            data = None

        # 2) If equity quote is missing/invalid, fallback to derivative quote
        if not isinstance(data, dict) or (isinstance(data, dict) and data.get("priceInfo") is None):
            try:
                data = quiet_call(nsepython.nse_fno, symbol)
            except Exception:
                return {"error": f"Could not fetch data for {symbol}", "symbol": symbol}

        if not isinstance(data, dict):
            return {"error": "Unexpected NSE response type", "symbol": symbol, "raw": data}

        info = (data.get("info") or {})
        meta = (data.get("metadata") or {})
        price = (data.get("priceInfo") or {})
        industry = (data.get("industryInfo") or {})

        intra = (price.get("intraDayHighLow") or {})
        week = (price.get("weekHighLow") or {})

        out = {
            "symbol": info.get("symbol") or meta.get("symbol") or symbol,
            "company": info.get("companyName", "N/A"),

            "price": safe_float(price.get("lastPrice")),
            "change": safe_float(price.get("change")),
            "p_change": safe_float(price.get("pChange")),

            "previous_close": safe_float(price.get("previousClose") or price.get("prevClose")),
            "open": safe_float(price.get("open")),

            "day_high": safe_float(intra.get("max")),
            "day_low": safe_float(intra.get("min")),
            "vwap": safe_float(price.get("vwap")),

            "52w_high": safe_float(week.get("max")),
            "52w_low": safe_float(week.get("min")),

            "pe": safe_float(meta.get("pdSymbolPe")),
            "sector_pe": safe_float(meta.get("pdSectorPe")),
            "industry": industry.get("basicIndustry") or info.get("industry") or meta.get("industry") or "N/A",
            "last_update": meta.get("lastUpdateTime") or "N/A",

            "raw": data,
        }

        # 3) Last resort: F&O “securities in F&O” lastPrice
        if out["price"] is None:
            try:
                lp = quiet_call(nsepython.nse_custom_function_secfno, symbol, "lastPrice")
                if lp is not None:
                    out["price"] = safe_float(lp)
            except Exception:
                pass

        if out["price"] is None:
            out["error"] = "Price not found in NSE response (check raw)."
            if logger:
                logger.warning("No price for %s; raw keys=%s", symbol, list((data or {}).keys()))

        return out


    # --------------------------------------------------------------------------------------------------------
    tools = [search_tool, llm_extract_company, get_nse_stock_data]
    tool_map = {t.name: t for t in tools}

    try:
        llm_with_tools = llm.bind_tools(tools)
        if logger:
            logger.info("Tools bound to LLM: %s", [t.name for t in tools])
    except Exception as e:
        llm_with_tools = llm
        if logger:
            logger.warning("Could not bind tools to LLM: %s", e)

    return ToolBundle(
        tools=tools,
        tool_map=tool_map,
        llm_with_tools=llm_with_tools,
        llm_choose_symbol=llm_choose_symbol,
    )
