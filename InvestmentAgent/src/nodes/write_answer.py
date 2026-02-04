# src/nodes/write_answer.py
from __future__ import annotations

import json
from typing import Any, Callable, TYPE_CHECKING, cast

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from src.agent_state import InvestState


def make_write_answer_node(
    *,
    build_final_error: Callable[["InvestState", str], dict[str, Any]],
    python_compare_analysis: Callable[[dict[str, dict[str, Any]]], list[str]],
    python_fallback_analysis: Callable[["InvestState"], list[str]],
    safe_json_loads: Callable[[str], dict[str, Any] | None],
    ensure_list: Callable[[Any], list[Any]],
    investor_prompt: Any,          # ChatPromptTemplate
    llm_with_tools: Any,           # runnable: .invoke(prompt_value)
) -> Callable[["InvestState"], dict[str, Any]]:
    def write_answer_node(state: "InvestState") -> dict[str, Any]:
        # 1) Error / ambiguity
        err = state.get("error")
        if err:
            final = build_final_error(state, err)
            cands = state.get("symbol_candidates") or []
            if cands:
                final["output"]["action_items"] = [
                    "Type one of these symbols to confirm: " + ", ".join(cands)
                ]
            return {
                "final": final,
                "messages": [AIMessage(content=json.dumps(final, ensure_ascii=False))],
            }

        # 2) Canonical payload
        stock_payload = {k: state.get(k) for k in (
            "symbol", "company", "price", "change", "p_change", "previous_close", "open",
            "day_high", "day_low", "vwap", "w52_high", "w52_low", "pe", "sector_pe",
            "industry", "last_update", "error"
        )}

        # 3) Compare shortcut (deterministic)
        quotes = state.get("quotes") or {}
        if isinstance(quotes, dict):
            usable = {
                k: v for k, v in quotes.items()
                if isinstance(k, str) and isinstance(v, dict) and not v.get("error")
            }
            if len(usable) >= 2:
                out: dict[str, Any] = {
                    "symbol": state.get("symbol", ""),
                    "company": state.get("company", "N/A"),
                    "market_data": stock_payload,
                    "news_summary": (state.get("news") or [])[:3],
                    "analysis": python_compare_analysis(cast(dict[str, dict[str, Any]], usable)),
                    "risks": [],
                    "action_items": [],
                    "disclaimers": [
                        "This is for education/information, not a recommendation to buy/sell.",
                        "Markets are risky; verify from official sources before acting.",
                    ],
                }
                final = {
                    "input": state.get("query", ""),
                    "tool_called": ["get_nse_stock_data", "search_tool"],
                    "output": out,
                }
                return {
                    "final": final,
                    "messages": [AIMessage(content=json.dumps(final, ensure_ascii=False))],
                }

        # 4) Otherwise: call LLM and parse
        stock_json = json.dumps(stock_payload, ensure_ascii=False)
        news_json = json.dumps(state.get("news", []), ensure_ascii=False)

        prompt_value = investor_prompt.invoke({
            "query": state.get("query", ""),
            "symbol": state.get("symbol", ""),
            "company": state.get("company", "N/A"),
            "stock_json": stock_json,
            "news_json": news_json,
        })

        resp = llm_with_tools.invoke(prompt_value)
        parsed = safe_json_loads(getattr(resp, "content", "") or "")

        obj: dict[str, Any] = cast(dict[str, Any], parsed) if isinstance(parsed, dict) else {}
        output_any = obj.get("output")
        out: dict[str, Any] = cast(dict[str, Any], output_any) if isinstance(output_any, dict) else {}

        # 5) Override untrusted fields
        out["symbol"] = state.get("symbol", "")
        out["company"] = state.get("company", "N/A")
        out["market_data"] = stock_payload
        out["news_summary"] = (state.get("news") or [])[:3]

        analysis_list = ensure_list(out.get("analysis"))
        analysis_list = [str(a).strip() for a in analysis_list if str(a).strip()]
        if not analysis_list:
            analysis_list = python_fallback_analysis(state)
        out["analysis"] = analysis_list

        risks_list = ensure_list(out.get("risks"))
        out["risks"] = [str(r).strip() for r in risks_list if str(r).strip()]

        action_list = ensure_list(out.get("action_items"))
        out["action_items"] = [str(a).strip() for a in action_list if str(a).strip()]

        out.setdefault("disclaimers", [
            "This is for education/information, not a recommendation to buy/sell.",
            "Markets are risky; verify from official sources before acting.",
        ])

        final = {
            "input": state.get("query", ""),
            "tool_called": ["get_nse_stock_data", "search_tool"],
            "output": out,
        }
        return {
            "final": final,
            "messages": [AIMessage(content=json.dumps(final, ensure_ascii=False))],
        }

    return write_answer_node
