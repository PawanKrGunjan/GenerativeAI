# src/nodes/resolve_symbol.py
from __future__ import annotations

import re
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent_state import InvestState


def make_resolve_symbol_node(
    *,
    symbol_index: dict[str, dict[str, str]],
    extract_symbols_from_query: Callable[[str], list[str]],
    rank_symbols: Callable[[str, int], list[tuple[str, float, str]]],
    llm_extract_company: Any,   # tool-like: .invoke({"query": q}) -> dict
    llm_choose_symbol: Callable[[str, str, list[dict[str, Any]]], str | None],
) -> Callable[["InvestState"], dict[str, Any]]:
    def resolve_symbol_node(state: "InvestState") -> dict[str, Any]:
        q = state.get("query", "") or ""
        prev = (state.get("symbol") or "").strip().upper()

        # 0) Multi-symbol compare
        symbols = extract_symbols_from_query(q)
        if len(symbols) >= 2:
            syms = [s.strip().upper() for s in symbols if s and s.strip()]
            seen: set[str] = set()
            syms = [s for s in syms if not (s in seen or seen.add(s))]

            first = syms[0]
            row0 = symbol_index.get(first, {})
            return {
                "intent": "compare",
                "symbols": syms,
                "symbol": first,
                "company": row0.get("companyName", "N/A"),
                "resolved_from": "multi_symbol_query",
                "symbol_candidates": syms,
                "error": None,
            }

        # 1) Single-symbol direct ticker
        for tok in re.findall(r"[A-Za-z0-9]+", q.upper()):
            if tok in symbol_index:
                row = symbol_index[tok]
                return {
                    "intent": "single",
                    "symbols": [tok],
                    "symbol": tok,
                    "company": row.get("companyName", "N/A"),
                    "resolved_from": "user_direct",
                    "symbol_candidates": [tok],
                    "error": None,
                }

        # 2) Extract company text via tool/LLM
        ex = llm_extract_company.invoke({"query": q}) or {}
        company_text = (ex.get("company") or "").strip()

        # 3) Fallback to previous
        if not company_text:
            if prev:
                row = symbol_index.get(prev, {})
                return {
                    "intent": "single",
                    "symbols": [prev],
                    "symbol": prev,
                    "company": row.get("companyName", state.get("company", "N/A")),
                    "resolved_from": "fallback_previous",
                    "symbol_candidates": [prev],
                    "error": None,
                }
            return {
                "error": "I couldn't identify the company name. Please type the NSE ticker (e.g., LT, TCS).",
                "resolved_from": "no_company_extracted",
            }

        # 4) difflib ranking
        ranked = rank_symbols(company_text, 8)
        if not ranked:
            return {"error": "No symbols found in master list for that company name.", "resolved_from": "rank_none"}

        best_sym, best_score, best_name = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # 5) High confidence pick
        min_score = 0.78
        min_gap = 0.06
        if best_score >= min_score and (best_score - second_score) >= min_gap:
            return {
                "intent": "single",
                "symbols": [best_sym],
                "symbol": best_sym,
                "company": best_name,
                "resolved_from": "llm_extract + difflib_high_conf",
                "symbol_candidates": [best_sym],
                "error": None,
            }

        # 6) LLM choose among candidates
        cand_payload = [{"symbol": s, "companyName": n, "score": sc} for (s, sc, n) in ranked]
        chosen = llm_choose_symbol(q, company_text, cand_payload)

        if chosen:
            chosen = chosen.strip().upper()
            row = symbol_index.get(chosen, {})
            return {
                "intent": "single",
                "symbols": [chosen],
                "symbol": chosen,
                "company": row.get("companyName", best_name),
                "resolved_from": "llm_extract + difflib + llm_rerank",
                "symbol_candidates": [c["symbol"] for c in cand_payload],
                "error": None,
            }

        # 7) Ask user
        return {
            "error": "Company name is ambiguous. Please confirm the correct one (type the symbol).",
            "resolved_from": "rank_low_confidence",
            "symbol_candidates": [f"{s} - {n} ({sc:.2f})" for (s, sc, n) in ranked],
        }

    return resolve_symbol_node
