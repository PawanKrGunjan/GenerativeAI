# stock_agent.py
import os
import re
import json
import uuid
import sqlite3
import difflib
from typing import Optional, Annotated, Literal, Any, cast

from dotenv import load_dotenv
from typing_extensions import TypedDict, NotRequired

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.globals import set_debug,  set_verbose
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.callbacks.stdout import StdOutCallbackHandler

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig

from pprint import pprint
from utils.logger_config import setup_logger

from utils import build_tools, safe_float, safe_json_loads, quiet_call


# --------------------------
# Env + Logging
# --------------------------
load_dotenv()

LOGS_DIR = os.getenv("LOGS_DIR", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logger = setup_logger(
    debug_mode=True,
    log_name="InventmentAgent.log",
    log_dir=LOGS_DIR,
)


# --------------------------
# Config
# --------------------------
class Config:
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL","granite4:350m") # "llama3.1:8b")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
    DDG_RESULTS = int(os.getenv("DDG_RESULTS", "5"))
    CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "data/invest_agent_checkpoints.sqlite")
    MASTER_STOCK_FILE = os.getenv("MASTER_STOCK_FILE", "data/all_nse_stocks.json")

    DEBUG = os.getenv("AGENT_DEBUG", "0") == "1"   # 1=ON, 0=OFF


set_debug(Config.DEBUG)      # bool [web:155]
set_verbose(Config.DEBUG)    # bool [web:143]

llm = ChatOllama(
    model=Config.OLLAMA_MODEL,
    temperature=Config.TEMPERATURE,
    verbose=Config.DEBUG,    # bool [web:162]
)


ddg_api = DuckDuckGoSearchAPIWrapper()


# --------------------------
# Load NSE master
# --------------------------
with open(Config.MASTER_STOCK_FILE, "r", encoding="utf-8") as f:
    NSE_STOCKS: list[dict[str, str]] = json.load(f)

SYMBOL_INDEX: dict[str, dict[str, str]] = {row["symbol"].upper(): row for row in NSE_STOCKS}
NAME_INDEX: list[tuple[str, str]] = [
    (row["symbol"].upper(), row["companyName"].lower()) for row in NSE_STOCKS
]


# --------------------------
# State
# --------------------------
class StockState(TypedDict):
    symbol: str
    company: str

    price: Optional[float]
    change: Optional[float]
    p_change: Optional[float]
    previous_close: Optional[float]
    open: Optional[float]
    day_high: Optional[float]
    day_low: Optional[float]
    vwap: Optional[float]
    w52_high: Optional[float]
    w52_low: Optional[float]

    pe: NotRequired[Optional[float]]
    sector_pe: NotRequired[Optional[float]]
    industry: NotRequired[str]
    last_update: NotRequired[str]

    raw: NotRequired[dict]
    error: NotRequired[str]

class InvestState(StockState):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str

    news: NotRequired[list[dict]]
    final: NotRequired[dict]

    resolved_from: NotRequired[str]
    symbol_candidates: NotRequired[list[str]]

    intent: NotRequired[str]                # "single", "compare", "industry"
    symbols: NotRequired[list[str]]         # <= NEW (for compare)
    quotes: NotRequired[dict[str, dict]]    # <= NEW (symbol -> quote dict)
    news_map: NotRequired[dict[str, list]]  # <= optional


def init_state(query: str) -> InvestState:
    return {
        "query": query,
        "messages": [HumanMessage(content=query)],

        "symbol": "",
        "company": "N/A",

        "price": None,
        "change": None,
        "p_change": None,
        "previous_close": None,
        "open": None,
        "day_high": None,
        "day_low": None,
        "vwap": None,
        "w52_high": None,
        "w52_low": None,
    }


# --------------------------
# Bundle tools
# --------------------------
bundle = build_tools(
    llm=llm,
    ddg_api=ddg_api,
    ddg_results=Config.DDG_RESULTS,
    safe_json_loads=safe_json_loads,
    safe_float=safe_float,
    quiet_call=quiet_call,
    logger=logger,
)

search_tool = bundle.tool_map["search_tool"]
llm_extract_company = bundle.tool_map["llm_extract_company"]
get_nse_stock_data = bundle.tool_map["get_nse_stock_data"]
llm_with_tools = bundle.llm_with_tools
llm_choose_symbol = bundle.llm_choose_symbol


# --------------------------
# Helpers (agent-side)
# --------------------------
def _has_any_price(d: dict) -> bool:
    return any(d.get(k) is not None for k in ("price", "open", "day_high", "day_low"))

def rank_symbols(company: str, limit: int = 8) -> list[tuple[str, float, str]]:
    """Return [(symbol, score_0_to_1, companyName), ...] using difflib ratio."""
    q = (company or "").strip().lower()
    if not q:
        return []

    scored: list[tuple[str, float, str]] = []
    sm = difflib.SequenceMatcher()
    sm.set_seq2(q)

    for sym, name_lower in NAME_INDEX:
        sm.set_seq1(name_lower)
        score = sm.ratio()
        scored.append((sym, score, SYMBOL_INDEX[sym]["companyName"]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]

def build_final_error(state: InvestState, message: str) -> dict:
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

def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def python_fallback_analysis(state: InvestState) -> list[str]:
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

    if price is not None and prev_close:
        bullets.append(
            f"Price action: Up {((price - prev_close)/prev_close*100):.2f}% vs previous close (short-term sentiment check)."
        )

    if price is not None and vwap:
        bullets.append(
            f"Intraday strength: Price is {((price - vwap)/vwap*100):.2f}% vs VWAP (above VWAP = stronger intraday demand)."
        )

    if day_high and day_low and price is not None and (day_high > day_low):
        pos = (price - day_low) / (day_high - day_low) * 100
        bullets.append(
            f"Day range position: Trading around {pos:.0f}% of today’s range (near 100% = closer to day high)."
        )

    if price is not None and w52h:
        bullets.append(
            f"52W context: {((price - w52h)/w52h*100):.2f}% from 52-week high (near 0% = near yearly high, momentum but higher pullback risk)."
        )

    if price is not None and w52l:
        bullets.append(
            f"Downside context: {((price - w52l)/w52l*100):.2f}% above 52-week low (bigger distance usually means better long-term cushion)."
        )

    if pe and sector_pe:
        prem = (pe - sector_pe) / sector_pe * 100
        tag = "premium" if prem > 0 else "discount"
        bullets.append(
            f"Valuation quick check: PE {pe:.2f} vs sector PE {sector_pe:.2f} ({abs(prem):.1f}% {tag}); premium often implies higher growth/quality expectations."
        )

    if not bullets:
        bullets.append("Data quality: Not enough fields to compute derived signals; verify data source or try again later.")

    bullets.append("This is for education/information, not a buy/sell recommendation.")
    return bullets

def extract_symbols_from_query(q: str) -> list[str]:
    if not q:
        return []

    text = q.upper()

    # Normalize common L&T spellings to LT
    text = re.sub(r"\bL\s*&\s*T\b", "LT", text)
    text = re.sub(r"\bL\s+AND\s+T\b", "LT", text)

    # Try direct ticker detection first
    found: list[str] = []
    for tok in re.findall(r"[A-Z0-9]+", text):
        if tok in SYMBOL_INDEX and tok not in found:
            found.append(tok)

    if len(found) >= 2:
        return found

    # If not enough tickers, split into parts and try fuzzy match to company names
    parts = re.split(r"\bVS\b|\bVERSUS\b|,|&|\band\b", text, flags=re.IGNORECASE)
    for part in parts:
        part = part.strip()
        if not part or part in {"COMPARE", "ANALYZE", "ANALYSIS"}:
            continue

        # If user typed a ticker in a chunk
        if part in SYMBOL_INDEX and part not in found:
            found.append(part)
            continue

        # Try company-name fuzzy match using your existing rank_symbols
        ranked = rank_symbols(part, limit=1)
        if ranked:
            sym = ranked[0][0]
            if sym not in found:
                found.append(sym)

    return found


# --------------------------
# Prompt
# --------------------------
SYSTEM_TEXT = (
    "You are an India-focused investment information & analysis assistant.\n"
    "Use the provided NSE market data and web news.\n"
    "Do NOT claim SEBI registration. Do NOT promise returns.\n"
    "Return ONLY valid JSON with keys: input, tool_called, output.\n"
    "output must include: symbol, company, market_data, news_summary, analysis, risks, action_items, disclaimers.\n"
    "analysis/risks/action_items must be plain strings or arrays of strings (not JSON inside a string).\n"
)

investor_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEXT),
    ("human",
     "User input: {query}\n"
     "Resolved symbol: {symbol}\n"
     "Company: {company}\n\n"
     "Market data JSON:\n{stock_json}\n\n"
     "News JSON:\n{news_json}\n"),
])


# --------------------------
# Nodes
# --------------------------
def resolve_symbol_node(state: InvestState) -> dict:
    q = state.get("query", "") or ""
    prev = (state.get("symbol") or "").strip().upper()

    # 0) Multi-symbol compare (e.g., "Compare TCS and L&T")
    symbols = extract_symbols_from_query(q)
    if len(symbols) >= 2:
        syms = [s.strip().upper() for s in symbols if s and s.strip()]
        # de-dup while preserving order
        seen: set[str] = set()
        syms = [s for s in syms if not (s in seen or seen.add(s))]

        first = syms[0]
        row0 = SYMBOL_INDEX.get(first, {})
        return {
            "intent": "compare",
            "symbols": syms,
            "symbol": first,  # keep backward compatibility
            "company": row0.get("companyName", "N/A"),
            "resolved_from": "multi_symbol_query",
            "symbol_candidates": syms,
            "error": None,
        }

    # 1) Single-symbol: direct ticker in text (tokens like LT, TCS)
    for tok in re.findall(r"[A-Za-z0-9]+", q.upper()):
        if tok in SYMBOL_INDEX:
            row = SYMBOL_INDEX[tok]
            return {
                "intent": "single",
                "symbols": [tok],
                "symbol": tok,
                "company": row.get("companyName", "N/A"),
                "resolved_from": "user_direct",
                "symbol_candidates": [tok],
                "error": None,
            }

    # 2) LLM extract company mention
    ex = llm_extract_company.invoke({"query": q}) or {}
    company_text = (ex.get("company") or "").strip()

    # 3) If nothing extracted: fallback to previous symbol if available
    if not company_text:
        if prev:
            row = SYMBOL_INDEX.get(prev, {})
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

    # 4) Candidate generation via difflib
    ranked = rank_symbols(company_text, limit=8)
    if not ranked:
        return {
            "error": "No symbols found in master list for that company name.",
            "resolved_from": "rank_none",
        }

    best_sym, best_score, best_name = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # 5) High-confidence auto-pick
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

    # 6) Otherwise: let LLM choose among candidates
    cand_payload = [{"symbol": s, "companyName": n, "score": sc} for (s, sc, n) in ranked]
    chosen = llm_choose_symbol(q, company_text, cand_payload)

    if chosen:
        chosen = chosen.strip().upper()
        row = SYMBOL_INDEX.get(chosen, {})
        return {
            "intent": "single",
            "symbols": [chosen],
            "symbol": chosen,
            "company": row.get("companyName", best_name),
            "resolved_from": "llm_extract + difflib + llm_rerank",
            "symbol_candidates": [c["symbol"] for c in cand_payload],
            "error": None,
        }

    # 7) Still ambiguous: ask user
    return {
        "error": "Company name is ambiguous. Please confirm the correct one (type the symbol).",
        "resolved_from": "rank_low_confidence",
        "symbol_candidates": [f"{s} - {n} ({sc:.2f})" for (s, sc, n) in ranked],
    }

def fetch_stock_node(state: InvestState) -> dict:
    if state.get("error"):
        return {}

    symbols = state.get("symbols") or []
    if len(symbols) >= 2:
        quotes: dict[str, dict] = {}
        for sym in symbols:
            sym = (sym or "").strip().upper()
            if not sym:
                continue

            logger.info("Fetching NSE data for symbol=%s", sym)
            data = get_nse_stock_data.invoke({"symbol": sym})
            if data.get("error") or not _has_any_price(data):
                quotes[sym] = {"symbol": sym, "error": data.get("error") or "Missing price fields"}
                continue

            # normalize 52w keys
            data["w52_high"] = data.get("w52_high") or data.get("52w_high")
            data["w52_low"]  = data.get("w52_low")  or data.get("52w_low")
            quotes[sym] = data

        # Also populate single-stock fields from the first symbol (so downstream nodes don’t break)
        first = symbols[0]
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

    # ---- Single symbol flow (your existing logic) ----
    sym = (state.get("symbol") or "").strip().upper()
    if not sym:
        return {"error": "Missing symbol after resolution."}

    logger.info("Fetching NSE data for symbol=%s", sym)
    data = get_nse_stock_data.invoke({"symbol": sym})

    if data.get("error"):
        return {"error": data["error"], "raw": data.get("raw", data)}
    if not _has_any_price(data):
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
        "w52_low":  data.get("w52_low")  or data.get("52w_low"),
        "pe": data.get("pe"),
        "sector_pe": data.get("sector_pe"),
        "industry": data.get("industry"),
        "last_update": data.get("last_update"),
        "raw": data.get("raw", data),
    }

def fetch_news_node(state: InvestState) -> dict:
    if state.get("error"):
        return {}

    sym = state.get("symbol") or ""
    company = state.get("company") or ""
    q = f"{sym} {company} latest news India stock"
    news = search_tool.invoke({"query": q})
    return {"news": news}

def resolve_intent_node(state: InvestState) -> dict:
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



def compute_metrics_node(state: InvestState) -> dict:
    price = state.get("price")
    vwap = state.get("vwap")
    w52h = state.get("w52_high")
    w52l = state.get("w52_low")
    pe = state.get("pe")
    spe = state.get("sector_pe")

    metrics: dict[str, Any] = {}

    if price is not None and vwap:
        metrics["pct_from_vwap"] = (price - vwap) / vwap * 100

    if price is not None and w52h:
        metrics["pct_from_52w_high"] = (price - w52h) / w52h * 100

    if price is not None and w52l:
        metrics["pct_from_52w_low"] = (price - w52l) / w52l * 100

    if pe is not None and spe:
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

def fetch_extras_node(state: InvestState) -> dict:
    company = state.get("company", "")

    bench = [
        search_tool.invoke({"query": "NIFTY 50 today change percent"}),
        search_tool.invoke({"query": "SENSEX today change percent"}),
    ]

    peers_web = search_tool.invoke({"query": f"{company} listed peers competitors India"})

    return {"benchmarks": {"raw": bench}, "peers": [], "raw": {"peers_web": peers_web}}

def python_compare_analysis(quotes: dict[str, dict]) -> list[str]:
    syms = [s for s in quotes.keys() if isinstance(quotes.get(s), dict)]
    if len(syms) < 2:
        return ["Comparison requested but less than 2 valid quotes were available."]

    a, b = syms[0], syms[1]
    qa, qb = quotes[a], quotes[b]

    def pct_from(price, ref):
        if price is None or not ref:
            return None
        return (price - ref) / ref * 100

    bullets: list[str] = []
    bullets.append(f"{a} vs {b}: Quick comparison (snapshot, not a recommendation).")

    pa, pb = qa.get("price"), qb.get("price")
    bullets.append(f"Daily move: {a} {qa.get('p_change')}% vs {b} {qb.get('p_change')}%.")

    avwap = pct_from(pa, qa.get("vwap"))
    bvwap = pct_from(pb, qb.get("vwap"))
    if avwap is not None and bvwap is not None:
        leader = a if avwap > bvwap else b
        bullets.append(f"Intraday strength vs VWAP: {a} {avwap:.2f}% vs {b} {bvwap:.2f}% (stronger: {leader}).")

    a52 = pct_from(pa, qa.get("w52_high"))
    b52 = pct_from(pb, qb.get("w52_high"))
    if a52 is not None and b52 is not None:
        closer = a if abs(a52) < abs(b52) else b
        bullets.append(f"Near 52W high: {a} {a52:.2f}% vs {b} {b52:.2f}% (closer to high: {closer}).")

    ape, bpe = qa.get("pe"), qb.get("pe")
    if ape is not None and bpe is not None:
        cheaper = a if ape < bpe else b
        bullets.append(f"Valuation (PE): {a} {ape} vs {b} {bpe} (lower PE: {cheaper}).")

    bullets.append("Next step: Ask 'Compare with NIFTY' or 'Add peers' if you want benchmark/sector context.")
    return bullets

def write_answer_node(state: InvestState) -> dict:
    # 1) Error / ambiguity
    err = state.get("error")
    if err:
        final = build_final_error(state, err)
        cands = state.get("symbol_candidates") or []
        if cands:
            final["output"]["action_items"] = [
                "Type one of these symbols to confirm: " + ", ".join(cands)
            ]
        return {"final": final, "messages": [AIMessage(content=json.dumps(final, ensure_ascii=False))]}

    # 2) Canonical payload (single-symbol snapshot; still useful in compare mode as “primary”)
    stock_payload = {k: state.get(k) for k in (
        "symbol", "company", "price", "change", "p_change", "previous_close", "open",
        "day_high", "day_low", "vwap", "w52_high", "w52_low", "pe", "sector_pe",
        "industry", "last_update", "error"
    )}

    # 3) If compare data exists, prefer deterministic compare analysis and skip LLM
    quotes = state.get("quotes") or {}
    if isinstance(quotes, dict):
        usable = {k: v for k, v in quotes.items() if isinstance(k, str) and isinstance(v, dict) and not v.get("error")}
        if len(usable) >= 2:
            out: dict[str, Any] = {
                "symbol": state.get("symbol", ""),
                "company": state.get("company", "N/A"),
                "market_data": stock_payload,
                "news_summary": (state.get("news") or [])[:3],
                "analysis": python_compare_analysis(usable),
                "risks": [],
                "action_items": [],
                "disclaimers": [
                    "This is for education/information, not a recommendation to buy/sell.",
                    "Markets are risky; verify from official sources before acting.",
                ],
            }
            final = {"input": state.get("query", ""), "tool_called": ["get_nse_stock_data", "search_tool"], "output": out}
            return {"final": final, "messages": [AIMessage(content=json.dumps(final, ensure_ascii=False))]}

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
    parsed = safe_json_loads(resp.content)

    obj: dict[str, Any] = {}
    if isinstance(parsed, dict) and all(isinstance(k, str) for k in parsed):
        obj = cast(dict[str, Any], parsed)

    output_any = obj.get("output")
    out: dict[str, Any] = cast(dict[str, Any], output_any) if isinstance(output_any, dict) else {}

    # 5) Override untrusted fields
    out["symbol"] = state.get("symbol", "")
    out["company"] = state.get("company", "N/A")
    out["market_data"] = stock_payload
    out["news_summary"] = (state.get("news") or [])[:3]

    analysis_list = _ensure_list(out.get("analysis"))
    analysis_list = [str(a).strip() for a in analysis_list if str(a).strip()]
    if not analysis_list:
        analysis_list = python_fallback_analysis(state)
    out["analysis"] = analysis_list

    risks_list = _ensure_list(out.get("risks"))
    out["risks"] = [str(r).strip() for r in risks_list if str(r).strip()]

    action_list = _ensure_list(out.get("action_items"))
    out["action_items"] = [str(a).strip() for a in action_list if str(a).strip()]

    out.setdefault("disclaimers", [
        "This is for education/information, not a recommendation to buy/sell.",
        "Markets are risky; verify from official sources before acting.",
    ])

    final = {"input": state.get("query", ""), "tool_called": ["get_nse_stock_data", "search_tool"], "output": out}
    return {"final": final, "messages": [AIMessage(content=json.dumps(final, ensure_ascii=False))]}

# --------------------------
# Build graph
# --------------------------
def route_after_stock(state: InvestState) -> Literal["fetch_extras", "write_answer"]:
    return "fetch_extras" if state.get("intent") in ("compare", "industry") else "write_answer"

def build_graph():
    builder = StateGraph(InvestState)
    builder.add_node("resolve_symbol", resolve_symbol_node)
    builder.add_node("fetch_stock", fetch_stock_node)
    builder.add_node("fetch_news", fetch_news_node)
    builder.add_node("write_answer", write_answer_node)
    builder.add_node("resolve_intent", resolve_intent_node)
    builder.add_node("fetch_extras", fetch_extras_node)   # benchmarks + peers
    builder.add_node("compute_metrics", compute_metrics_node)

    builder.add_edge(START, "resolve_symbol")
    builder.add_edge("resolve_symbol", "resolve_intent")
    builder.add_edge("resolve_intent", "fetch_stock")
    builder.add_conditional_edges("fetch_stock", route_after_stock, {
        "fetch_extras": "fetch_extras",
        "write_answer": "write_answer",
    })
    builder.add_edge("fetch_extras", "compute_metrics")
    builder.add_edge("compute_metrics", "fetch_news")
    builder.add_edge("fetch_news", "write_answer")
    builder.add_edge("write_answer", END)


    conn = sqlite3.connect(Config.CHECKPOINT_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)

print("AGENT_DEBUG env =", os.getenv("AGENT_DEBUG"))
print("Config.DEBUG    =", Config.DEBUG)

graph = build_graph()
# ---- Mermaid graph output
mermaid = graph.get_graph().draw_mermaid()
print("\n================= MERMAID GRAPH (copy into Mermaid Live Editor) =================")
print(mermaid)

with open("StockAgent.mmd", "w", encoding="utf-8") as f:
    f.write(mermaid)

# Optional PNG render (uses Mermaid rendering; may require internet depending on method)
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("StockAgent.png", "wb") as f:
        f.write(png_bytes)
    print("\nSaved StockAgent.png")
except Exception as e:
    print("\nCould not render graph.png:", repr(e))

# --------------------------
# CLI rendering
# --------------------------
def render_response(obj: dict) -> str:
    out = obj.get("output", {}) if isinstance(obj, dict) else {}
    md = out.get("market_data", {}) if isinstance(out, dict) else {}

    sym = out.get("symbol") or md.get("symbol") or "N/A"
    company = out.get("company") or md.get("company") or "N/A"
    err = md.get("error") or out.get("error")

    lines = [f"Symbol  : {sym}", f"Company : {company}"]

    if err:
        lines.append(f"Error   : {err}")
        return "\n".join(lines)

    analysis = out.get("analysis") or []
    analysis = [a for a in analysis if str(a).strip()]
    lines.append("\nAnalysis:")
    for a in analysis[:8]:
        lines.append(f"- {a}")

    lines.append(f"\nKey stats: Price {md.get('price')} | %Chg {md.get('p_change')} | VWAP {md.get('vwap')}")
    lines.append(f"Open    : {md.get('open')} | PrevClose: {md.get('previous_close')}")
    lines.append(f"Day     : High {md.get('day_high')} | Low {md.get('day_low')}")

    news = out.get("news_summary") or []
    if isinstance(news, list) and news:
        lines.append("\nNews (top):")
        for i, item in enumerate(news[:3], 1):
            if isinstance(item, dict):
                title = item.get("title") or item.get("text") or item.get("result") or "N/A"
                link = item.get("link") or item.get("href") or ""
            else:
                title, link = str(item), ""
            lines.append(f"{i}. {title}")
            if link:
                lines.append(f"   {link}")

    return "\n".join(lines)

# --------------------------
# Chat loop
# --------------------------
def run_chat():
    thread_id = str(uuid.uuid4())
    handler = StdOutCallbackHandler()

    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    if Config.DEBUG:
        cfg["callbacks"] = [handler]

    print("Invest bot ready. Examples:")
    print("- HAL")
    print("- Analyze HAL stock")
    print("- Analyze Tata Consultancy stock")
    print("Type 'exit' to quit.\n")

    first = input("You: ").strip()
    if first.lower() in {"exit", "quit"}:
        return

    out = graph.invoke(init_state(first), config=cfg)
    if Config.DEBUG:
        pprint(out.get("final", {}))
    print("\nBot:\n" + render_response(out.get("final", {})))

    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        updates = init_state(user_text)
        updates["messages"] = [HumanMessage(content=user_text)]
        updates["symbol"] = (out.get("symbol") or "")

        out = graph.invoke(updates, config=cfg)

        # if Config.DEBUG:
        #     pprint(out.get("final", {}))
        print("\nBot:\n" + render_response(out.get("final", {})))

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        os.environ["AGENT_DEBUG"] = "1"

    run_chat()
