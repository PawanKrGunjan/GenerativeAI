# stock_agent.py
import os
import re
import json
import uuid
import sqlite3
import difflib
from typing import Optional, Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict, NotRequired

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

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
    log_name="InventmentAgent",
    log_dir=LOGS_DIR,
)


# --------------------------
# Config
# --------------------------
class Config:
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL","granite4:350m") # "llama3.1:8b")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
    DDG_RESULTS = int(os.getenv("DDG_RESULTS", "5"))
    CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "invest_agent_checkpoints.sqlite")
    MASTER_STOCK_FILE = os.getenv("MASTER_STOCK_FILE", "all_nse_stocks.json")


llm = ChatOllama(
    model=Config.OLLAMA_MODEL,
    temperature=Config.TEMPERATURE,
    verbose=True,
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
    q = state.get("query", "")
    prev = (state.get("symbol") or "").strip().upper()

    # A) Direct ticker shortcut
    for tok in re.findall(r"[A-Za-z0-9]+", (q or "").upper()):
        if tok in SYMBOL_INDEX:
            row = SYMBOL_INDEX[tok]
            return {
                "symbol": tok,
                "company": row.get("companyName", "N/A"),
                "resolved_from": "user_direct",
                "symbol_candidates": [tok],
                "error": None,
            }

    # B) LLM extract company mention
    ex = llm_extract_company.invoke({"query": q})
    company_text = (ex.get("company") or "").strip()

    if not company_text:
        if prev:
            row = SYMBOL_INDEX.get(prev, {})
            return {
                "symbol": prev,
                "company": row.get("companyName", state.get("company", "N/A")),
                "resolved_from": "fallback_previous",
                "symbol_candidates": [prev],
                "error": None,
            }
        return {"error": "I couldn't identify the company name. Please type the NSE ticker (e.g., LT, TCS)."}

    # C) Candidate generation via difflib
    ranked = rank_symbols(company_text, limit=8)
    if not ranked:
        return {"error": "No symbols found in master list for that company name."}

    best_sym, best_score, best_name = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # D) If confident enough, accept immediately
    min_score = 0.78
    min_gap = 0.06
    if best_score >= min_score and (best_score - second_score) >= min_gap:
        return {
            "symbol": best_sym,
            "company": best_name,
            "resolved_from": "llm_extract + difflib_high_conf",
            "symbol_candidates": [best_sym],
            "error": None,
        }

    # E) Otherwise: ask LLM to choose among top candidates (constrained)
    cand_payload = [{"symbol": s, "companyName": n, "score": sc} for (s, sc, n) in ranked]
    chosen = llm_choose_symbol(q, company_text, cand_payload)

    if chosen:
        row = SYMBOL_INDEX.get(chosen, {})
        return {
            "symbol": chosen,
            "company": row.get("companyName", best_name),
            "resolved_from": "llm_extract + difflib + llm_rerank",
            "symbol_candidates": [c["symbol"] for c in cand_payload],
            "error": None,
        }

    # F) Still ambiguous: ask user
    return {
        "error": "Company name is ambiguous. Please confirm the correct one (type the symbol).",
        "resolved_from": "rank_low_confidence",
        "symbol_candidates": [f"{s} - {n} ({sc:.2f})" for (s, sc, n) in ranked],
    }


def fetch_stock_node(state: InvestState) -> dict:
    if state.get("error"):
        return {}

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
        "w52_high": data.get("52w_high"),
        "w52_low": data.get("52w_low"),
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


def write_answer_node(state: InvestState) -> dict:
    # Do not call LLM if error/ambiguity
    if state.get("error"):
        final = build_final_error(state, state["error"])
        if state.get("symbol_candidates"):
            final["output"]["action_items"] = [
                "Type one of these symbols to confirm: " + ", ".join(state["symbol_candidates"])
            ]
        return {"final": final, "messages": [AIMessage(content=json.dumps(final, ensure_ascii=False))]}

    stock_payload = {k: state.get(k) for k in (
        "symbol", "company", "price", "change", "p_change", "previous_close", "open", "day_high", "day_low",
        "vwap", "w52_high", "w52_low", "pe", "sector_pe", "industry", "last_update", "error"
    )}

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
    obj = safe_json_loads(resp.content)

    if obj is None:
        obj = {
            "input": state.get("query", ""),
            "tool_called": ["search_tool", "get_nse_stock_data"],
            "output": {
                "symbol": state.get("symbol", ""),
                "company": state.get("company", "N/A"),
                "market_data": stock_payload,
                "news_summary": state.get("news", [])[:3],
                "analysis": [resp.content],
                "risks": [],
                "action_items": [],
                "disclaimers": [
                    "This is for education/information, not a recommendation to buy/sell.",
                    "Markets are risky; verify from official sources before acting.",
                ],
            },
        }

    return {"final": obj, "messages": [AIMessage(content=json.dumps(obj, ensure_ascii=False))]}


# --------------------------
# Build graph
# --------------------------
def build_graph():
    builder = StateGraph(InvestState)
    builder.add_node("resolve_symbol", resolve_symbol_node)
    builder.add_node("fetch_stock", fetch_stock_node)
    builder.add_node("fetch_news", fetch_news_node)
    builder.add_node("write_answer", write_answer_node)

    builder.add_edge(START, "resolve_symbol")
    builder.add_edge("resolve_symbol", "fetch_stock")
    builder.add_edge("fetch_stock", "fetch_news")
    builder.add_edge("fetch_news", "write_answer")
    builder.add_edge("write_answer", END)

    conn = sqlite3.connect(Config.CHECKPOINT_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


# --------------------------
# CLI rendering
# --------------------------
def render_response(obj: dict) -> str:
    out = obj.get("output", {}) if isinstance(obj, dict) else {}
    md = out.get("market_data", {}) if isinstance(out, dict) else {}

    sym = out.get("symbol") or md.get("symbol") or "N/A"
    company = out.get("company") or md.get("company") or "N/A"
    err = md.get("error") or out.get("error")

    lines = []
    lines.append(f"Symbol  : {sym}")
    lines.append(f"Company : {company}")

    if err:
        lines.append(f"Error   : {err}")
        return "\n".join(lines)

    lines.append(f"Price   : {md.get('price')} | Change: {md.get('change')} | %Chg: {md.get('p_change')}")
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

    analysis = out.get("analysis")
    if isinstance(analysis, list):
        analysis = [a for a in analysis if str(a).strip()]
    if analysis:
        lines.append("\nAnalysis:")
        for a in analysis[:6]:
            lines.append(f"- {str(a).strip()}")

    return "\n".join(lines)


# --------------------------
# Chat loop
# --------------------------
def run_chat():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("Invest bot ready. Examples:")
    print("- HAL")
    print("- Analyze HAL stock")
    print("- Analyze Tata Consultancy stock")
    print("Type 'exit' to quit.\n")

    first = input("You: ").strip()
    if first.lower() in {"exit", "quit"}:
        return

    out = graph.invoke(init_state(first), config=config)
    pprint(out.get("final", {}))
    print("\nBot:\n" + render_response(out.get("final", {})))

    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        updates = {
            "query": user_text,
            "messages": [HumanMessage(content=user_text)],
            "symbol": (out.get("symbol") or ""),
        }
        out = graph.invoke(updates, config=config)
        pprint(out.get("final", {}))
        print("\nBot:\n" + render_response(out.get("final", {})))


if __name__ == "__main__":
    run_chat()
