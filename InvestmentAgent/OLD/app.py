from __future__ import annotations

import argparse
import os
import uuid
from pprint import pprint

from dotenv import load_dotenv, find_dotenv

from langchain_core.globals import set_debug, set_verbose
from langchain_core.messages import HumanMessage
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.runnables import RunnableConfig

from utils.logger_config import setup_logger
from utils import (
    build_tools,
    safe_float, safe_json_loads, quiet_call,
    has_any_price, rank_symbols, ensure_list,
    extract_symbols_from_query,
    build_final_error,
)

from src import Config, build_graph, load_nse_master, build_investor_prompt
from src.config import apply_langchain_debug, build_llm, build_ddg_api
from src.agent_state import init_state

from src.analysis import python_compare_analysis, python_fallback_analysis
from src.nodes import (
    make_resolve_symbol_node,
    make_fetch_stock_node,
    make_fetch_news_node,
    resolve_intent_node,
    compute_metrics_node,
    make_fetch_extras_node,
    make_write_answer_node,
)


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


def run_chat(*, graph, debug: bool) -> None:
    thread_id = str(uuid.uuid4())

    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Agent-level per-run debug output (callbacks to stdout) [web:171]
    if debug:
        cfg["callbacks"] = [StdOutCallbackHandler()]

    print("Invest bot ready. Examples:")
    print("- HAL")
    print("- Analyze HAL stock")
    print("- Compare HAL vs LT")
    print("Type 'exit' to quit.\n")

    first = input("You: ").strip()
    if first.lower() in {"exit", "quit"}:
        return

    out = graph.invoke(init_state(first), config=cfg)
    if debug:
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

        if debug:
            pprint(out.get("final", {}))
        print("\nBot:\n" + render_response(out.get("final", {})))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Let CLI force debug before env/config load
    if args.debug:
        os.environ["AGENT_DEBUG"] = "1"

    # Load root .env once (project root)
    load_dotenv(find_dotenv(".env", usecwd=True), override=False)

    cfg = Config.from_env()

    # Agent-level global debug/verbose (one-time global switches) [web:155][web:143]
    # Keep your existing wrapper (recommended), but also hard-enable here to be explicit.
    apply_langchain_debug(cfg.debug)
    if cfg.debug:
        set_debug(True)
        set_verbose(True)
    else:
        set_debug(False)
        set_verbose(False)

    logs_dir = os.getenv("LOGS_DIR", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    logger = setup_logger(
        debug_mode=cfg.debug,
        log_name="InvestmentAgent.log",
        log_dir=logs_dir,
    )

    llm = build_llm(cfg)
    ddg_api = build_ddg_api()

    master = load_nse_master(cfg.master_stock_file)
    SYMBOL_INDEX = master.symbol_index
    NAME_INDEX = master.name_index

    bundle = build_tools(
        llm=llm,
        ddg_api=ddg_api,
        ddg_results=cfg.ddg_results,
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

    resolve_symbol_node = make_resolve_symbol_node(
        symbol_index=SYMBOL_INDEX,
        extract_symbols_from_query=lambda q: extract_symbols_from_query(
            q,
            SYMBOL_INDEX,
            company_ranker=lambda txt: rank_symbols(txt, NAME_INDEX, SYMBOL_INDEX, limit=1),
        ),
        rank_symbols=lambda company_text, limit: rank_symbols(company_text, NAME_INDEX, SYMBOL_INDEX, limit=limit),
        llm_extract_company=llm_extract_company,
        llm_choose_symbol=llm_choose_symbol,
    )

    fetch_stock_node = make_fetch_stock_node(
        get_nse_stock_data=get_nse_stock_data,
        has_any_price=has_any_price,
        logger=logger,
    )

    fetch_news_node = make_fetch_news_node(search_tool=search_tool, logger=logger, max_items=8)
    fetch_extras_node = make_fetch_extras_node(search_tool=search_tool, logger=logger)

    investor_prompt = build_investor_prompt()

    write_answer_node = make_write_answer_node(
        build_final_error=build_final_error,
        python_compare_analysis=python_compare_analysis,
        python_fallback_analysis=python_fallback_analysis,
        safe_json_loads=safe_json_loads,
        ensure_list=ensure_list,
        investor_prompt=investor_prompt,
        llm_with_tools=llm_with_tools,
    )

    graph = build_graph(
        checkpoint_db=cfg.checkpoint_db,
        resolve_symbol_node=resolve_symbol_node,
        resolve_intent_node=resolve_intent_node,
        fetch_stock_node=fetch_stock_node,
        fetch_extras_node=fetch_extras_node,
        compute_metrics_node=compute_metrics_node,
        fetch_news_node=fetch_news_node,
        write_answer_node=write_answer_node,
    )

    if cfg.debug:
        print("\n==== MERMAID GRAPH ====\n", graph.get_graph().draw_mermaid())

    run_chat(graph=graph, debug=cfg.debug)


if __name__ == "__main__":
    main()
