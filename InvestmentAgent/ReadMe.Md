project/
  stock_agent.py              # tiny entrypoint (or src/main.py)
  .env
  src/
    __init__.py
    agent_state.py            # StockState, InvestState, init_state
    config.py                 # Config + debug/bootstrap helpers
    indices.py                # benchmark fetching (NIFTY/SENSEX later)
    graph.py                  # build_graph() only (wiring)
    nodes/
      __init__.py
      resolve_symbol.py
      resolve_intent.py
      fetch_stock.py
      fetch_news.py
      fetch_extras.py
      compute_metrics.py
      write_answer.py
    analysis/
      __init__.py
      compare.py              # python_compare_analysis
      fallback.py             # python_fallback_analysis
  utils/
    __init__.py
    helper.py                 # pure helpers (safe_json_loads, ensure_list, etc.)
    tools.py                  # build_tools + ToolBundle
    logger_config.py
