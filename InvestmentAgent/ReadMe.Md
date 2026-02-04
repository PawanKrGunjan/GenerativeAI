# Stock Agent (NSE) — LangGraph + LangChain CLI

A modular CLI “stock agent” for Indian equities that resolves a company/symbol from user text, fetches NSE market data, pulls news/extras, computes quick metrics, and returns a clean, human-readable answer using a LangGraph workflow.

## Features

- Symbol resolution from free-text (e.g., “Analyze HAL”, “Tata motors share”).
- Market snapshot (price, change %, OHLC, day range, VWAP where available).
- News + extra context (web search summaries, optional).
- Metrics + lightweight analysis (single stock or comparison).
- Pluggable graph nodes (easy to swap data/news providers).
- Debug mode (global LangChain debug + per-run stdout callbacks).

## Project structure

```text
project/
  stock_agent.py              # tiny entrypoint (or src/main.py)
  .env
  src/
    __init__.py
    agent_state.py            # StockState, InvestState, init_state
    config.py                 # Config + debug/bootstrap helpers
    indices.py                # benchmark fetching (NIFTY/SENSEX later)
    graph.py                  # build_graph() only (wiring)
    prompts.py
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
```

## Requirements

- Python 3.10+ (recommended)
- An LLM provider/key supported by your `build_llm()` implementation (see `src/config.py`)
- Internet access (for news/search; optional if you disable those nodes/tools)

## Setup

### 1) Clone and install

```bash
git clone <your-repo-url>
cd project

python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

If you don’t have `requirements.txt` yet, generate it once your environment is working:

```bash
pip freeze > requirements.txt
```

### 2) Create `.env`

Create a `.env` in the project root:

```env
# ---- runtime ----
LOGS_DIR=logs
AGENT_DEBUG=0

# ---- persistence ----
DATA_DIR=data
CHECKPOINT_DB=data/checkpoints.sqlite

# ---- behavior ----
DDG_RESULTS=8

# ---- LLM (examples; match your src/config.py) ----
# OPENAI_API_KEY=...
# GROQ_API_KEY=...
# ANTHROPIC_API_KEY=...
# MODEL_NAME=...
```

Notes:
- Keep `.env` out of git: add it to `.gitignore`.
- Ensure `data/` and `logs/` exist or let the app create them.

## Run

### Basic

```bash
#python stock_agent.py
python app.py
```

Example prompts:

- `HAL`
- `Analyze HAL stock`
- `Compare HAL vs LT`
- `Tata Motors news`
- `Why is INFY falling?`

Type `exit` to quit.

### Debug mode

Debug mode should print internal agent/tool/LLM events to your terminal.

```bash
python app.py --debug
```

What it typically enables:
- Global LangChain debug/verbose flags (more logs).
- A stdout callback handler wired into the LangGraph invocation (agent run tracing).

## How it works

The agent is a LangGraph state machine:

1. **resolve_symbol**: extracts/chooses NSE symbol from user query.
2. **resolve_intent**: decides whether user wants analysis, compare, news, etc.
3. **fetch_stock**: fetches latest stock quote / market data.
4. **fetch_news**: searches news and summarizes.
5. **fetch_extras**: pulls additional context (optional).
6. **compute_metrics**: computes derived metrics used by the final response.
7. **write_answer**: formats the final response and analysis.

Core wiring lives in `src/graph.py` (only the graph wiring), while each node is isolated in `src/nodes/`.

## Configuration

All configuration should come from environment variables (loaded from `.env`) and parsed by `Config.from_env()` in `src/config.py`.

Common settings:

- `LOGS_DIR`: where application logs go.
- `CHECKPOINT_DB`: sqlite path for LangGraph checkpointing.
- `AGENT_DEBUG`: `1` enables debug by default (can also be enabled with `--debug`).

## Logging

- App logs go to `LOGS_DIR/InvestmentAgent.log` (name depends on your `setup_logger` usage).
- Debug mode prints execution traces to stdout.

## Development notes

### Add a new node

1. Create `src/nodes/my_node.py` with a `make_my_node(...)` factory.
2. Export it in `src/nodes/__init__.py` if you want clean imports.
3. Wire it into `src/graph.py` by adding the node and edges.
4. Update `stock_agent.py` to pass dependencies into `make_my_node(...)`.

### Testing (suggested)

- Unit test pure helpers in `utils/helper.py`.
- Unit test symbol resolution with a small fixture master list.
- Integration test the graph with mocked tools (no network, deterministic outputs).

## Security

- Never commit `.env` or API keys.
- Treat tool outputs (web/news) as untrusted; sanitize/validate before storing.

## Roadmap (ideas)

- Index benchmarks (NIFTY, SENSEX) in `src/indices.py`.
- Add caching for quotes/news.
- Add JSON output mode (`--json`) for programmatic usage.
- Add backtesting/screeners and portfolio tracking.

## License

Choose a license (MIT/Apache-2.0 recommended) and add `LICENSE`.