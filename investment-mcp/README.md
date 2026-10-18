# Investment Data MCP Server

A server built on the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for stock-market and mutual-fund research. It exposes stock quotes, price history, company fundamentals, scheme discovery, NAV data, performance information, and portfolio calculations as tools that MCP-compatible applications and AI agents can discover and invoke.

> **Disclaimer:** This project provides information for research purposes only. Data may be delayed, incomplete, or unavailable. It does not provide personalized financial or investment advice.

## Contents

- [Capabilities](#capabilities)
- [Requirements and installation](#requirements-and-installation)
- [Run the server](#run-the-server)
- [Use the clients](#use-the-clients)
- [Stock ticker and market conventions](#stock-ticker-and-market-conventions)
- [MCP tools](#mcp-tools)
- [Agentic pipeline integration](#agentic-pipeline-integration)
- [Testing](#testing)
- [Data sources and limitations](#data-sources-and-limitations)

## Capabilities

- **Stocks:** quotes, historical OHLCV prices, company metrics, and annual or quarterly financial statements through Yahoo Finance.
- **Mutual funds:** scheme search and metadata, NAV quotes and history, performance data, and portfolio/SIP calculations through `mftool`.
- **MCP integration:** tools are registered on a single server and can be discovered at runtime by MCP clients and agent frameworks.
- **Cache operations:** inspect and manage the `mftool` cache.

## Requirements and installation

- Python 3.10 or later
- Dependencies from [`requirements.txt`](./requirements.txt)

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the server

Start the streamable HTTP server with its default host and port:

```bash
python server.py
```

The MCP endpoint is:

```text
http://127.0.0.1:8000/mcp
```

To choose a different host or port:

```bash
python server.py --host 127.0.0.1 --port 8001
```

For MCP hosts that launch the server as a subprocess, use stdio transport:

```bash
python server.py --transport stdio
```

Restart the server after changing source code or registering tools. A running process will not automatically load code changes.

## Use the clients

The default [`client.py`](./client.py) demonstrates the stock-market tools. The separate [`mf_client.py`](./mf_client.py) demonstrates mutual-fund tools. Start the server first unless your MCP host launches it automatically.

### Stock-market client

Fetch a quote for Reliance Industries. Bare symbols use NSE India by default:

```bash
python client.py --ticker RELIANCE
```

Fetch a quote, historical prices, and company fundamentals:

```bash
python client.py --ticker RELIANCE --market india --examples
```

Calculate technical indicators, historical returns, and risk metrics:

```bash
python client.py --ticker RELIANCE --analysis --period 1y --interval 1d
```

Fetch global or BSE-listed data:

```bash
python client.py --ticker AAPL --market global --examples --period 1y --interval 1wk --frequency quarterly
python client.py --ticker TCS --market bse --examples
```

The client prints up to five historical-price and financial-statement rows as pandas DataFrames. Analysis includes SMA, EMA, RSI, MACD, Bollinger Bands, ATR, total and annualized returns, annualized volatility, Sharpe and Sortino ratios, maximum drawdown, and interval VaR/expected shortfall. These describe historical data and are not forecasts or trading recommendations. Run `python client.py --help` to see all options. [`stock_client.py`](./stock_client.py) is retained as a compatibility alias.

### Mutual-fund client

Search for a scheme by name:

```bash
python mf_client.py --query "Parag Parikh Flexi Cap Fund"
```

Run the sample search, scheme-details, quote, and NAV-history calls:

```bash
python mf_client.py --examples
```

NAV history is displayed as a DataFrame, limited to the first five rows. `search_schemes` searches scheme names, not numeric scheme codes. Use code-based tools such as `get_scheme_details` and `get_scheme_quote` when you have an AMFI code.

## Stock ticker and market conventions

Stock tools accept `ticker` and `market` arguments:

| `market` value | Bare ticker handling | Example |
| --- | --- | --- |
| `india` (default) | Appends `.NS` for NSE | `RELIANCE` → `RELIANCE.NS` |
| `bse` | Appends `.BO` for BSE | `TCS` → `TCS.BO` |
| `global` | Uses ticker without a market suffix | `AAPL` → `AAPL` |

Tickers that already contain a dot, such as `RELIANCE.NS` or `7203.T`, are passed through as provided. Availability depends on Yahoo Finance.

## MCP tools

### Stock-market tools

| Tool | Description | Key arguments |
| --- | --- | --- |
| `get_stock_quote` | Latest available quote and selected metrics | `ticker`, `market` |
| `get_stock_history` | Historical OHLCV prices | `ticker`, `period`, `interval`, `market` |
| `get_stock_fundamentals` | Company metrics and financial statements | `ticker`, `frequency`, `market` |
| `get_stock_technical_indicators` | SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and average volume | `ticker`, `period`, `interval`, `market` |
| `get_stock_return_metrics` | Total and annualized historical return | `ticker`, `period`, `interval`, `market` |
| `get_stock_risk_metrics` | Volatility, Sharpe/Sortino, drawdown, VaR, and expected shortfall | `ticker`, `period`, `interval`, `market`, `risk_free_rate_pct` |

### Mutual-fund tools

| Category | Tools |
| --- | --- |
| Scheme discovery and metadata | `get_scheme_codes`, `get_available_schemes`, `get_scheme_details`, `get_scheme_info`, `is_valid_scheme_code`, `is_valid_new_scheme_code`, `search_schemes`, `search_schemes_by_amc`, `search_schemes_by_type` |
| NAV and history | `get_scheme_quote`, `get_bulk_quotes`, `get_scheme_historical_nav`, `get_scheme_historical_nav_for_dates`, `get_scheme_history` |
| Performance | `get_equity_scheme_performance`, `get_debt_scheme_performance`, `get_hybrid_scheme_performance`, `get_solution_scheme_performance`, `get_other_scheme_performance` |
| Calculations | `calculate_balance_units_value`, `calculate_portfolio_value`, `calculate_returns` |
| Other | `get_average_aum`, `compare_mutual_funds`, `get_cache_stats`, `clear_cache`, `set_cache_enabled` |

AMFI scheme codes and BSE/Yahoo Finance mutual-fund codes are distinct formats. Follow the relevant tool description when providing a code. To get a code from a scheme name, first call `search_schemes`; do not encode operations in a search string such as `nav_history:122640`.

## Agentic pipeline integration

This server supplies callable data tools. The MCP-compatible agent runtime supplies the model and coordinates reasoning, tool selection, and the final response. A recommended pipeline is:

```text
User request
    → identify intent, instrument, market, and requested period
    → discover available MCP tools and their argument schemas
    → resolve or clarify ticker / scheme identity
    → invoke the minimum necessary tool or tools
    → validate tool status and returned data
    → present results with identifiers, periods, and limitations
```

### Recommended orchestration steps

1. **Connect and discover tools.** Connect to `http://127.0.0.1:8000/mcp` (or configure stdio) and request the live tool catalog. Use returned tool descriptions and schemas rather than assuming a tool exists.
2. **Classify the request.** Determine whether the user asks about an individual stock, a mutual fund, a portfolio calculation, or a comparison.
3. **Resolve the instrument.** For stocks, establish the ticker and market. For mutual funds, search by name and use the selected result's AMFI code for NAV and details. If multiple plans or listings match, ask the user to choose instead of silently selecting one.
4. **Gather required arguments.** Confirm dates, frequency, intervals, investment units, and other required inputs from the tool schema. Clarify ambiguous or missing information before calling tools.
5. **Invoke tools selectively.** Call only the tools needed. Requests can chain results—for example, search for a scheme, then fetch its quote and history. Independent requests may run concurrently when supported by the agent runtime.
6. **Validate responses.** Check the MCP call status, error flag, and returned content. Treat empty or error responses as unavailable data, not as valid zero values. Preserve ticker/scheme identity and period when combining results.
7. **Respond transparently.** State the ticker or scheme code, market, requested period, and whether financial statements are annual or quarterly. Note missing or delayed data, and avoid presenting results as guaranteed real-time prices or personalized advice.
8. **Handle state-changing tools cautiously.** `clear_cache` and `set_cache_enabled` change server state. Use them only when explicitly requested and authorized.

### Example agent plans

**Request:** “Find the Parag Parikh Flexi Cap Fund and show its latest NAV and recent history.”

1. Call `search_schemes` with the fund name.
2. If multiple plans are returned, ask which scheme the user means.
3. Call `get_scheme_quote` and `get_scheme_historical_nav` with the selected AMFI code.
4. Summarize the results with the chosen scheme identity and any data-availability caveats.

**Request:** “Show me a year of weekly prices and quarterly fundamentals for Apple.”

1. Resolve the listing as `AAPL` on the global market, asking for clarification if needed.
2. Call `get_stock_history` with `ticker="AAPL"`, `market="global"`, `period="1y"`, and `interval="1wk"`.
3. Call `get_stock_fundamentals` with `ticker="AAPL"`, `market="global"`, and `frequency="quarterly"`.
4. Present price history and financial statements separately; identify unavailable data rather than inferring it.

### Minimal MCP discovery example

This Python snippet connects to the HTTP server and lists its currently registered tools. In an agent pipeline, the agent framework typically makes these schemas available to the model and dispatches the tool calls it selects.

```python
import asyncio
from mcp import Client


async def main():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        result = await client.list_tools()
        for tool in result.tools:
            print(tool.name, tool.description)


asyncio.run(main())
```

This project does not bundle a specific LLM, agent framework, credentials, or autonomous investment strategy. Configure the MCP endpoint in the agent runtime of your choice.

## Testing

Run the offline test suite:

```bash
python -m unittest discover -s tests -v
```

Stock-tool tests mock Yahoo Finance and do not require live market-data requests.

## Data sources and limitations

Stock data is retrieved through Yahoo Finance. Mutual-fund data is provided through `mftool` and its upstream data sources. These services may change, rate-limit requests, return delayed or incomplete results, or be temporarily unavailable. Verify important values with the relevant exchange, fund house, or AMFI.
