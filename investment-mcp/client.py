"""Default sample MCP client for stock-market tools."""

import argparse
import asyncio
import json
import sys
from typing import Any

import pandas as pd
from mcp import Client


def _result_payload(result: object) -> Any:
    payload = getattr(result, "structured_content", None)
    if payload is not None:
        return payload
    for content in getattr(result, "content", []):
        if hasattr(content, "text"):
            try:
                return json.loads(content.text)
            except json.JSONDecodeError:
                return content.text
    return None


def _print_json_result(result: object) -> None:
    print("-" * 70)
    print(json.dumps(_result_payload(result), indent=2, default=str))
    print("\n" + "*" * 70)


def _print_dataframe(records: Any, *, limit: int = 5) -> None:
    print("-" * 70)
    if not isinstance(records, list) or not records:
        print("No rows returned.")
    else:
        print(pd.DataFrame(records).head(limit).to_string(index=False))
    print("\n" + "*" * 70)


def _print_section(title: str) -> None:
    print(f"\n{'-' * 70}\n{title}\n{'-' * 70}")


async def run_client(
    url: str,
    ticker: str,
    market: str,
    period: str,
    interval: str,
    frequency: str,
    examples: bool,
    analysis: bool,
) -> None:
    async with Client(url) as client:
        required_tools = {"get_stock_quote"}
        if examples:
            required_tools.update({"get_stock_history", "get_stock_fundamentals"})
        if analysis:
            required_tools.update(
                {
                    "get_stock_technical_indicators",
                    "get_stock_return_metrics",
                    "get_stock_risk_metrics",
                }
            )
        available_tools = {tool.name for tool in (await client.list_tools()).tools}
        missing_tools = sorted(required_tools - available_tools)
        if missing_tools:
            missing = ", ".join(missing_tools)
            print(
                f"The MCP server at {url} does not provide required stock tool(s): "
                f"{missing}.\nRestart the updated server with `python server.py`, "
                "then run this client again.",
                file=sys.stderr,
            )
            return

        quote = await client.call_tool(
            "get_stock_quote", {"ticker": ticker, "market": market}
        )
        _print_section("Stock quote")
        _print_json_result(quote)

        if analysis:
            analysis_calls = [
                (
                    "Technical indicators",
                    "get_stock_technical_indicators",
                    {
                        "ticker": ticker,
                        "market": market,
                        "period": period,
                        "interval": interval,
                    },
                ),
                (
                    "Historical return metrics",
                    "get_stock_return_metrics",
                    {
                        "ticker": ticker,
                        "market": market,
                        "period": period,
                        "interval": interval,
                    },
                ),
                (
                    "Historical risk metrics",
                    "get_stock_risk_metrics",
                    {
                        "ticker": ticker,
                        "market": market,
                        "period": period,
                        "interval": interval,
                    },
                ),
            ]
            for label, tool_name, arguments in analysis_calls:
                result = await client.call_tool(tool_name, arguments)
                _print_section(label)
                _print_json_result(result)

        if not examples:
            return

        history = await client.call_tool(
            "get_stock_history",
            {
                "ticker": ticker,
                "market": market,
                "period": period,
                "interval": interval,
            },
        )
        history_payload = _result_payload(history)
        _print_section(f"Historical prices ({period}, {interval}; first 5 rows)")
        if isinstance(history_payload, dict) and "error" not in history_payload:
            _print_dataframe(history_payload.get("data"))
        else:
            _print_json_result(history)

        fundamentals = await client.call_tool(
            "get_stock_fundamentals",
            {"ticker": ticker, "market": market, "frequency": frequency},
        )
        fundamentals_payload = _result_payload(fundamentals)
        _print_section(f"Company fundamentals ({frequency})")
        if not isinstance(fundamentals_payload, dict) or "error" in fundamentals_payload:
            _print_json_result(fundamentals)
            return

        _print_section("Metrics")
        print(json.dumps(fundamentals_payload.get("metrics", {}), indent=2, default=str))
        print("\n" + "*" * 70)
        for name, records in fundamentals_payload.get("financial_statements", {}).items():
            _print_section(f"{name.replace('_', ' ').title()} (first 5 rows)")
            _print_dataframe(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample client for stock-market MCP tools.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    parser.add_argument("--ticker", default="RELIANCE", help="Yahoo Finance ticker.")
    parser.add_argument(
        "--market",
        choices=("india", "bse", "global"),
        default="india",
        help="Market used to resolve a bare ticker (default: india).",
    )
    parser.add_argument("--period", default="1y", help="History period, e.g. 1y or 5d.")
    parser.add_argument("--interval", default="1d", help="History interval, e.g. 1d or 1wk.")
    parser.add_argument(
        "--frequency",
        choices=("annual", "quarterly"),
        default="annual",
        help="Fundamental statement frequency.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Also fetch historical prices and company fundamentals.",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Also calculate technical indicators, historical returns, and risk metrics.",
    )
    args = parser.parse_args()
    asyncio.run(
        run_client(
            args.url,
            args.ticker,
            args.market,
            args.period,
            args.interval,
            args.frequency,
            args.examples,
            args.analysis,
        )
    )


if __name__ == "__main__":
    main()
