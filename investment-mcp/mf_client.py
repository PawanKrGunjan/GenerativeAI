"""Small MCP client for discovering tools and searching schemes."""

import argparse
import asyncio
import json

import pandas as pd
from mcp import Client


def print_result(result: object) -> None:
    """Print a tool result as structured JSON when available."""
    print("-" * 70)
    structured_content = getattr(result, "structured_content", None)
    if structured_content is not None:
        print(json.dumps(structured_content, indent=2, default=str))
    else:
        for content in getattr(result, "content", []):
            if hasattr(content, "text"):
                print(content.text)
    print("\n" + "*" * 70)


def print_nav_dataframe(result: object) -> None:
    """Print historical NAV entries as a pandas DataFrame."""
    payload = getattr(result, "structured_content", None)
    if payload is None:
        for content in getattr(result, "content", []):
            if hasattr(content, "text"):
                try:
                    payload = json.loads(content.text)
                except json.JSONDecodeError:
                    continue
                break

    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        nav_data = payload["data"]
        if nav_data:
            dataframe = pd.DataFrame(nav_data).head(5)
            columns = [column for column in ("date", "nav") if column in dataframe]
            if columns:
                dataframe = dataframe[columns]
            print(dataframe.to_string(index=False))
            return

    print_result(result)


async def run_client(url: str, query: str) -> None:
    async with Client(url) as client:
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools.tools:
            print(f"- {tool.name}: {tool.description or ''}")
            print("-" * 70)

        result = await client.call_tool("search_schemes", {"query": query})
        print("\nSearch results:")
        print_result(result)


async def run_examples(url: str) -> None:
    """Run sample calls for the tools that support the selected fund."""
    fund_data = {
        "code": "122640",
        "name": "Parag Parikh Flexi Cap Fund - Growth",
    }
    examples = [
        (
            "Get scheme details by code",
            "get_scheme_details",
            {"scheme_code": fund_data["code"]},
        ),
        ("Search by scheme name", "search_schemes", {"query": fund_data["name"]}),
        ("Latest NAV quote", "get_scheme_quote", {"scheme_code": fund_data["code"]}),
        (
            "Historical NAV",
            "get_scheme_historical_nav",
            {"scheme_code": fund_data["code"]},
        ),
    ]

    async with Client(url) as client:
        for label, tool_name, arguments in examples:
            print(f"\n{label}:")
            result = await client.call_tool(tool_name, arguments)
            if tool_name == "get_scheme_historical_nav":
                print_nav_dataframe(result)
            else:
                print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample client for the mutual-fund MCP server."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    parser.add_argument("--query", default="Parag Parikh Flexi Cap Fund", help="Scheme info")
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run additional supported searches, quote, and historical NAV examples.",
    )
    args = parser.parse_args()
    asyncio.run(run_client(args.url, args.query))
    if args.examples:
        asyncio.run(run_examples(args.url))


if __name__ == "__main__":
    main()
