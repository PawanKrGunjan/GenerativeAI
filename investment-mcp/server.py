"""Configure and run the mutual-fund MCP server."""

import argparse

from mcp_instance import mcp
from tools.stock_tool import analysis, stock_info
from tools.mf_tool import cache, calculations, nav, performance, schemes


@mcp.tool()
def compare_mutual_funds(scheme_code_1: str, scheme_code_2: str) -> dict:
    """Compare the latest metadata and NAV quotes for two AMFI scheme codes."""
    return {
        "fund_1": {
            "details": schemes.get_scheme_details(scheme_code_1),
            "quote": nav.get_scheme_quote(scheme_code_1),
        },
        "fund_2": {
            "details": schemes.get_scheme_details(scheme_code_2),
            "quote": nav.get_scheme_quote(scheme_code_2),
        },
    }


@mcp.resource(
    "mf://compare/{code1}/{code2}",
    name="mutual-fund-comparison",
    description="Metadata and latest NAV quotes for two AMFI scheme codes.",
    mime_type="application/json",
)
def mutual_fund_comparison_resource(code1: str, code2: str) -> dict:
    """Expose latest scheme comparisons as a resource template."""
    return compare_mutual_funds(code1, code2)


@mcp.resource(
    "mf://scheme/{scheme_code}/quote",
    name="mutual-fund-quote",
    description="Latest NAV quote for an AMFI scheme code.",
    mime_type="application/json",
)
def scheme_quote_resource(scheme_code: str) -> dict:
    """Expose the latest quote for a scheme as a resource template."""
    return nav.get_scheme_quote(scheme_code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indian mutual-fund MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="streamable-http",
        help="MCP transport (default: streamable-http).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port.")
    args = parser.parse_args()

    if args.transport == "streamable-http":
        mcp.run(args.transport, host=args.host, port=args.port)
    else:
        mcp.run(args.transport)


if __name__ == "__main__":
    main()
