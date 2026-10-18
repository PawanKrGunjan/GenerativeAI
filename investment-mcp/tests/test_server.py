"""Offline tests for MCP tool registration and response wiring."""

import json
import unittest
from unittest.mock import patch

from mcp import Client

from mcp_instance import mf, mcp
import server as _server  # noqa: F401 - importing registers MCP tools


class MutualFundServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_registers_expected_tool_groups(self) -> None:
        async with Client(mcp) as client:
            result = await client.list_tools()

        self.assertIs(_server.mcp, mcp)
        names = {tool.name for tool in result.tools}
        self.assertTrue(
            {
                "search_schemes",
                "get_scheme_quote",
                "get_scheme_historical_nav",
                "get_equity_scheme_performance",
                "calculate_portfolio_value",
                "compare_mutual_funds",
                "get_cache_stats",
                "get_stock_quote",
                "get_stock_history",
                "get_stock_fundamentals",
                "get_stock_technical_indicators",
                "get_stock_return_metrics",
                "get_stock_risk_metrics",
            }.issubset(names)
        )

    async def test_search_tool_returns_structured_results(self) -> None:
        expected = [{"code": "119597", "name": "Example Fund"}]
        with patch.object(mf, "search_schemes", return_value=expected):
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "search_schemes", {"query": "Example", "limit": 10}
                )

        self.assertFalse(result.is_error)
        self.assertEqual(
            json.loads(result.content[0].text),
            {"query": "Example", "results": expected},
        )


if __name__ == "__main__":
    unittest.main()
