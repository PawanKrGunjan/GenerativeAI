"""Shared MCP server and mutual-fund client instances."""

from mcp.server.mcpserver import MCPServer
from mftool import Mftool

mcp = MCPServer(
    "Investment Data Server",
    title="Stock Market and Mutual Fund Data",
    description="Read-only stock-market and Indian mutual-fund data and analytics.",
    instructions=(
        "Use these tools for informational research only. Market and fund data can be "
        "delayed or unavailable; verify important figures with exchanges, fund houses, "
        "or AMFI. This server does not provide personalized investment advice."
    ),
)
mf = Mftool()
