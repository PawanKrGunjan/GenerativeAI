from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from api.schemas import AgentRunRequest, AgentRunResponse, MCPToolInfo, ToolRequest, ToolResponse
from agents.investment_agent_mcp import run_mcp_agent
from tools.tool_registry import TOOLS, AGENT_TOOL_MAP

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.get("/tools")
async def list_mcp_tools() -> Dict[str, Any]:
    """Return available MCP tools and their categories."""

    tool_map = {t.name: t for t in TOOLS}
    tool_list: List[Dict[str, Any]] = []

    for category, tools in AGENT_TOOL_MAP.items():
        for tool in tools:
            tool_list.append(
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", "") or "",
                    "category": category,
                }
            )

    return {
        "total_tools": len(tool_list),
        "tools": tool_list,
    }


@router.post("/execute", response_model=ToolResponse)
async def execute_tool(req: ToolRequest):
    """Execute a registered tool from the investment agent tool registry."""

    tool_map = {t.name: t for t in TOOLS}
    tool = tool_map.get(req.tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")

    try:
        result = tool.invoke(req.args)
    except Exception as exc:
        return ToolResponse(status="error", error=str(exc), result=None)

    status = "success"
    if isinstance(result, dict) and "status" in result:
        status = result.get("status")

    return ToolResponse(status=status, result=result, error=None)


@router.post("/agent/run", response_model=AgentRunResponse)
async def run_investment_agent(req: AgentRunRequest):
    """Run the MCP-backed investment agent against a user message."""

    result = await run_mcp_agent(req.message)

    return AgentRunResponse(
        answer=result.get("answer", ""),
        memory_summary=result.get("memory_summary"),
        thread_id=result.get("thread_id", "mcp_agent"),
    )
