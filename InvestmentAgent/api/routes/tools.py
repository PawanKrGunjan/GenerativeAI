from fastapi import APIRouter

from api.schemas import ToolRequest
from tools.tool_registry import TOOLS

router = APIRouter(prefix="/tools", tags=["tools"])

tool_map = {t.name: t for t in TOOLS}


@router.get("/")
def list_tools():

    return {
        "tools": list(tool_map.keys())
    }


@router.post("/run")
def run_tool(req: ToolRequest):

    if req.tool_name not in tool_map:
        return {"error": "Tool not found"}

    result = tool_map[req.tool_name].invoke(req.args)

    return result