from fastapi import APIRouter, WebSocket
from agents.investment_agent_mcp import run_mcp_agent

router = APIRouter()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):

    await websocket.accept()

    while True:

        data = await websocket.receive_json()

        # session_id is accepted but MCP agent is stateless in this wrapper
        message = data.get("message", "")

        result = await run_mcp_agent(message)

        await websocket.send_json({
            "answer": result.get("answer", "")
        })