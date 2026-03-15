from fastapi import APIRouter, WebSocket
from chat.chat_service import run_chat

router = APIRouter()

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):

    await websocket.accept()

    while True:

        data = await websocket.receive_json()

        session_id = data["session_id"]
        message = data["message"]

        result = await run_chat(session_id, message)

        await websocket.send_json({
            "answer": result["answer"]
        })