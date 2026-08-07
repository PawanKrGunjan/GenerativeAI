# api/routes/chat.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil
import pandas as pd
from typing import Optional

from api.schemas import ChatRequest, ChatResponse
from utils.logger import LOGGER
from utils.config import DATA_DIR
from agents.investment_agent_mcp import run_mcp_agent

router = APIRouter(prefix="/chat", tags=["chat"])

# Allowed portfolio file types
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}


# ─────────────────────────────────────────────
# Chat UI
# ─────────────────────────────────────────────
@router.get("/ui", response_class=HTMLResponse)
async def chat_ui():
    """Serve interactive chat UI."""
    try:
        html = Path("web/index.html").read_text()
        return HTMLResponse(content=html)
    except FileNotFoundError:
        return HTMLResponse(
            content="❌ UI not found. Please create web/index.html",
            status_code=404
        )


# ─────────────────────────────────────────────
# Chat Endpoint
# ─────────────────────────────────────────────
@router.post("/", response_model=ChatResponse)
async def chat(request: Request, file: Optional[UploadFile] = File(None)):
    """Chat with optional portfolio upload. Supports JSON body and form-data."""

    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        payload = await request.json()
        chat_payload = ChatRequest(**payload)
        message = chat_payload.message
        session_id = chat_payload.session_id or "default"
    else:
        form = await request.form()
        message = form.get("message", "")
        session_id = form.get("session_id") or "default"
        file = form.get("file") if "file" in form else file

    if not message:
        raise HTTPException(status_code=422, detail="Message is required")

    original_message = message

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------
    # Handle portfolio upload
    # ---------------------------------------
    if file and getattr(file, "filename", None):

        LOGGER.info("Processing upload: %s", file.filename)

        ext = Path(file.filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}"
            )

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"holdings-{Path(file.filename).stem}-{timestamp}{ext}"

        portfolio_path = DATA_DIR / safe_name

        with open(portfolio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        LOGGER.info("Portfolio saved: %s", portfolio_path)

        message = f"{original_message}\n\n📁 Portfolio uploaded: {safe_name}"

    # ---------------------------------------
    # Run Investment Agent
    # ---------------------------------------
    try:

        result = await run_mcp_agent(message)

        answer = result.get("answer", "No response from agent")

        time_ist = pd.Timestamp.now().strftime("%H:%M:%S")

    except Exception:

        LOGGER.exception("Agent execution failed")
        raise

    return ChatResponse(
        answer=answer,
        time=time_ist)



# ---------------------------------------
# JSON-only Chat Endpoint
# ---------------------------------------
@router.post("/json", response_model=ChatResponse)
async def chat_json(payload: ChatRequest):
    """POST JSON: {"message": "...", "session_id": "..."}

    This endpoint is provided so clients that send pure JSON bodies
    (including programmatic API consumers) have a clean OpenAPI schema
    and do not need to use multipart/form-data.
    """
    message = payload.message
    if not message:
        raise HTTPException(status_code=422, detail="Message is required")

    try:
        result = await run_mcp_agent(message)
        answer = result.get("answer", "No response from agent")
        time_ist = pd.Timestamp.now().strftime("%H:%M:%S")
    except Exception:
        LOGGER.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail="Agent execution failed")

    return ChatResponse(answer=answer, time=time_ist)