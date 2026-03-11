# api/routes/chat.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil
import pandas as pd
from typing import Optional

from api.schemas import ChatResponse
from agents.investment_agent import agent
from utils.logger import LOGGER
from utils.config import DATA_DIR

router = APIRouter(prefix="/chat", tags=["chat"])

# Allowed portfolio file types
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}


# ─────────────────────────────────────────────
# Chat UI
# ─────────────────────────────────────────────
@router.get("/", response_class=HTMLResponse)
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
async def chat(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """Chat with optional portfolio upload."""

    original_message = message

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if file and file.filename:

        LOGGER.info("Processing upload: %s", file.filename)

        ext = Path(file.filename).suffix.lower()

        # Validate file type
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}"
            )

        # Create safe filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"holdings-{Path(file.filename).stem}-{timestamp}{ext}"

        portfolio_path = DATA_DIR / safe_name

        # Save uploaded file
        with open(portfolio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        LOGGER.info("Portfolio saved: %s", portfolio_path)

        # Enhance agent message
        message = f"{original_message}\n\n📁 Portfolio uploaded: {safe_name}"

    # Run investment agent safely
    try:

        result = agent.run(query=message)

        answer = result.get("answer", "No response from agent")
        time_ist = result.get(
            "current_time_ist",
            pd.Timestamp.now().strftime("%H:%M:%S")
        )

    except Exception as e:

        LOGGER.exception("Agent execution failed")

        answer = "⚠️ AI agent failed to process the request."
        time_ist = pd.Timestamp.now().strftime("%H:%M:%S")

    return ChatResponse(
        answer=answer,
        time=time_ist
    )