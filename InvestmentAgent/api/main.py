# api/main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from api.routes import tools, chat, portfolio

from tools.market_tools import get_market_indices, get_stock_info


app = FastAPI(
    title="Stock Investment Agent API",
    version="1.0",
    description="Indian Stock Market AI with portfolio upload & analysis"
)


# ─────────────────────────────────────────────
# CORS (required for web UI / external apps)
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Static Files
# ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="web"), name="static")


# ─────────────────────────────────────────────
# Register Routers
# ─────────────────────────────────────────────
app.include_router(chat.router)
app.include_router(tools.router)
app.include_router(portfolio.router)


# ─────────────────────────────────────────────
# Homepage
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def homepage():
    """Serve main chat UI."""

    index_path = Path("web/index.html")

    if index_path.exists():
        return HTMLResponse(index_path.read_text())

    return HTMLResponse(
        "<h2>UI not found</h2><p>Please create web/index.html</p>",
        status_code=404
    )


# ─────────────────────────────────────────────
# Market APIs
# ─────────────────────────────────────────────
@app.get("/market/indices")
def market_indices():
    """Return major Indian market indices."""

    try:
        return get_market_indices.invoke({})
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/market/quote/{symbol}")
def market_quote(symbol: str):
    """Return stock quote."""

    try:
        return get_stock_info.invoke({"symbol": symbol})
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# Local Dev Runner
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        log_level='critical',
        reload=True
    )