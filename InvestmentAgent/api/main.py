# api/main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from tools.market_tools import get_market_indices, get_current_stock_price

from api.routes import tools, chat, portfolio

app = FastAPI(
    title="Stock Investment Agent API",
    version="1.0",
    description="Indian Stock Market AI with portfolio upload & analysis"
)

# Serve frontend
app.mount("/static", StaticFiles(directory="web"), name="static")

# Register routers
app.include_router(chat.router)
app.include_router(tools.router)
app.include_router(portfolio.router)

# ─────────────────────────────────────────────
# Homepage
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def homepage():
    """Serve the main chat UI."""

    index_path = Path("web/index.html")

    if index_path.exists():
        return HTMLResponse(index_path.read_text())

    return HTMLResponse(
        "<h2>UI not found</h2><p>Please create web/index.html</p>",
        status_code=404
    )

@app.get("/market/indices")
def market_indices():

    data = get_market_indices.invoke({})

    return data

@app.get("/market/quote/{symbol}")
def market_quote(symbol: str):

    data = get_current_stock_price.invoke({"symbol": symbol})
    return data

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
        reload=True
    )