# api/routes/portfolio.py

from fastapi import APIRouter
from utils.config import DATA_DIR
from pathlib import Path
import json

from api.schemas import PortfolioRequest, PortfolioResponse

router = APIRouter(
    prefix="/portfolio",
    tags=["portfolio"]
)

PORTFOLIO_FILE = DATA_DIR / "portfolio.json"


# ─────────────────────────────────────────────
# Load Portfolio
# ─────────────────────────────────────────────

@router.get("/", response_model=PortfolioResponse)
def load_portfolio():

    if not PORTFOLIO_FILE.exists():
        return {"portfolio": []}

    try:
        with open(PORTFOLIO_FILE, "r") as f:
            data = json.load(f)

        return {"portfolio": data}

    except Exception:
        return {"portfolio": []}


# ─────────────────────────────────────────────
# Save Portfolio
# ─────────────────────────────────────────────

@router.post("/save")
def save_portfolio(payload: PortfolioRequest):

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    portfolio_data = [stock.model_dump() for stock in payload.portfolio]

    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio_data, f, indent=2)

    return {
        "status": "success",
        "message": "Portfolio saved"
    }