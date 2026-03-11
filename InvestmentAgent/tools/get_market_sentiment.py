import yfinance as yf
import pandas as pd
from langchain_core.tools import tool
from typing import Dict, Any

from utils.logger import LOGGER


@tool
def get_market_sentiment() -> Dict[str, Any]:
    """
    Estimate overall Indian market sentiment.

    Uses Nifty 50 daily return.
    """
    LOGGER.info("get_market_sentiment")

    try:

        df = yf.download("^NSEI", period="2d", progress=False)

        today = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2]

        change_pct = (today - prev) / prev * 100

        if change_pct > 1:
            sentiment = "strong_bullish"
        elif change_pct > 0:
            sentiment = "bullish"
        elif change_pct < -1:
            sentiment = "strong_bearish"
        else:
            sentiment = "bearish"

        return {
            "status": "success",
            "index": "NIFTY 50",
            "change_pct": float(change_pct),
            "sentiment": sentiment
        }

    except Exception as e:

        LOGGER.exception("get_market_sentiment failed")

        return {"status": "error", "message": str(e)}