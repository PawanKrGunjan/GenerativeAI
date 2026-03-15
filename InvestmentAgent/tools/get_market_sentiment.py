import yfinance as yf
import pandas as pd
from langchain_core.tools import tool
from typing import Dict, Any

from utils.logger import LOGGER


@tool
def get_nifty50_market_sentiment() -> Dict[str, Any]:
    """
    Analyzes the latest Nifty 50 index performance to determine current Indian stock market sentiment.

    Use this tool when you need to know if the Indian market (NSE Nifty 50) is bullish, bearish, or neutral based on recent daily price change.
    Fetches the last 2 days of closing prices from Yahoo Finance using ticker '^NSEI'.
    Computes percentage change and classifies sentiment as:
    - strong_bullish: >1% gain
    - bullish: 0-1% gain
    - bearish: 0 to -1% loss
    - strong_bearish: <-1% loss

    Returns a dictionary with status, index name, change percentage, and sentiment label.
    Handles errors gracefully with error status.
    """
    LOGGER.info("Fetching Nifty 50 market sentiment")

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
        LOGGER.exception("get_nifty50_market_sentiment failed")
        return {"status": "error", "message": str(e)}
