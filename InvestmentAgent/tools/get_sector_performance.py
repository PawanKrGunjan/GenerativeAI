from langchain_core.tools import tool
import yfinance as yf
import pandas as pd
from typing import Dict, Any
from utils.logger import LOGGER

@tool
def get_sector_performance() -> Dict[str, Any]:
    """
    Estimate sector performance using representative stocks.

    Returns:
        Sector return percentages for today.
    """

    LOGGER.info("get_sector_performance")

    try:

        sectors = {
            "banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
            "it": ["INFY.NS", "TCS.NS", "WIPRO.NS",'HCLTECH.NS'],
            "energy": ["RELIANCE.NS", "ONGC.NS"],
            "fmcg": ["ITC.NS", "HINDUNILVR.NS"],
            "auto": ["TATAMOTORS.NS", "MARUTI.NS"]
        }

        results = {}

        for sector, tickers in sectors.items():

            df = yf.download(tickers, period="2d", progress=False)["Close"]

            returns = (df.iloc[-1] - df.iloc[-2]) / df.iloc[-2] * 100

            results[sector] = float(returns.mean())

        return {
            "status": "success",
            "sector_returns_pct": results
        }

    except Exception as e:

        LOGGER.exception("get_sector_performance failed")

        return {"status": "error", "message": str(e)}