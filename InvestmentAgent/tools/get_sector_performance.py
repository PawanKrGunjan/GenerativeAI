from langchain_core.tools import tool
import yfinance as yf
import pandas as pd
from typing import Dict, Any
from utils.logger import LOGGER

@tool
def get_sector_performance() -> Dict[str, Any]:
    """
    Analyzes performance of major Indian stock market sectors using representative NSE stocks.

    Use this tool to understand which sectors are leading or lagging in the Indian market today.
    Fetches last 2 days of closing prices from Yahoo Finance and calculates average daily % return for each sector:
    
    Sectors analyzed:
    - banking: HDFCBANK.NS, ICICIBANK.NS, SBIN.NS
    - it: INFY.NS, TCS.NS, WIPRO.NS, HCLTECH.NS  
    - energy: RELIANCE.NS, ONGC.NS
    - fmcg: ITC.NS, HINDUNILVR.NS
    - auto: TATAMOTORS.NS, MARUTI.NS

    Returns dictionary with sector names as keys and average percentage returns as values.
    Example: {"banking": 1.2, "it": -0.5, "energy": 0.8, ...}

    Perfect for market analysis, portfolio allocation decisions, or sector rotation strategies.
    """
    LOGGER.info("Fetching Indian sector performance")

    try:
        sectors = {
            "banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
            "it": ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS"],
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
        LOGGER.exception("get_indian_sector_performance failed")
        return {"status": "error", "message": str(e)}
