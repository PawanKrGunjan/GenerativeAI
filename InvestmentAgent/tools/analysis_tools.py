"""
Technical analysis tools.
"""
import yfinance as yf
import pandas as pd
from langchain_core.tools import tool
from typing import Dict, Any

from utils.logger import LOGGER
from analysis.indicators import add_sma, add_rsi
from tools.market_tools import normalize_symbol

@tool
def compute_technical_indicators(price_data) -> Dict[str, Any]:
    """
    Compute RSI and SMA indicators.

    Args:
        price_data: Price history returned by fetch_price_data.

    Returns:
        Technical indicator summary.
    """

    LOGGER.info("compute_technical_indicators called")

    try:

        df = pd.DataFrame(price_data)

        add_sma(df, 20)
        add_sma(df, 50)
        add_rsi(df)

        latest = df.iloc[-1]

        return {
            "status": "success",
            "close": latest["close"],
            "rsi": latest["RSI"],
            "sma20": latest["SMA_20"],
            "sma50": latest["SMA_50"],
        }

    except Exception as e:

        LOGGER.exception("compute_technical_indicators failed")

        return {"status": "error", "message": str(e)}
    
@tool
def get_top_movers(limit: int = 10) -> Dict[str, Any]:
    """
    Fetch top gainers and losers from NSE.

    Args:
        limit: number of stocks per category

    Returns:
        top gainers and losers
    """

    try:

        tickers = [
            "RELIANCE.NS","INFY.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS",
            "LT.NS","SBIN.NS","AXISBANK.NS","ITC.NS","BHARTIARTL.NS"
        ]

        data = yf.download(tickers, period="2d", progress=False)["Close"]

        returns = (data.iloc[-1] - data.iloc[-2]) / data.iloc[-2] * 100

        df = pd.DataFrame({
            "symbol": returns.index,
            "return_pct": returns.values
        })

        gainers = df.sort_values("return_pct", ascending=False).head(limit)
        losers = df.sort_values("return_pct").head(limit)

        return {
            "status": "success",
            "top_gainers": gainers.to_dict("records"),
            "top_losers": losers.to_dict("records")
        }

    except Exception as e:

        return {"status": "error", "message": str(e)}
    

@tool
def compare_stock_returns(symbols: list, period: str = "1y") -> Dict[str, Any]:
    """
    Compare returns of multiple stocks.

    Args:
        symbols: list of NSE symbols
        period: 6mo, 1y, 3y etc.

    Returns:
        return comparison
    """

    try:

        #tickers = [s + ".NS" for s in symbols]
        tickers = normalize_symbol(symbols)
        df = yf.download(tickers, period=period, progress=False)["Close"]

        returns = {}

        for col in df.columns:
            start = df[col].iloc[0]
            end = df[col].iloc[-1]

            returns[col.replace(".NS","")] = (end-start)/start*100

        return {
            "status": "success",
            "period": period,
            "returns_pct": returns
        }

    except Exception as e:

        return {"status": "error", "message": str(e)}
    
@tool
def predict_stock_trend(symbol: str) -> Dict[str, Any]:
    """
    Predict short-term trend using moving averages.
    """

    try:

        df = yf.download(symbol + ".NS", period="6mo", progress=False)

        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()

        latest = df.iloc[-1]

        if latest["SMA20"] > latest["SMA50"]:
            trend = "bullish"
        else:
            trend = "bearish"

        return {
            "status": "success",
            "symbol": symbol,
            "trend": trend,
            "price": float(latest["Close"])
        }

    except Exception as e:

        return {"status": "error", "message": str(e)}
    
@tool
def market_breadth(symbols: list) -> Dict[str, Any]:
    """
    Calculate market breadth (advancers vs decliners).
    """

    try:

        tickers = [s + ".NS" for s in symbols]

        df = yf.download(tickers, period="2d", progress=False)["Close"]

        advances = 0
        declines = 0

        for col in df.columns:

            if df[col].iloc[-1] > df[col].iloc[-2]:
                advances += 1
            else:
                declines += 1

        return {
            "status": "success",
            "advances": advances,
            "declines": declines
        }

    except Exception as e:

        return {"status": "error", "message": str(e)}