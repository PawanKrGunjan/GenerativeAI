"""
Market data tools for Indian stocks using Yahoo Finance.

Design goals for AI agents:
- Low RAM usage
- Prevent hallucinations
- Return compressed data to the LLM
- Save full raw data to disk for deeper analysis
"""

from typing import Dict, Any
import yfinance as yf
import pandas as pd
import json
from pathlib import Path

from utils.config import DATA_DIR
from langchain_core.tools import tool
from utils.logger import LOGGER


# Ensure memory directory exists
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------
# SYMBOL NORMALIZATION
# ---------------------------------------------------

def normalize_symbol(symbol: str) -> str:
    """
    Normalize a user-provided stock symbol to NSE format.

    Examples
    --------
    RELIANCE -> RELIANCE.NS
    INFY.NS  -> INFY.NS
    ^NSEI    -> ^NSEI
    """

    if not symbol:
        return symbol

    symbol = symbol.strip().upper()

    if symbol.startswith("^"):
        return symbol

    if symbol.endswith(".NS"):
        return symbol

    return f"{symbol}.NS"


# ---------------------------------------------------
# STOCK INFORMATION TOOL
# ---------------------------------------------------

@tool
def get_stock_info(symbol: str) -> Dict[str, Any]:
    """
    Retrieve the latest stock snapshot and key fundamentals for an NSE stock.

    PURPOSE
    -------
    This tool fetches the latest market snapshot for a stock including
    current price, 52-week range, valuation metrics, and company details.

    It is typically the FIRST tool to call after resolving a stock ticker.

    WHEN TO USE
    -----------
    Use this tool when you need:
    - Current stock price
    - 52-week high / low
    - Market capitalization
    - P/E ratio or dividend yield
    - Basic company information (sector, industry)

    Example user questions:
    - "What is the current price of RELIANCE?"
    - "Show me TCS fundamentals"
    - "Is INFY close to its 52 week high?"

    INPUT
    -----
    symbol : str
        NSE ticker symbol.

        Valid examples:
        - "RELIANCE"
        - "INFY"
        - "TCS"
        - "HDFCBANK"
        - "^NSEI" (index)

        The tool automatically converts tickers to NSE format (.NS).

        IMPORTANT:
        Never pass company names like "Reliance Industries".
        Always use a ticker symbol.

    OUTPUT
    ------
    Returns a dictionary containing:

    status : "success" or "error"

    symbol : normalized ticker symbol

    data : compressed market snapshot containing
        - price
        - previous_close
        - 52w_high
        - 52w_low
        - dist_52w_high_pct
        - dist_52w_low_pct
        - market_cap
        - pe
        - forward_pe
        - dividend_yield
        - beta
        - sector
        - industry
        - company name

    file_path : location of the full raw JSON data saved on disk.

    MEMORY BEHAVIOR
    ---------------
    The complete Yahoo Finance dataset is saved to:

        DATA_DIR/<symbol>_info.json

    The LLM only receives compressed data to reduce memory usage
    and prevent hallucination.

    NOTES FOR AGENTS
    ----------------
    Always trust values returned by this tool.
    Do NOT estimate or fabricate market data.
    """

    symbol = normalize_symbol(symbol)
    LOGGER.info(f"get_stock_info: {symbol}")

    try:

        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info:
            return {"status": "error", "symbol": symbol, "message": "No data available"}

        # Save raw JSON
        file_path = Path(DATA_DIR) / f"{symbol}_info.json"

        with open(file_path, "w") as f:
            json.dump(info, f, indent=2, default=str)

        price = info.get("currentPrice")
        high = info.get("fiftyTwoWeekHigh")
        low = info.get("fiftyTwoWeekLow")

        dist_high = None
        dist_low = None

        if price and high:
            dist_high = ((high - price) / high) * 100

        if price and low:
            dist_low = ((price - low) / low) * 100

        compressed = {
            "company": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "price": price,
            "previous_close": info.get("previousClose"),
            "52w_high": high,
            "52w_low": low,
            "dist_52w_high_pct": dist_high,
            "dist_52w_low_pct": dist_low,
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
        }

        return {
            "status": "success",
            "symbol": symbol,
            "data": compressed,
            "file_path": str(file_path),
        }

    except Exception as e:
        return {"status": "error", "symbol": symbol, "message": str(e)}


# ---------------------------------------------------
# PRICE HISTORY TOOL
# ---------------------------------------------------

@tool
def get_price_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> Dict[str, Any]:
    """
    Retrieve historical price data for a stock or index.

    PURPOSE
    -------
    Provides historical OHLCV data used for trend analysis,
    technical indicators, and return calculations.

    WHEN TO USE
    -----------
    Use this tool when you need to analyze:

    - Price trends
    - Moving averages
    - Volatility
    - Historical returns
    - Technical indicators

    Example user questions:
    - "Show the trend for TCS"
    - "How has INFY performed over the last year?"
    - "Calculate moving averages for RELIANCE"

    INPUT
    -----
    symbol : str
        Valid ticker symbol such as:

        - RELIANCE
        - INFY
        - TCS
        - ^NSEI

    period : str
        Time range of historical data.

        Common values:
        - 1mo
        - 3mo
        - 6mo
        - 1y
        - 2y
        - 5y

    interval : str
        Data granularity.

        Common values:
        - 1d (daily)
        - 1wk (weekly)
        - 1mo (monthly)
        - 1h (intraday)

    OUTPUT
    ------
    Returns:

    status : success or error

    symbol : normalized ticker

    summary : compressed analytics including
        - latest_close
        - number of rows
        - 1 month return
        - 3 month return
        - 30 day volatility
        - 50 day moving average
        - 200 day moving average

    file_path : location of full CSV data.

    MEMORY BEHAVIOR
    ---------------
    The full historical dataset is saved to:

        DATA_DIR/<symbol>_history.csv

    Only summarized metrics are returned to the LLM to
    minimize token usage and memory footprint.

    NOTES FOR AGENTS
    ----------------
    Use this tool for technical analysis rather than
    requesting raw historical rows.
    """

    symbol = normalize_symbol(symbol)
    LOGGER.info(f"get_price_history: {symbol} {period}:{interval}")

    try:

        df = yf.download(symbol, period=period, interval=interval, progress=False)

        if df.empty:
            return {"status": "error", "symbol": symbol, "message": "No data found"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index().rename(columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })

        df["date"] = df["date"].astype(str)

        file_path = Path(DATA_DIR) / f"{symbol}_history.csv"
        df.to_csv(file_path, index=False)

        df["returns"] = df["close"].pct_change()

        summary = {
            "latest_close": float(df["close"].iloc[-1]),
            "rows": len(df),
            "1m_return_pct": float(df["close"].pct_change(20).iloc[-1] * 100) if len(df) > 20 else None,
            "3m_return_pct": float(df["close"].pct_change(60).iloc[-1] * 100) if len(df) > 60 else None,
            "volatility_30d": float(df["returns"].rolling(30).std().iloc[-1] * 100) if len(df) > 30 else None,
            "sma50": float(df["close"].rolling(50).mean().iloc[-1]) if len(df) > 50 else None,
            "sma200": float(df["close"].rolling(200).mean().iloc[-1]) if len(df) > 200 else None,
        }

        return {
            "status": "success",
            "symbol": symbol,
            "summary": summary,
            "file_path": str(file_path),
        }

    except Exception as e:
        return {"status": "error", "symbol": symbol, "message": str(e)}
    
# ---------------------------------------------------
# INDEX DATA TOOL
# ---------------------------------------------------

@tool
def get_market_indices() -> dict:
    """
    Get recent daily closing values for major Indian indices:
    - NIFTY 50 (^NSEI)
    - BSE Sensex (^BSESN)

    Useful for:
    - Market overview
    - Benchmark comparison
    - Chart visualization

    Returns:
        Dict with:
        - status
        - nifty:  list of {"date": str, "close": float}
        - sensex: list of {"date": str, "close": float}
        (last ~1 month of daily closes)
    """
    try:

        nifty = yf.download(
            "^NSEI",
            period="1mo",
            interval="1d",
            progress=False
        )

        sensex = yf.download(
            "^BSESN",
            period="1mo",
            interval="1d",
            progress=False
        )

        if nifty.empty or sensex.empty:
            return {
                "status": "error",
                "message": "Market data unavailable"
            }

        # remove multi-index columns if present
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)

        if isinstance(sensex.columns, pd.MultiIndex):
            sensex.columns = sensex.columns.get_level_values(0)

        nifty = nifty.reset_index()
        sensex = sensex.reset_index()

        nifty_data = []
        for _, row in nifty.iterrows():

            nifty_data.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": float(row["Close"])
            })

        sensex_data = []
        for _, row in sensex.iterrows():

            sensex_data.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": float(row["Close"])
            })

        return {
            "status": "success",
            "nifty": nifty_data,
            "sensex": sensex_data
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }