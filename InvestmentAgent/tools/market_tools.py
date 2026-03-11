"""
Market data tools using Yahoo Finance.
Robust implementation for AI agents.
"""

from typing import Dict, Any
import yfinance as yf
import pandas as pd

from langchain_core.tools import tool
from utils.logger import LOGGER


# ---------------------------------------------------
# SYMBOL NORMALIZATION
# ---------------------------------------------------

def normalize_symbol(symbol: str) -> str:
    """
    Normalize user provided symbol to NSE format.

    Examples
    --------
    RELIANCE -> RELIANCE.NS
    INFY.NS -> INFY.NS
    ^NSEI -> ^NSEI
    """

    if not symbol:
        return symbol

    symbol = symbol.strip().upper()

    # indices like ^NSEI
    if symbol.startswith("^"):
        return symbol

    # already NSE formatted
    if symbol.endswith(".NS"):
        return symbol

    return f"{symbol}.NS"


# ---------------------------------------------------
# CURRENT PRICE TOOL
# ---------------------------------------------------

@tool
def get_current_stock_price(symbol: str) -> Dict[str, Any]:
    """
    Fetch the latest trading price for an NSE stock.

    Uses fast_info when available, otherwise falls back
    to 1-day historical data.

    Args:
        symbol: NSE symbol (RELIANCE, INFY)

    Returns:
        Dict containing market data.
    """

    symbol = normalize_symbol(symbol)

    LOGGER.info("get_current_stock_price | symbol=%s", symbol)

    try:

        ticker = yf.Ticker(symbol)

        last_price = None
        previous_close = None
        open_price = None
        high = None
        low = None
        volume = None

        # ----------------------------
        # Attempt fast_info first
        # ----------------------------
        try:
            info = ticker.fast_info

            last_price = info.get("last_price")
            previous_close = info.get("previous_close")
            open_price = info.get("open")
            high = info.get("day_high")
            low = info.get("day_low")
            volume = info.get("last_volume")

        except Exception:
            LOGGER.warning("fast_info unavailable → using history fallback")

        # ----------------------------
        # Fallback to history
        # ----------------------------
        if last_price is None:

            hist = ticker.history(period="1d")

            if hist is None or hist.empty:
                return {
                    "status": "error",
                    "symbol": symbol,
                    "message": "No price data available"
                }

            row = hist.iloc[-1]

            last_price = float(row["Close"])
            open_price = float(row["Open"])
            high = float(row["High"])
            low = float(row["Low"])
            volume = int(row["Volume"])

        return {
            "status": "success",
            "symbol": symbol,
            "data": {
                "last_price": float(last_price),
                "previous_close": float(previous_close) if previous_close else None,
                "open": float(open_price) if open_price else None,
                "day_high": float(high) if high else None,
                "day_low": float(low) if low else None,
                "volume": int(volume) if volume else None,
                "currency": "INR"
            }
        }

    except Exception as e:

        LOGGER.exception("get_current_stock_price failed")

        return {
            "status": "error",
            "symbol": symbol,
            "message": str(e)
        }


# ---------------------------------------------------
# HISTORICAL DATA TOOL
# ---------------------------------------------------

@tool
def download_price_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    limit: int = 200
) -> Dict[str, Any]:
    """
    Download historical OHLCV data from Yahoo Finance.

    Used for:
    - technical indicators
    - trend analysis
    - backtesting
    """

    symbol = normalize_symbol(symbol)

    LOGGER.info(
        "download_price_history | symbol=%s period=%s interval=%s",
        symbol,
        period,
        interval
    )

    try:

        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False
        )

        if df is None or df.empty:
            return {
                "status": "error",
                "symbol": symbol,
                "message": "No historical price data found"
            }

        # ----------------------------
        # Handle multi-index columns
        # ----------------------------
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        df.rename(columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        }, inplace=True)

        df["date"] = df["date"].astype(str)

        df = df.tail(limit)

        records = df.to_dict("records")

        return {
            "status": "success",
            "symbol": symbol,
            "rows": len(records),
            "columns": list(df.columns),
            "data": records
        }

    except Exception as e:

        LOGGER.exception("download_price_history failed")

        return {
            "status": "error",
            "symbol": symbol,
            "message": str(e)
        }
    
# ---------------------------------------------------
# INDEX DATA TOOL
# ---------------------------------------------------

@tool
def get_market_indices() -> Dict[str, Any]:
    """
    Fetch Nifty50 and Sensex historical data
    for chart visualization.
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