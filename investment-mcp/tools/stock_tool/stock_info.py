"""Yahoo Finance-backed stock market MCP tools."""

from datetime import date, datetime
from typing import Any

import pandas as pd
import yfinance as yf

from mcp_instance import mcp

_MARKET_SUFFIXES = {
    "india": ".NS",
    "bse": ".BO",
    "global": "",
}
_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo",
}


def _resolve_symbol(ticker: str, market: str) -> str:
    if not ticker.strip():
        raise ValueError("ticker must not be empty.")
    normalized_market = market.lower()
    if normalized_market not in _MARKET_SUFFIXES:
        raise ValueError("market must be one of: india, bse, global.")
    if "." in ticker:
        return ticker.upper()
    return f"{ticker.upper()}{_MARKET_SUFFIXES[normalized_market]}"


def _ticker(ticker: str, market: str) -> tuple[str, Any]:
    symbol = _resolve_symbol(ticker, market)
    return symbol, yf.Ticker(symbol)


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in dataframe.reset_index().to_dict(orient="records"):
        records.append({str(key): _clean_value(value) for key, value in row.items()})
    return records


@mcp.tool()
def get_stock_quote(ticker: str, market: str = "india") -> dict:
    """Get the latest available quote and selected metrics for a stock.

    Bare tickers default to NSE India (for example, RELIANCE becomes RELIANCE.NS).
    Set market to "bse" for .BO symbols or "global" for unsuffixed global symbols.
    An explicit Yahoo Finance suffix (for example, RELIANCE.NS or 7203.T) is preserved.
    """
    try:
        symbol, stock = _ticker(ticker, market)
        info = stock.info
        if not info:
            return {"error": f"No quote data found for ticker: {symbol}"}
        fields = (
            "shortName", "longName", "symbol", "currency", "exchange",
            "quoteType", "currentPrice", "regularMarketPrice", "previousClose",
            "open", "dayLow", "dayHigh", "fiftyTwoWeekLow", "fiftyTwoWeekHigh",
            "volume", "marketCap", "trailingPE", "forwardPE", "dividendYield",
        )
        quote = {field: _clean_value(info[field]) for field in fields if field in info}
        return {"ticker": symbol, "quote": quote}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_stock_history(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
    market: str = "india",
) -> dict:
    """Get historical OHLCV price data for a Yahoo Finance stock ticker.

    Supports Yahoo Finance periods and intervals. Bare tickers default to NSE India;
    use market="bse" or market="global" for other symbols.
    """
    try:
        if period not in _PERIODS:
            return {"error": f"Unsupported period '{period}'. Choose from {sorted(_PERIODS)}."}
        if interval not in _INTERVALS:
            return {"error": f"Unsupported interval '{interval}'. Choose from {sorted(_INTERVALS)}."}
        symbol, stock = _ticker(ticker, market)
        history = stock.history(period=period, interval=interval)
        if history.empty:
            return {"error": f"No price history found for ticker: {symbol}"}
        return {
            "ticker": symbol,
            "period": period,
            "interval": interval,
            "data": _dataframe_records(history),
        }
    except Exception as exc:
        return {"error": str(exc)}
