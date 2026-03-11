from langchain_core.tools import tool
import yfinance as yf
from typing import Dict, Any
from utils.logger import LOGGER
from tools.market_tools import normalize_symbol

@tool
def get_52_week_range(symbol: str) -> Dict[str, Any]:
    """
    Fetch 52-week high/low and current distance.

    Args:
        symbol: NSE stock symbol (RELIANCE, INFY)

    Returns:
        52 week range statistics.
    """
    symbol = normalize_symbol(symbol)

    LOGGER.info("get_52_week_range | symbol=%s", symbol)

    try:

        ticker = yf.Ticker(symbol)
        info = ticker.info

        high = info.get("fiftyTwoWeekHigh")
        low = info.get("fiftyTwoWeekLow")
        price = info.get("currentPrice")

        if not high or not low or not price:
            return {"status": "error", "message": "Missing price data"}

        return {
            "status": "success",
            "symbol": symbol,
            "current_price": price,
            "52w_high": high,
            "52w_low": low,
            "distance_from_high_pct": ((high - price) / high) * 100,
            "distance_from_low_pct": ((price - low) / low) * 100
        }

    except Exception as e:

        LOGGER.exception("get_52_week_range failed")

        return {
            "status": "error",
            "message": str(e)
        }