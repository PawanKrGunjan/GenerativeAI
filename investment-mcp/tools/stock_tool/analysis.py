"""
Analysis module for the share market tool.
"""

from typing import Any

import pandas as pd

from mcp_instance import mcp

from .stock_info import (
    _INTERVALS,
    _PERIODS,
    _clean_value,
    _dataframe_records,
    _ticker,
)

_PERIODS_PER_YEAR = {
    "1m": 252 * 390,
    "2m": 252 * 195,
    "5m": 252 * 78,
    "15m": 252 * 26,
    "30m": 252 * 13,
    "60m": 252 * 6.5,
    "90m": 252 * 4,
    "1h": 252 * 6.5,
    "1d": 252,
    "5d": 52,
    "1wk": 52,
    "1mo": 12,
    "3mo": 4,
}


def _statement_records(statement: pd.DataFrame) -> list[dict[str, Any]]:
    if statement.empty:
        return []
    transposed = statement.T.copy()
    transposed.index.name = "period"
    return _dataframe_records(transposed)


def _historical_close(
    ticker: str, period: str, interval: str, market: str
) -> tuple[str, pd.DataFrame] | dict[str, str]:
    if period not in _PERIODS:
        return {"error": f"Unsupported period '{period}'. Choose from {sorted(_PERIODS)}."}
    if interval not in _INTERVALS:
        return {"error": f"Unsupported interval '{interval}'. Choose from {sorted(_INTERVALS)}."}

    symbol, stock = _ticker(ticker, market)
    history = stock.history(period=period, interval=interval)
    if history.empty or "Close" not in history:
        return {"error": f"No closing-price history found for ticker: {symbol}"}
    history = history.loc[history["Close"].notna()].copy()
    if len(history) < 2:
        return {"error": f"At least two closing prices are required for ticker: {symbol}"}
    return symbol, history


def _latest_number(series: pd.Series) -> float | None:
    values = series.dropna()
    return float(values.iloc[-1]) if not values.empty else None


@mcp.tool()
def get_stock_fundamentals(
    ticker: str,
    frequency: str = "annual",
    market: str = "india",
) -> dict:
    """Get company valuation metrics and financial statements.

    Frequency may be "annual" or "quarterly". Bare tickers default to NSE India;
    use market="bse" or market="global" for other symbols.
    """
    try:
        normalized_frequency = frequency.lower()
        if normalized_frequency not in {"annual", "quarterly"}:
            return {"error": "frequency must be 'annual' or 'quarterly'."}
        symbol, stock = _ticker(ticker, market)
        info = stock.info
        statement_prefix = "quarterly_" if normalized_frequency == "quarterly" else ""
        statements = {
            "income_statement": getattr(stock, f"{statement_prefix}income_stmt"),
            "balance_sheet": getattr(stock, f"{statement_prefix}balance_sheet"),
            "cash_flow": getattr(stock, f"{statement_prefix}cashflow"),
        }
        if not info and not any(not statement.empty for statement in statements.values()):
            return {"error": f"No fundamentals found for ticker: {symbol}"}

        info_fields = (
            "shortName", "longName", "sector", "industry", "country", "currency",
            "marketCap", "enterpriseValue", "trailingPE", "forwardPE", "priceToBook",
            "pegRatio", "profitMargins", "operatingMargins", "returnOnAssets",
            "returnOnEquity", "revenueGrowth", "earningsGrowth", "totalRevenue",
            "grossProfits", "ebitda", "totalDebt", "totalCash", "freeCashflow",
            "dividendYield",
        )
        fundamentals = {
            field: _clean_value(info[field]) for field in info_fields if field in info
        }
        return {
            "ticker": symbol,
            "frequency": normalized_frequency,
            "metrics": fundamentals,
            "financial_statements": {
                name: _statement_records(statement)
                for name, statement in statements.items()
            },
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_stock_technical_indicators(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    market: str = "india",
    sma_window: int = 20,
    rsi_window: int = 14,
) -> dict:
    """Calculate common technical indicators from historical stock prices.

    Returns latest SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and volume average
    where the source data supports them. Indicator values are descriptive, not
    trading recommendations. Daily intervals are recommended for these windows.
    """
    try:
        if sma_window < 2 or rsi_window < 2:
            return {"error": "sma_window and rsi_window must be at least 2."}
        result = _historical_close(ticker, period, interval, market)
        if isinstance(result, dict):
            return result
        symbol, history = result
        close = history["Close"].astype(float)
        sma = close.rolling(sma_window, min_periods=sma_window).mean()
        ema = close.ewm(span=sma_window, adjust=False, min_periods=sma_window).mean()

        price_changes = close.diff()
        gains = price_changes.clip(lower=0)
        losses = -price_changes.clip(upper=0)
        average_gain = gains.ewm(
            alpha=1 / rsi_window, adjust=False, min_periods=rsi_window
        ).mean()
        average_loss = losses.ewm(
            alpha=1 / rsi_window, adjust=False, min_periods=rsi_window
        ).mean()
        relative_strength = average_gain / average_loss
        rsi = 100 - (100 / (1 + relative_strength))
        rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
        rsi = rsi.mask((average_gain == 0) & (average_loss > 0), 0)

        middle_band = close.rolling(20, min_periods=20).mean()
        band_width = close.rolling(20, min_periods=20).std()
        macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(
            span=26, adjust=False
        ).mean()
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()

        atr_value: float | None = None
        if {"High", "Low"}.issubset(history.columns):
            previous_close = close.shift()
            true_range = pd.concat(
                [
                    history["High"].astype(float) - history["Low"].astype(float),
                    (history["High"].astype(float) - previous_close).abs(),
                    (history["Low"].astype(float) - previous_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr_value = _latest_number(
                true_range.ewm(
                    alpha=1 / 14, adjust=False, min_periods=14
                ).mean()
            )

        indicators: dict[str, float | None] = {
            f"sma_{sma_window}": _latest_number(sma),
            f"ema_{sma_window}": _latest_number(ema),
            f"rsi_{rsi_window}": _latest_number(rsi),
            "macd": _latest_number(macd_line),
            "macd_signal": _latest_number(macd_signal),
            "macd_histogram": _latest_number(macd_line - macd_signal),
            "bollinger_middle_20": _latest_number(middle_band),
            "bollinger_upper_20": _latest_number(middle_band + 2 * band_width),
            "bollinger_lower_20": _latest_number(middle_band - 2 * band_width),
            "atr_14": atr_value,
        }
        if "Volume" in history:
            indicators["average_volume_20"] = _latest_number(
                history["Volume"].rolling(20, min_periods=20).mean()
            )

        return {
            "ticker": symbol,
            "period": period,
            "interval": interval,
            "latest_close": float(close.iloc[-1]),
            "indicators": indicators,
            "calculation_note": (
                "Indicators describe historical prices and are not trading advice. "
                "Values are null when there is insufficient history."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_stock_return_metrics(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    market: str = "india",
) -> dict:
    """Calculate total and annualized price return over a historical period.

    Uses Yahoo Finance's Close series (adjusted by default in current yfinance
    versions). Annualized return is the compound growth rate over the observed dates.
    """
    try:
        result = _historical_close(ticker, period, interval, market)
        if isinstance(result, dict):
            return result
        symbol, history = result
        close = history["Close"].astype(float)
        start_price = float(close.iloc[0])
        end_price = float(close.iloc[-1])
        if start_price <= 0 or end_price <= 0:
            return {"error": "Closing prices must be positive to calculate returns."}
        total_return = end_price / start_price - 1
        elapsed_days = (history.index[-1] - history.index[0]).total_seconds() / 86400
        annualized_return = (
            (end_price / start_price) ** (365.25 / elapsed_days) - 1
            if elapsed_days > 0
            else None
        )
        return {
            "ticker": symbol,
            "period": period,
            "interval": interval,
            "observations": len(close),
            "start_date": _clean_value(history.index[0]),
            "end_date": _clean_value(history.index[-1]),
            "start_close": start_price,
            "end_close": end_price,
            "total_return_pct": total_return * 100,
            "annualized_return_pct": (
                annualized_return * 100 if annualized_return is not None else None
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_stock_risk_metrics(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    market: str = "india",
    risk_free_rate_pct: float = 0.0,
) -> dict:
    """Calculate historical volatility, drawdown, Sharpe/Sortino, and VaR metrics.

    The risk-free rate is an annual percentage (for example, 6.5 for 6.5%).
    Annualization uses conventional trading-period counts for the selected interval.
    These are historical measures, not forecasts or investment recommendations.
    """
    try:
        if interval not in _PERIODS_PER_YEAR:
            supported = sorted(_PERIODS_PER_YEAR)
            return {
                "error": (
                    f"Risk metrics do not support interval '{interval}'. "
                    f"Choose from {supported}."
                )
            }
        result = _historical_close(ticker, period, interval, market)
        if isinstance(result, dict):
            return result
        symbol, history = result
        close = history["Close"].astype(float)
        if (close <= 0).any():
            return {"error": "Closing prices must be positive to calculate risk metrics."}

        returns = close.pct_change().dropna()
        if len(returns) < 2:
            return {"error": "At least three closing prices are required for risk metrics."}
        periods_per_year = _PERIODS_PER_YEAR[interval]
        periodic_risk_free_rate = (risk_free_rate_pct / 100) / periods_per_year
        excess_returns = returns - periodic_risk_free_rate
        volatility = float(returns.std(ddof=1))
        downside_returns = excess_returns.clip(upper=0)
        downside_deviation = float((downside_returns.pow(2).mean()) ** 0.5)
        max_drawdown = float((close / close.cummax() - 1).min())
        var_95 = float(returns.quantile(0.05))
        tail_returns = returns[returns <= var_95]
        sortino = (
            float(excess_returns.mean() / downside_deviation * periods_per_year**0.5)
            if downside_deviation > 0
            else None
        )
        sharpe = (
            float(excess_returns.mean() / volatility * periods_per_year**0.5)
            if volatility > 0
            else None
        )

        return {
            "ticker": symbol,
            "period": period,
            "interval": interval,
            "observations": len(returns),
            "annualized_volatility_pct": volatility * periods_per_year**0.5 * 100,
            "downside_deviation_pct": downside_deviation * periods_per_year**0.5 * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_drawdown * 100,
            "period_var_95_pct": var_95 * 100,
            "period_expected_shortfall_95_pct": float(tail_returns.mean() * 100),
            "best_period_return_pct": float(returns.max() * 100),
            "worst_period_return_pct": float(returns.min() * 100),
            "risk_free_rate_pct": risk_free_rate_pct,
            "calculation_note": (
                "Historical risk statistics are not forecasts. VaR and expected "
                "shortfall use the selected observation interval."
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}
