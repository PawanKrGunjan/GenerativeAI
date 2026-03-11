"""
app/analysis/indicators.py
"""

import pandas as pd


def add_sma(df: pd.DataFrame, period=50):
    df[f"SMA_{period}"] = df["close"].rolling(period).mean()
    return df


def add_rsi(df: pd.DataFrame, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def bullish_rsi_screener(df):
    """
    Evaluate the latest RSI of a stock and provide
    an easy-to-understand signal.

    RSI < 30 → Buy Opportunity
    RSI > 70 → Sell Pressure
    Otherwise → Stable
    """
    latest = df.iloc[-1]

    if latest["RSI"] < 30:
        return "Buy Opportunity"
    elif latest["RSI"] > 70:
        return "Sell Pressure"
    else:
        return "Neutral"