
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