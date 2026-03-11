INDEX_MAP = {
    "nifty": "^NSEI",
    "nifty 50": "^NSEI",
    "sensex": "^BSESN",
    "bank nifty": "^NSEBANK",
}

COMPANY_MAP = {
    "l&t": "LT",
    "larsen": "LT",
    "larsen toubro": "LT",
    "tcs": "TCS",
    "reliance": "RELIANCE",
    "infosys": "INFY",
    "hindustan aeronautics": "HAL",
    "hal": "HAL",
}


def resolve_symbols(query: str) -> list[str]:
    """
    Resolve stock/index symbols from natural language query.

    Returns Yahoo Finance compatible tickers.
    Example:
        "Compare HAL vs Nifty" → ["^NSEI", "HAL.NS"]
    """

    query_lower = query.lower()

    symbols = set()

    # ── Detect indices ─────────────────────
    for key, ticker in INDEX_MAP.items():
        if key in query_lower:
            symbols.add(ticker)

    # ── Detect companies ───────────────────
    for key, ticker in COMPANY_MAP.items():
        if key in query_lower:
            symbols.add(f"{ticker}.NS")

    return list(symbols)