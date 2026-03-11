"""
run.py
Main orchestration file for Investment App + Terminal Chat
"""

import time
import pandas as pd
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from psycopg.rows import dict_row
from utils.config import NIFTY_50_FILE, ALL_NSE_SYMBOLS
from utils.logger import LOGGER
from utils.db_connect import get_connection, initialize_database
from app.market_data import nse_builder, nifty_builder
from app.market_data.price_loader import load_nifty50_prices
from app.analysis.indicators import add_rsi, add_sma
from app.analysis.screener import bullish_rsi_screener
from app.analytics.market_analysis import (
    get_green_stocks,
    get_red_stocks,
    get_neutral_stocks,
    get_top_gainers,
    get_top_losers,
)
from app.analytics.metrics import compute_metrics

from agents.investment_agent import agent



IST = ZoneInfo("Asia/Kolkata")


# ────────────────────────────────────────────────
# SETUP DATABASE & MASTER TABLES
# ────────────────────────────────────────────────
def setup_environment(log):
    log.info("🔧 Initializing Database...")
    initialize_database(log)

    log.info("📥 Loading NSE Master Data...")
    nse_builder.build_nse_master_table(ALL_NSE_SYMBOLS, log)

    log.info("🏷 Tagging NIFTY 50 Stocks...")
    nifty_builder.build_nifty_knowledge_base(NIFTY_50_FILE, log)


# ────────────────────────────────────────────────
# ANALYZE NIFTY 50 TECHNICALS + METRICS
# ────────────────────────────────────────────────
def analyze_nifty50(log):
    log.info("📊 Running Technical Analysis + Metrics from Stored DB Data")

    with get_connection(log) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT symbol FROM nifty50_stocks;")
            symbols = [row["symbol"] for row in cur.fetchall()]

            all_metrics = {}

            for symbol in symbols:
                try:
                    cur.execute("""
                        SELECT trade_date, open, high, low, close, volume
                        FROM stock_prices
                        WHERE symbol = %s
                        ORDER BY trade_date ASC;
                    """, (symbol,))
                    rows = cur.fetchall()

                    if not rows:
                        log.warning(f"No price data found for {symbol}")
                        continue

                    df = pd.DataFrame(rows)
                    df.columns = df.columns.str.lower()
                    df.rename(columns={"trade_date": "date"}, inplace=True)

                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = df[col].astype(float)

                    df = add_sma(df)
                    df = add_rsi(df)
                    signal = bullish_rsi_screener(df)
                    log.info(f"{symbol} → {signal}")

                    try:
                        metrics = compute_metrics(df)
                        log.info(
                            f"{symbol} metrics → Return: {metrics['return_pct']}%, "
                            f"Volatility: {metrics['volatility_pct']}%, "
                            f"Sharpe: {metrics['sharpe']}, "
                            f"Max Drawdown: {metrics['max_drawdown_pct']}%"
                        )
                        all_metrics[symbol] = metrics
                    except Exception as me:
                        log.warning(f"Metrics failed for {symbol}: {me}")

                except Exception as e:
                    log.warning(f"Analysis failed for {symbol}: {e}")

    return all_metrics

# ────────────────────────────────────────────────
# PRINT MARKET SUMMARY
# ────────────────────────────────────────────────
def print_market_summary(log):
    greens = get_green_stocks(log)
    reds = get_red_stocks(log)
    neutrals = get_neutral_stocks(log)
    gainers = get_top_gainers(log, limit=10)
    losers = get_top_losers(log, limit=10)

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("📊 NIFTY 50 Market Summary")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    log.info(f"🟢 Green Stocks   : {len(greens)}")
    log.info(f"🔴 Red Stocks     : {len(reds)}")
    log.info(f"⚪ Neutral Stocks : {len(neutrals)}\n")

    log.info("📈 Top 5 Gainers")
    for row in gainers:
        log.info(f"{row['company_name']} ({row['symbol']}) → +{row['pct_gain']:.2f}%")

    log.info("\n📉 Top 5 Losers")
    for row in losers:
        log.info(f"{row['company_name']} ({row['symbol']}) → {row['pct_loss']:.2f}%")

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ────────────────────────────────────────────────
# TERMINAL CHAT INTERFACE
# ────────────────────────────────────────────────
def terminal_chat():
    print("\n" + "═" * 70)
    print("  Investment Agent Terminal Chat")
    print("  Type your query (e.g. 'Is TCS bullish?', 'Screen NIFTY50 RSI < 30')")
    print("  Type 'exit', 'quit', or 'q' to stop")
    print("═" * 70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nGoodbye! 👋\n")
                break

            if not user_input:
                continue
            result = agent.run(query=user_input)

            print("\nAgent:")
            print(result["answer"])

            if result.get("memory_summary"):
                print("\n" + result["memory_summary"])

            print("-" * 70)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit cleanly.")
        except Exception as e:
            print(f"\nError: {str(e)}")


# ────────────────────────────────────────────────
# MAIN EXECUTION FLOW
# ────────────────────────────────────────────────
def main():
    start_time = time.time()
    LOGGER.info("🚀 Starting Investment App")

    try:

        # 2. Load latest NIFTY 50 prices
        LOGGER.info("💹 Loading Latest NIFTY 50 Prices...")
        load_nifty50_prices(LOGGER, period="ytd")

        # 3. Run classical technical analysis
        all_metrics = analyze_nifty50(LOGGER)

        if all_metrics:
            df_metrics = pd.DataFrame(all_metrics).T
            print("\nTop 10 Performers (by total return):")
            print(df_metrics.sort_values("return_pct", ascending=False).head(10))

        # 4. Market breadth + summary
        print_market_summary(LOGGER)

        elapsed = round(time.time() - start_time, 2)
        LOGGER.info(f"✅ Batch execution completed in {elapsed} seconds")

        # 5. Start interactive terminal chat
        print("\n" + "=" * 70)
        print("Batch processing complete. Starting interactive chat mode...")
        print("=" * 70)
        terminal_chat()

    except Exception as e:
        LOGGER.exception(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    main()