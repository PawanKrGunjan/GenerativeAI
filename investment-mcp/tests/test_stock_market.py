"""Offline tests for Yahoo Finance stock-market tools."""

import unittest
from unittest.mock import patch

import pandas as pd

from tools.stock_tool import analysis, stock_info
import server as _server  # noqa: F401 - importing registers MCP tools


class StockMarketToolTests(unittest.TestCase):
    def test_quote_defaults_bare_ticker_to_nse(self) -> None:
        ticker = type("TickerStub", (), {"info": {"shortName": "Example Ltd"}})()
        with patch.object(stock_info.yf, "Ticker", return_value=ticker) as create_ticker:
            result = stock_info.get_stock_quote("example")

        create_ticker.assert_called_once_with("EXAMPLE.NS")
        self.assertEqual(result["ticker"], "EXAMPLE.NS")
        self.assertEqual(result["quote"]["shortName"], "Example Ltd")

    def test_history_returns_serializable_price_records(self) -> None:
        history = pd.DataFrame(
            {"Close": [100.5], "Volume": [1200]},
            index=pd.DatetimeIndex(["2026-01-02"], name="Date"),
        )
        ticker = type("TickerStub", (), {"history": lambda self, **kwargs: history})()
        with patch.object(stock_info.yf, "Ticker", return_value=ticker):
            result = stock_info.get_stock_history("RELIANCE", period="5d")

        self.assertEqual(result["ticker"], "RELIANCE.NS")
        self.assertEqual(
            result["data"],
            [{"Date": "2026-01-02T00:00:00", "Close": 100.5, "Volume": 1200}],
        )

    def test_fundamentals_support_quarterly_statements(self) -> None:
        statement = pd.DataFrame(
            {pd.Timestamp("2025-12-31"): [42]}, index=["Total Revenue"]
        )
        ticker = type(
            "TickerStub",
            (),
            {
                "info": {"sector": "Technology"},
                "quarterly_income_stmt": statement,
                "quarterly_balance_sheet": pd.DataFrame(),
                "quarterly_cashflow": pd.DataFrame(),
            },
        )()
        with patch.object(stock_info.yf, "Ticker", return_value=ticker):
            result = analysis.get_stock_fundamentals(
                "AAPL", frequency="quarterly", market="global"
            )

        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["frequency"], "quarterly")
        self.assertEqual(
            result["financial_statements"]["income_statement"],
            [{"period": "2025-12-31T00:00:00", "Total Revenue": 42}],
        )

    def test_invalid_market_returns_error(self) -> None:
        result = stock_info.get_stock_quote("EXAMPLE", market="unknown")

        self.assertIn("market must be one of", result["error"])

    def test_technical_indicators_calculate_rsi_and_atr(self) -> None:
        close = pd.Series(range(100, 130), dtype=float)
        history = pd.DataFrame(
            {
                "Close": close.to_numpy(),
                "High": close.to_numpy() + 1,
                "Low": close.to_numpy() - 1,
                "Volume": [1000] * len(close),
            },
            index=pd.date_range("2026-01-01", periods=len(close), freq="D"),
        )
        ticker = type("TickerStub", (), {"history": lambda self, **kwargs: history})()
        with patch.object(analysis, "_ticker", return_value=("EXAMPLE.NS", ticker)):
            result = analysis.get_stock_technical_indicators("EXAMPLE")

        self.assertEqual(result["ticker"], "EXAMPLE.NS")
        self.assertEqual(result["indicators"]["rsi_14"], 100.0)
        self.assertAlmostEqual(result["indicators"]["atr_14"], 2.0)
        self.assertEqual(result["indicators"]["average_volume_20"], 1000.0)

    def test_return_metrics_report_total_and_annualized_return(self) -> None:
        history = pd.DataFrame(
            {"Close": [100.0, 110.0]},
            index=pd.to_datetime(["2026-01-01", "2027-01-01"]),
        )
        ticker = type("TickerStub", (), {"history": lambda self, **kwargs: history})()
        with patch.object(analysis, "_ticker", return_value=("EXAMPLE.NS", ticker)):
            result = analysis.get_stock_return_metrics("EXAMPLE")

        self.assertAlmostEqual(result["total_return_pct"], 10.0)
        self.assertAlmostEqual(result["annualized_return_pct"], 10.0, places=1)

    def test_risk_metrics_return_volatility_and_drawdown(self) -> None:
        history = pd.DataFrame(
            {"Close": [100.0, 110.0, 99.0, 108.9]},
            index=pd.date_range("2026-01-01", periods=4, freq="D"),
        )
        ticker = type("TickerStub", (), {"history": lambda self, **kwargs: history})()
        with patch.object(analysis, "_ticker", return_value=("EXAMPLE.NS", ticker)):
            result = analysis.get_stock_risk_metrics("EXAMPLE")

        self.assertAlmostEqual(result["max_drawdown_pct"], -10.0)
        self.assertIsNotNone(result["annualized_volatility_pct"])
        self.assertIsNotNone(result["sharpe_ratio"])

    def test_analysis_rejects_unsupported_risk_interval(self) -> None:
        result = analysis.get_stock_risk_metrics("EXAMPLE", interval="2d")

        self.assertIn("Risk metrics do not support interval", result["error"])


if __name__ == "__main__":
    unittest.main()
