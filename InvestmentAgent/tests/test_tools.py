# tests/test_tools.py

"""
Pytest suite for all tools in app/agents/tools.py

Run with:
    pytest tests/test_tools.py -v
    # or just
    pytest -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from app.agents.tools import (
    lookup_stock_symbol,
    get_nifty50_list,
    get_index_symbol,
    fetch_price_data,
    compute_technical_indicators,
    compare_returns,
    get_fundamental_data,
    search_recent_news,
    get_user_portfolio,
    top_gainers_losers,
    nifty_screener,
)

# ────────────────────────────────────────────────
# Fixtures & Helpers
# ────────────────────────────────────────────────

@pytest.fixture
def mock_yf_download():
    with patch("yfinance.download") as mock:
        yield mock


@pytest.fixture
def mock_yf_ticker():
    with patch("yfinance.Ticker") as mock:
        yield mock


@pytest.fixture
def mock_db_connection():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    with patch("app.db_connect.get_connection", return_value=mock_conn):
        yield mock_cursor


# ────────────────────────────────────────────────
# 1. lookup_stock_symbol
# ────────────────────────────────────────────────

def test_lookup_stock_symbol_found(mock_db_connection):
    # Mock DB returns two matches
    mock_db_connection.fetchall.return_value = [
        {"symbol": "HAL", "company_name": "Hindustan Aeronautics Limited", "score": 0.95},
        {"symbol": "HALFIN", "company_name": "Hindustan Finance", "score": 0.65},
    ]

    result = lookup_stock_symbol.invoke({
        "company_name_or_keyword": "Hindustan Aeronautics",
        "max_candidates": 3
    })

    assert result["status"] == "success"
    assert len(result["matches"]) == 2
    assert result["matches"][0]["symbol"] == "HAL"
    assert result["matches"][0]["score"] >= 0.9


def test_lookup_stock_symbol_no_match(mock_db_connection):
    mock_db_connection.fetchall.return_value = []

    result = lookup_stock_symbol.invoke({
        "company_name_or_keyword": "NonExistentCompanyXYZ",
        "max_candidates": 5
    })

    assert result["status"] == "success"
    assert len(result["matches"]) == 0


# ────────────────────────────────────────────────
# 2. get_nifty50_list
# ────────────────────────────────────────────────

def test_get_nifty50_list_success(mock_db_connection):
    mock_db_connection.fetchall.return_value = [
        {"symbol": "RELIANCE", "company_name": "Reliance Industries Limited"},
        {"symbol": "TCS", "company_name": "Tata Consultancy Services Limited"},
    ]

    result = get_nifty50_list.invoke({})

    assert result["status"] == "success"
    assert result["count"] == 2
    assert len(result["stocks"]) == 2
    assert result["stocks"][0]["symbol"] == "RELIANCE"


def test_get_nifty50_list_fallback(mock_db_connection):
    mock_db_connection.fetchall.side_effect = Exception("DB error")

    result = get_nifty50_list.invoke({})

    assert result["status"] == "fallback"
    assert "count" in result
    assert len(result["stocks"]) > 0


# ────────────────────────────────────────────────
# 3. get_index_symbol
# ────────────────────────────────────────────────

@pytest.mark.parametrize("input_name, expected_ticker", [
    ("nifty 50", "^NSEI"),
    ("Nifty", "^NSEI"),
    ("Bank Nifty", "^NSEBANK"),
    ("sensex", "^BSESN"),
    ("unknown index", None),
])
def test_get_index_symbol(input_name, expected_ticker):
    result = get_index_symbol.invoke({"index_name": input_name})

    if expected_ticker:
        assert result["status"] == "found"
        assert result["ticker"] == expected_ticker
    else:
        assert result["status"] == "not_found"
        assert result["ticker"] is None


# ────────────────────────────────────────────────
# 4. fetch_price_data
# ────────────────────────────────────────────────

def test_fetch_price_data_success(mock_yf_download):
    # Mock successful dataframe
    mock_df = pd.DataFrame({
        "Open": [100, 102],
        "High": [105, 106],
        "Low": [98, 99],
        "Close": [101, 104],
        "Volume": [1000, 1200]
    }, index=pd.date_range("2025-01-01", periods=2))

    mock_yf_download.return_value = mock_df

    result = fetch_price_data.invoke({"symbol": "RELIANCE", "days_back": 2})

    assert result["status"] == "success"
    assert "data" in result
    assert len(result["data"]) <= 2
    assert "close" in result["data"][0]


def test_fetch_price_data_empty(mock_yf_download):
    mock_yf_download.return_value = pd.DataFrame()

    result = fetch_price_data.invoke({"symbol": "INVALID", "days_back": 10})

    assert result["status"] == "error"


# ────────────────────────────────────────────────
# 5. compute_technical_indicators
# ────────────────────────────────────────────────

def test_compute_technical_indicators_basic():
    fake_data = [
        {"date": "2025-01-01", "close": 100.0},
        {"date": "2025-01-02", "close": 102.0},
        {"date": "2025-01-03", "close": 101.0},
    ] * 30  # enough rows for SMA 20 + RSI 14

    result = compute_technical_indicators.invoke({
        "price_data": fake_data,
        "rsi_period": 14,
        "sma_periods": [20]
    })

    assert result["status"] == "success"
    assert "rsi" in result
    assert "sma" in result
    assert "sma_sma_20" in result["sma"]


def test_compute_technical_indicators_too_few_rows():
    fake_data = [{"date": "2025-01-01", "close": 100.0}] * 10

    result = compute_technical_indicators.invoke({
        "price_data": fake_data
    })

    assert result["status"] == "error"
    assert "few rows" in result["message"].lower()


# ────────────────────────────────────────────────
# 6. compare_returns
# ────────────────────────────────────────────────

def test_compare_returns_basic(mock_yf_download):
    # Mock two symbols
    df1 = pd.DataFrame({"Close": [100, 110]}, index=pd.date_range("2025-01-01", periods=2))
    df2 = pd.DataFrame({"Close": [200, 210]}, index=pd.date_range("2025-01-01", periods=2))

    mock_yf_download.side_effect = [df1, df2]

    result = compare_returns.invoke({
        "symbols": ["LT", "^NSEI"],
        "days_back": 2
    })

    assert result["status"] == "success"
    assert len(result["comparison"]) == 2
    assert all("return_pct" in item for item in result["comparison"])


# ────────────────────────────────────────────────
# 7. get_fundamental_data
# ────────────────────────────────────────────────

def test_get_fundamental_data_success(mock_yf_ticker):
    mock_info = {
        "longName": "Hindustan Aeronautics Limited",
        "sector": "Industrials",
        "marketCap": 1500000000000,
        "trailingPE": 28.5,
    }
    mock_ticker = MagicMock()
    mock_ticker.info = mock_info
    mock_yf_ticker.return_value = mock_ticker

    result = get_fundamental_data.invoke({"symbol": "HAL"})

    assert result["status"] == "success"
    assert result["company_name"] == "Hindustan Aeronautics Limited"
    assert "market_cap" in result


# ────────────────────────────────────────────────
# 8. search_recent_news
# ────────────────────────────────────────────────

def test_search_recent_news_basic():
    # Mock DuckDuckGo (hard to mock fully, but basic check)
    with patch("langchain_community.tools.DuckDuckGoSearchRun.invoke") as mock_search:
        mock_search.return_value = "Some news title https://example.com\nAnother https://news.com"

        result = search_recent_news.invoke({
            "query": "HAL Hindustan Aeronautics",
            "max_results": 2
        })

    assert result["status"] == "success"
    assert "results" in result
    assert len(result["results"]) <= 2


# ────────────────────────────────────────────────
# 9. get_user_portfolio
# ────────────────────────────────────────────────

def test_get_user_portfolio_no_file():
    with patch("os.path.exists", return_value=False):
        result = get_user_portfolio.invoke({})

    assert result["status"] == "no_data"
    assert "Portfolio file not found" in result["message"]


# ────────────────────────────────────────────────
# 10. top_gainers_losers
# ────────────────────────────────────────────────

def test_top_gainers_losers_basic(mock_yf_ticker):
    mock_hist = pd.DataFrame({
        "Close": [100, 110]
    }, index=pd.date_range("2025-01-01", periods=2))

    mock_t = MagicMock()
    mock_t.history.return_value = mock_hist

    with patch("yfinance.Ticker", return_value=mock_t):
        result = top_gainers_losers.invoke({
            "symbols": ["HAL", "LT"]
        })

    assert result["status"] == "success"
    assert "gainers" in result
    assert "losers" in result


# ────────────────────────────────────────────────
# 11. nifty_screener
# ────────────────────────────────────────────────

def test_nifty_screener_basic(mock_yf_ticker):
    mock_hist = pd.DataFrame({
        "Close": [24000, 24500]
    }, index=pd.date_range("2025-01-01", periods=2))

    mock_t = MagicMock()
    mock_t.history.return_value = mock_hist

    with patch("yfinance.Ticker", return_value=mock_t):
        result = nifty_screener.invoke({})

    assert result["status"] == "success"
    assert "price" in result
    assert "change_percent" in result