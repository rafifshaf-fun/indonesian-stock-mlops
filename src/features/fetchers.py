"""
Data fetchers: fundamentals, USD/IDR, FRED macro, BI Rate, Google Trends.
"""
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# sys.path trick so submodules can import from src/
import sys as _sys
_curr = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_curr)
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)

from config import (
    FRED_SERIES, TICKER_SEARCH_TERMS, CACHE_CONFIG, get_logger,
)

logger = get_logger(__name__)

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logger.warning("fredapi not installed — run: pip install fredapi")

try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except ImportError:
    TRENDS_AVAILABLE = False
    logger.warning("pytrends not installed — run: pip install pytrends")


def fetch_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio": info.get("trailingPE", np.nan),
            "pb_ratio": info.get("priceToBook", np.nan),
            "dividend_yield": info.get("dividendYield", np.nan) or 0.0,
            "market_cap": info.get("marketCap", np.nan),
            "debt_to_equity": info.get("debtToEquity", np.nan),
            "return_on_equity": info.get("returnOnEquity", np.nan),
            "revenue_growth": info.get("revenueGrowth", np.nan),
            "profit_margins": info.get("profitMargins", np.nan),
        }
    except Exception:
        return {k: np.nan for k in ["pe_ratio", "pb_ratio", "dividend_yield", "market_cap", "debt_to_equity", "return_on_equity", "revenue_growth", "profit_margins"]}


def fetch_usdidr(start: str, end: str) -> pd.Series:
    try:
        usdidr = yf.download("IDR=X", start=start, end=end, progress=False)
        series = usdidr["Close"].squeeze()
        series.name = "usdidr_rate"
        ret_series = series.pct_change()
        ret_series.name = "usdidr_return"
        return pd.concat([series, ret_series], axis=1)
    except Exception:
        return pd.DataFrame(columns=["usdidr_rate", "usdidr_return"])


def fetch_fred_macro(start: str, end: str) -> pd.DataFrame:
    api_key = os.environ.get("FRED_API_KEY")
    if not FRED_AVAILABLE or not api_key:
        return pd.DataFrame()
    try:
        fred = Fred(api_key=api_key)
        frames = {}
        for name, series_id in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start, observation_end=end)
                s.name = name
                frames[name] = s
            except Exception:
                pass

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames.values(), axis=1)
        df.index = pd.to_datetime(df.index)
        # PREVENT LOOKAHEAD BIAS: shift 1 day
        df = df.shift(1).resample("D").last().ffill().bfill()
        return df
    except Exception:
        return pd.DataFrame()


def fetch_bi_rate(start: str, end: str) -> pd.Series:
    try:
        url = "https://www.bi.go.id/en/statistik/indikator/bi-rate.aspx"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(resp.text)
        for table in tables:
            if table.shape[1] >= 12:
                table.columns = [str(c) for c in table.columns]
                melted = table.melt(id_vars=table.columns[0], var_name="Month", value_name="bi_rate")
                melted.columns = ["Year", "Month", "bi_rate"]
                melted = melted.dropna(subset=["bi_rate"])
                melted["date"] = pd.to_datetime(melted["Year"].astype(str) + "-" + melted["Month"].astype(str) + "-01", errors="coerce")
                melted = melted.dropna(subset=["date"])

                series = melted.set_index("date")["bi_rate"]
                series = pd.to_numeric(series, errors="coerce").dropna()
                # Shift monthly release to next month for lookahead safety
                series.index = series.index + pd.DateOffset(months=1)
                series = series.resample("D").last().ffill().bfill()
                series.name = "bi_rate_official"
                return series
    except Exception:
        pass
    return pd.Series(name="bi_rate_official", dtype=float)


def fetch_google_trends(ticker: str, start: str, end: str) -> pd.Series:
    if not TRENDS_AVAILABLE:
        return pd.Series(name="google_trend", dtype=float)
    search_term = TICKER_SEARCH_TERMS.get(ticker, ticker.replace(".JK", ""))
    try:
        pt = TrendReq(hl="id-ID", tz=420, timeout=(10, 25))
        pt.build_payload([search_term], cat=0, timeframe=f"{start} {end}", geo="ID")
        df = pt.interest_over_time()
        if df.empty:
            return pd.Series(name="google_trend", dtype=float)

        series = df[search_term]
        # Shift by 1 week (weekly data reported retroactively)
        series.index = series.index + pd.DateOffset(days=7)
        series = series.resample("D").last().ffill().bfill()
        series.name = "google_trend"
        time.sleep(2)
        return series
    except Exception:
        return pd.Series(name="google_trend", dtype=float)


def fetch_news_sentiment(ticker: str) -> float:
    """Get VADER sentiment score for a ticker from news headlines.

    Uses NewsAPI (if NEWSAPI_KEY is set) + VADER sentiment analysis.
    Falls back to 0.0 if unavailable or on error.

    A float cannot say *why* it is 0.0, so this function cannot report a
    failure: the module behind it returns ``0.0`` for a dead endpoint, a 429, a
    missing key and an absent VADER, exactly as it does for genuinely neutral
    news.  Callers that must tell those apart want
    :func:`fetch_news_sentiment_status`.

    Returns:
        Float between -1 (very negative) and +1 (very positive).
    """
    try:
        from news_sentiment import get_sentiment_feature
        return get_sentiment_feature(ticker)
    except Exception as e:
        logger.debug("News sentiment unavailable for %s: %s", ticker, e)
        return 0.0


def fetch_news_sentiment_status(ticker: str) -> tuple[float, bool, Optional[str]]:
    """Get the sentiment score *and* whether it is a measurement.

    ``fetch_news_sentiment`` returns a bare float, and every production failure
    mode of the news path — no ``NEWSAPI_KEY``, a dead or 401 endpoint, a 429, no
    ``vaderSentiment`` — returns the same ``0.0`` as a quiet news day.  That is
    how a dead news API came to be presented as neutral news (F8).  This
    companion is the seam the serving overlay reads instead: it reports the
    failure as data rather than expecting an exception that the real code never
    raises.

    Returns:
        ``(score, available, reason)``.  ``score`` is a float in [-1, 1];
        ``available`` is ``False`` when it is the fallback rather than a
        measurement, and ``reason`` then names the failure.  A plain 3-tuple
        rather than a class, so the module-import failure below can report the
        same shape without importing the class it could not import.
    """
    try:
        from news_sentiment import get_sentiment_with_status
    except Exception as e:
        reason = f"news_sentiment module is not importable ({type(e).__name__}: {e})"
        return 0.0, False, reason
    return get_sentiment_with_status(ticker)


def fetch_idx_fundamentals(ticker: str) -> dict:
    try:
        import importlib
        if importlib.util.find_spec("idx_fundamental_analysis"):
            logger.debug("idx-fundamental-analysis found but API may vary")
    except Exception:
        pass

    fields = ["idn_der", "idn_roe", "idn_roa", "idn_current_ratio",
              "idn_gpm", "idn_npm", "idn_eps_growth", "idn_bvps",
              "idn_per", "idn_pbv"]
    return {f: np.nan for f in fields}
