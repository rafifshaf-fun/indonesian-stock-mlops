"""
Data fetchers: fundamentals, USD/IDR, FRED macro, BI Rate, Google Trends.
"""
import json
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


#: ``fetch_fundamentals``'s eight keys, in the order the model contract has
#: always seen them.  Module-level so the value-only fetcher, the status seam and
#: every caller agree on the key set from one place: a key added to one of them
#: and not the other is a column that silently becomes NaN downstream.
FUNDAMENTAL_KEYS = (
    "pe_ratio", "pb_ratio", "dividend_yield", "market_cap",
    "debt_to_equity", "return_on_equity", "revenue_growth", "profit_margins",
)

#: yfinance field name per returned key.
_FUNDAMENTAL_FIELDS = {
    "pe_ratio": "trailingPE",
    "pb_ratio": "priceToBook",
    "dividend_yield": "dividendYield",
    "market_cap": "marketCap",
    "debt_to_equity": "debtToEquity",
    "return_on_equity": "returnOnEquity",
    "revenue_growth": "revenueGrowth",
    "profit_margins": "profitMargins",
}

#: How a response with no fundamental data is reported.  Named for the *fields*,
#: not for an empty body: the body is never empty.  Yahoo answers 200 with an
#: empty ``result`` list for a delisted or unknown symbol and raises for none of
#: these cases — which is what makes an ``except``-based detector structurally
#: insufficient — and the real ``Quote.info`` property then appends a
#: complementary key of its own, so the body has a key and no fundamentals.  The
#: ASCII part comes first and the non-ASCII dash last, as in the sentiment
#: overlay's warning: on a console whose encoding cannot represent the dash the
#: tail can render as replacement characters, and the reason is the part an
#: operator has to be able to read.
NO_FUNDAMENTAL_FIELDS = (
    "Yahoo returned no fundamental data for .info - none of the eight fields "
    "was published"
)


def _published(value):
    """One field's value, with yfinance's ``None`` read as an absent measurement.

    ``None`` cannot be passed through to the frame.  ``inject_macro_features``
    assigns each field with ``df[col] = val``, and in pandas 2.3.3 assigning
    ``None`` **creates an object-dtype column** rather than deleting one — and an
    object column is not numeric, so
    ``serve.build_inference_features``'s ``select_dtypes(include=[np.number])``
    (``src/serve.py:819``) drops it before the frame ever reaches
    ``align_to_schema``.  A schema-required fundamental would then be *absent*
    when the contract is enforced, and ``align_to_schema``
    (``src/features/schema.py:218``) raises ``FeatureSchemaError`` — so the
    request would answer a 500 where it used to answer a silent 200.  Both are
    wrong; ``NaN`` is the honest one, stays numeric, and is the value a field
    yfinance never published has always reached the fleet as.
    """
    return np.nan if value is None else value


def _fundamental_values(info: dict) -> dict:
    """The eight feature values from a Yahoo ``.info`` body.

    ``dividend_yield`` is passed through **as published**, with no truthiness
    fallback.  ``info.get("dividendYield", np.nan) or 0.0`` — what stood here —
    could not tell an unpublished yield from a real zero: ``None`` is falsy, so
    the fallback fired on it and turned "yfinance does not know" into a
    ``0.0`` in a column all 45 models consume.  XGBoost reads ``0.0`` (this
    ticker pays nothing) differently from ``NaN`` (nobody measured this), so the
    fabrication was not a lost value but an invented one.

    The reachable spelling is ``None``.  yfinance drops unknown fields entirely
    rather than publishing a null (``yfinance/scrapers/quote.py:650``), so a
    *missing key* already reached ``np.nan`` through the ``get`` default and was
    never affected; ``None`` is a value in the dict, which is why the fallback
    had to be removed rather than the default changed.  A published ``0.0`` is
    left as published: it is a measurement, and the one honest reading of it is
    the one Yahoo sent.

    The coercion itself — why ``None`` must become ``NaN`` rather than travel
    through to the frame — belongs to :func:`_published`.
    """
    return {
        key: _published(info.get(field))
        for key, field in _FUNDAMENTAL_FIELDS.items()
    }


def _fundamentals_failure_reason(exc: BaseException) -> str:
    """One distinct reason string per production failure mode of ``.info``.

    The modes are those the Yahoo client actually produces: an
    ``HTTPError`` carrying a status (429 is a rate limit; 401/403 is a dead or
    rejected credential and is *not* the same fault), a refused connection, a
    body that is not JSON — Yahoo serves an HTML error page under load — and
    anything else, named by its own type so an unrecognised failure still travels
    as text.  A rate limit reported as a dead key sends an operator to the wrong
    place, which is the reason Task 1.9 gave each sentiment failure its own
    string.
    """
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        if exc.response.status_code == 429:
            return "Yahoo Finance rate limit reached (HTTP 429)"
        return f"Yahoo Finance returned HTTP {exc.response.status_code}"
    if isinstance(exc, json.JSONDecodeError):
        return "Yahoo Finance returned a body that is not JSON"
    if isinstance(exc, requests.RequestException):
        return f"Yahoo Finance request failed ({type(exc).__name__}: {exc})"
    return f"Yahoo Finance .info fetch failed ({type(exc).__name__}: {exc})"


def fetch_fundamentals_status(ticker: str) -> tuple[dict, bool, Optional[str]]:
    """Get the eight fundamental values *and* whether they are a measurement.

    ``fetch_fundamentals`` returns a bare dict, and every production failure mode
    of the fundamentals path — a rate limit, a rejected key, a dead endpoint, an
    HTML error page, and an empty 200 — arrives as the same eight ``NaN`` values
    that a healthy ``.info`` produces for a field Yahoo does not publish.  Task
    2.3's retrain made five of those eight schema-required for all 45 models, so
    an outage became five constant-NaN inputs to a real prediction with nothing in
    the response saying so.  This is the seam the serving path reads instead: it
    reports the failure as data rather than expecting an exception that the real
    code does not always raise (the empty-200 mode raises nothing at all).

    Returns:
        ``(values, available, reason)``.  ``values`` always carries all eight
        keys; ``available`` is ``False`` when they are the fallback rather than a
        measurement, and ``reason`` then names the failure.  A plain 3-tuple
        rather than a class, mirroring
        :func:`fetch_news_sentiment_status`.

    The "no data" test is on the **fetched fields' values**:

    * ``not info`` — what stood here — is not a detector for this mode.  The real
      ``Quote.info`` property (``yfinance/scrapers/quote.py:502-508``) runs
      ``_fetch_complementary()`` immediately after ``_fetch_info()``, and that
      function writes a key of its own even when there is nothing to write:
      ``self.info[k] = None`` (``quote.py:755-760``).  An empty response therefore
      arrives as ``{'trailingPegRatio': None}``, whose truthiness is ``True`` —
      measured against the real ``Quote`` fed an in-memory ``YfData``, not
      inferred.  The predicate returned ``available=True`` with all eight fields
      ``NaN``, five of them schema-required: exactly the invisible outage this
      seam exists to remove, surviving on the one shape the brief names.
    * ``field in info`` — what replaced it — is a detector for key *presence*, and
      a body can carry all eight keys with ``None`` in each.  That is eight
      public fields nobody published a number for, and it read as
      ``available=True`` with all eight values ``NaN``.  This yfinance drops
      ``None`` values from the flattened body (``yfinance/scrapers/quote.py:650``),
      so the shape is defensive rather than reachable today — but "the key is
      there" and "a value is there" are different questions, and only the second
      is the caller's.
    * ``info.get(field) is not None`` asks it: *did any of the eight carry a
      value?*  An identity test rather than a truthiness one, because a published
      ``0.0`` is a measurement of zero — ``not info.get(field)`` would
      re-introduce the ``or 0.0`` fabrication one layer up.

    The distinction this preserves is the one the serving path depends on: a field
    that is absent **or** null means "Yahoo does not publish this", which is not
    "the source is down".  A body with a value for even one of the eight is a
    healthy fetch — the partially-populated case — so nothing here requires all
    eight.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        return {k: np.nan for k in FUNDAMENTAL_KEYS}, False, _fundamentals_failure_reason(e)

    if not any(info.get(field) is not None for field in _FUNDAMENTAL_FIELDS.values()):
        return {k: np.nan for k in FUNDAMENTAL_KEYS}, False, NO_FUNDAMENTAL_FIELDS

    return _fundamental_values(info), True, None


def fetch_fundamentals(ticker: str) -> dict:
    """Get fundamental metrics for a ticker.

    Never raises: an unavailable source degrades to ``{key: np.nan}`` for all
    eight keys, which is the safe default the project's convention requires.
    That default is indistinguishable from a healthy fetch of a ticker whose
    fields Yahoo does not publish, so a caller that must tell them apart wants
    :func:`fetch_fundamentals_status` — the value-only view of it, the way
    ``fetch_news_sentiment`` is the value-only view of its own status companion.
    """
    values, _available, _reason = fetch_fundamentals_status(ticker)
    return values


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
