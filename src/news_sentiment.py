"""
News Sentiment Feature — NewsAPI + VADER sentiment analysis.

Fetches daily news headlines for each ticker and computes sentiment scores.
Adds as a feature during feature engineering.

Usage:
    python src/news_sentiment.py BBCA.JK          # Single ticker
    python src/news_sentiment.py --all             # All 45 tickers
    python src/news_sentiment.py --test            # Quick test

Requires:
    pip install vaderSentiment
    pip install nltk
    NEWSAPI_KEY environment variable (get free key at https://newsapi.org)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import json
from typing import NamedTuple, Optional

from config import TICKERS, CACHE_CONFIG, get_logger

logger = get_logger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    # ``logger.warning``, not ``print``: a bare print of this message writes a
    # non-ASCII character to stdout, and on a stream whose encoding cannot carry
    # it (a pipe or a redirect on a cp1252 host) it raises ``UnicodeEncodeError``
    # *during this module's import*.  ``fetch_news_sentiment`` catches every
    # exception, so the whole news path then failed as a silent 0.0 — a
    # dependency notice turning into the very defect this task removes.  Logging
    # is also what the sibling fetcher does for its missing optional dependency
    # (``features/fetchers.py``: fredapi, pytrends), and it is visible to
    # operators instead of only to whatever stdout happens to be attached.
    logger.warning("vaderSentiment not installed — run: pip install vaderSentiment")


class SentimentResult(NamedTuple):
    """A sentiment score together with whether it is a *measurement*.

    ``0.0`` has two entirely different meanings — "the headlines VADER read were
    neutral" and "there was no news product at all" — and this module reported
    both as the same float.  The serving overlay built its ``degraded`` list on
    that ambiguity, so a missing ``NEWSAPI_KEY``, a dead NewsAPI endpoint, a 429
    and an absent ``vaderSentiment`` all arrived as a clean neutral reading, with
    no warning (F8).

    ``score`` is always a float, so a caller that only wants the feature value is
    unaffected.  ``available`` is ``False`` exactly when ``score`` is the
    fallback rather than a computed value, and ``reason`` then names the failure
    in operator-readable terms.

    Ordered ``(score, available, reason)`` so it can be unpacked positionally by
    a caller that does not want to import this class.
    """

    score: float
    available: bool
    reason: Optional[str] = None


#: Reason strings for the failure modes that make a score unavailable.  Each is
#: the reason an operator reads on the overlay's ``warning`` line, so they are
#: distinct and specific rather than one shared "sentiment failed".
NO_NEWSAPI_KEY = "NEWSAPI_KEY is not set"
NO_VADER = "vaderSentiment is not installed"
NEWSAPI_RATE_LIMITED = "NewsAPI rate limit reached (HTTP 429)"

# ── Configuration ─────────────────────────────────────────────────────────────
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
NEWSAPI_URL = "https://newsapi.org/v2/everything"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sentiment")
CACHE_TTL_HOURS = 6

# Ticker → search keywords
TICKER_KEYWORDS = {
    "BBCA.JK": "Bank BCA Indonesia",
    "BBRI.JK": "Bank BRI Indonesia",
    "BMRI.JK": "Bank Mandiri Indonesia",
    "BBNI.JK": "Bank BNI Indonesia",
    "BBTN.JK": "Bank BTN Indonesia",
    "BRIS.JK": "Bank Syariah Indonesia",
    "ARTO.JK": "Bank Jago Indonesia",
    "TLKM.JK": "Telkom Indonesia",
    "ISAT.JK": "Indosat Indonesia",
    "EXCL.JK": "XL Axiata Indonesia",
    "TOWR.JK": "Sarana Menara Tower Indonesia",
    "ASII.JK": "Astra International Indonesia",
    "UNTR.JK": "United Tractors Indonesia",
    "ADRO.JK": "Adaro Energy Indonesia",
    "ANTM.JK": "Aneka Tambang Indonesia",
    "PTBA.JK": "Bukit Asam Indonesia",
    "ITMG.JK": "Indo Tambangraya Megah",
    "MEDC.JK": "Medco Energi Indonesia",
    "PGEO.JK": "Pertamina Geothermal Indonesia",
    "INCO.JK": "Vale Indonesia",
    "MDKA.JK": "Merdeka Copper Gold",
    "MBMA.JK": "Merdeka Battery Materials",
    "UNVR.JK": "Unilever Indonesia",
    "ICBP.JK": "Indofood CBP Indonesia",
    "INDF.JK": "Indofood Indonesia",
    "KLBF.JK": "Kalbe Farma Indonesia",
    "SIDO.JK": "Industri Jamu Sido Muncul",
    "GOTO.JK": "GoTo Gojek Tokopedia Indonesia",
    "BREN.JK": "Barito Renewables Energy",
    "AMMN.JK": "Amman Mineral Internasional",
    "CPIN.JK": "Charoen Pokphand Indonesia",
    "JPFA.JK": "JAPFA Comfeed Indonesia",
    "SMGR.JK": "Semen Indonesia",
    "CTRA.JK": "Ciputra Development Indonesia",
    "SMRA.JK": "Summarecon Agung Indonesia",
    "JSMR.JK": "Jasa Marga Indonesia",
    "PGAS.JK": "Perusahaan Gas Negara",
    "AKRA.JK": "AKR Corporindo Indonesia",
    "INKP.JK": "Indah Kiat Pulp Paper",
    "BRPT.JK": "Barito Pacific Indonesia",
    "MAPA.JK": "MAP Aktif Adiperkasa",
    "MAPI.JK": "Mitra Adiperkasa Indonesia",
    "AADI.JK": "Adaro Andalan Indonesia",
    "ADMR.JK": "Adaro Minerals Indonesia",
    "AMRT.JK": "Sumber Alfaria Trijaya",
}

# ── Sentiment Analyzer ────────────────────────────────────────────────────────

_sentiment_analyzer = None

def get_sentiment_analyzer():
    """Lazy-load VADER sentiment analyzer."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None and VADER_AVAILABLE:
        _sentiment_analyzer = SentimentIntensityAnalyzer()
    return _sentiment_analyzer

def analyze_sentiment(texts: list) -> float:
    """Compute average compound sentiment score for a list of texts.

    Returns a score between -1 (very negative) and +1 (very positive).
    Returns 0 if no texts or analyzer unavailable.
    """
    analyzer = get_sentiment_analyzer()
    if analyzer is None or not texts:
        return 0.0

    scores = []
    for text in texts:
        try:
            vs = analyzer.polarity_scores(text)
            scores.append(vs["compound"])
        except Exception:
            continue

    return float(np.mean(scores)) if scores else 0.0


# ── News Fetching ─────────────────────────────────────────────────────────────

def _fetch_news_with_reason(ticker: str, days_back: int = 3) -> "tuple[list, Optional[str]]":
    """Fetch recent news headlines, and say why there are none when there are none.

    ``fetch_news`` is the headline-only view of this function.  The reason it
    returns is what lets a caller tell "NewsAPI answered and had nothing" from
    "NewsAPI never answered" — the distinction every failure mode used to be
    collapsed into, because all of them ended as an empty list and therefore as
    a ``0.0`` score.

    Args:
        ticker: Stock ticker (e.g. 'BBCA.JK')
        days_back: How many days of news to fetch

    Returns:
        ``(headlines, reason)``.  ``reason`` is ``None`` only when the endpoint
        answered and produced at least one headline; otherwise it names the
        failure (no key, rate limit, endpoint error, or no articles).
    """
    if not NEWSAPI_KEY:
        logger.debug("No NEWSAPI_KEY set — skipping news fetch for %s", ticker)
        return [], NO_NEWSAPI_KEY

    search_term = TICKER_KEYWORDS.get(ticker, ticker.replace(".JK", ""))
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # Try multiple search strategies
    strategies = [
        # Strategy 1: English keywords
        {"q": search_term, "language": "en"},
        # Strategy 2: Ticker name, any language
        {"q": ticker.replace(".JK", "")},
        # Strategy 3: Broader company name, English
        {"q": search_term.split()[0] if " " in search_term else search_term, "language": "en"},
    ]

    endpoint_error: Optional[str] = None
    answered_without_articles = False
    for strategy in strategies:
        try:
            r = requests.get(NEWSAPI_URL, params={
                "q": strategy["q"],
                "from": from_date,
                "sortBy": "publishedAt",
                "apiKey": NEWSAPI_KEY,
                "pageSize": 10,
                **{k: v for k, v in strategy.items() if k != "q"},
            }, timeout=10)
        except Exception as e:
            endpoint_error = f"NewsAPI request failed ({type(e).__name__}: {e})"
            continue

        if r.status_code == 429:
            logger.warning("NewsAPI rate limited for %s", ticker)
            return [], NEWSAPI_RATE_LIMITED
        if r.status_code != 200:
            # 401/403 (bad or revoked key) and 5xx land here, and every one of
            # them used to be indistinguishable from a quiet news day.
            endpoint_error = f"NewsAPI returned HTTP {r.status_code}"
            continue

        try:
            data = r.json()
        except Exception as e:
            endpoint_error = f"NewsAPI returned unparseable JSON ({type(e).__name__}: {e})"
            continue

        articles = data.get("articles", [])
        headlines = [a.get("title", "") for a in articles if a.get("title")]
        if headlines:
            logger.debug("Found %d articles for %s (strategy: %s)",
                        len(headlines), ticker, strategy)
            return headlines, None
        answered_without_articles = True

    logger.debug("No articles found for %s with any strategy", ticker)

    if endpoint_error is not None:
        return [], f"{endpoint_error} for {ticker} (every search strategy failed)"
    if answered_without_articles:
        # A 200 with no usable titles is not a neutral reading: VADER was never
        # given a headline to score, so the 0.0 that follows is an absent
        # measurement.  Naming it is what keeps the fallback from passing as one.
        return [], (f"NewsAPI returned no articles for {ticker} "
                    f"(all {len(strategies)} search strategies, HTTP 200)")
    return [], f"no search strategy produced a result for {ticker}"


def fetch_news(ticker: str, days_back: int = 3) -> list:
    """Fetch recent news headlines for a ticker via NewsAPI.

    Tries English keywords first, then falls back to ticker name search,
    then tries without language filter for Indonesian-language coverage.

    The headline-only view of :func:`_fetch_news_with_reason`: an empty list
    here carries no explanation, so a caller that must tell an unavailable
    source from a quiet one wants that function instead.

    Args:
        ticker: Stock ticker (e.g. 'BBCA.JK')
        days_back: How many days of news to fetch

    Returns:
        List of headline text strings, or empty list on failure.
    """
    headlines, _reason = _fetch_news_with_reason(ticker, days_back)
    return headlines


# ── Caching ───────────────────────────────────────────────────────────────────

def load_cache(ticker: str) -> dict:
    """Load cached sentiment for a ticker, or empty dict if stale/missing."""
    path = os.path.join(CACHE_DIR, f"{ticker}.json")
    if not os.path.exists(path):
        return {}

    try:
        with open(path) as f:
            data = json.load(f)
        cache_time = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        if age_hours < CACHE_TTL_HOURS:
            return data
    except Exception:
        pass

    return {}


def save_cache(ticker: str, data: dict):
    """Save sentiment data to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    data["cached_at"] = datetime.now().isoformat()
    path = os.path.join(CACHE_DIR, f"{ticker}.json")
    with open(path, "w") as f:
        json.dump(data, f)


# ── Main API ──────────────────────────────────────────────────────────────────

def get_ticker_sentiment(ticker: str, force_refresh: bool = False) -> dict:
    """Get sentiment score for a ticker.

    Checks cache first, then fetches fresh news and computes sentiment.

    Returns:
        dict with keys: ticker, sentiment_score, n_articles, n_headlines,
        available, reason.  ``available`` is ``False`` when ``sentiment_score``
        is the fallback ``0.0`` rather than a measurement, and ``reason`` then
        names why.  Both are written to the cache alongside the score, so an
        unavailable reading cannot come back out of the cache looking like a
        neutral one.
    """
    if not force_refresh:
        cached = load_cache(ticker)
        # An entry written before this module recorded ``available`` has unknown
        # provenance: it may be a *failed* fetch cached as a neutral score, which
        # is how a dead key used to be persisted (``data/sentiment/`` is full of
        # ``{"sentiment_score": 0.0, "n_articles": 0}`` from exactly that).  Such
        # an entry is refetched rather than believed — the refetch overwrites it
        # with a marked one, so this costs one call per ticker after an upgrade.
        if cached and "sentiment_score" in cached and "available" in cached:
            return cached

    headlines, reason = _fetch_news_with_reason(ticker)

    if reason is None and get_sentiment_analyzer() is None:
        # The endpoint answered, so the blocker is local: VADER is absent.
        # Deliberately checked *after* the fetch — checking it first would report
        # a missing dependency for a request that never got that far, and a 429
        # has to stay distinguishable from an absent analyser.
        reason = NO_VADER

    score = 0.0 if reason is not None else analyze_sentiment(headlines)
    n_headlines = len(headlines)

    result = {
        "ticker": ticker,
        "sentiment_score": score,
        "n_articles": n_headlines,
        "n_headlines": n_headlines,
        "available": reason is None,
        "reason": reason,
        "fetched_at": datetime.now().isoformat(),
    }

    save_cache(ticker, result)
    return result


def get_all_sentiments(tickers: list = None, force_refresh: bool = False,
                       rate_limit_delay: float = 1.5) -> dict:
    """Get sentiment for all tickers.

    Args:
        tickers: List of tickers (default: all TICKERS)
        force_refresh: Bypass cache
        rate_limit_delay: Seconds between API calls (NewsAPI free tier: 100/day)

    Returns:
        dict of {ticker: sentiment_data}
    """
    if tickers is None:
        tickers = TICKERS

    results = {}
    for i, ticker in enumerate(tickers):
        logger.info("[%d/%d] Fetching sentiment for %s...", i + 1, len(tickers), ticker)
        result = get_ticker_sentiment(ticker, force_refresh=force_refresh)
        results[ticker] = result
        logger.info("  -> %s: score=%.3f (articles=%d)",
                    ticker, result["sentiment_score"], result["n_articles"])

        if i < len(tickers) - 1:
            time.sleep(rate_limit_delay)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_sentiment_with_status(ticker: str) -> SentimentResult:
    """Get the sentiment score *and* whether it is a measurement.

    This is the seam the serving overlay reads.  A caller that takes the ``0.0``
    without consulting ``available`` is reporting a failure as a successful
    neutral reading, which is F8.

    Never raises — an unavailable source is reported through
    ``available``/``reason`` rather than through an exception, because none of
    the four production failure modes (no ``NEWSAPI_KEY``, a dead endpoint, a
    429, no ``vaderSentiment``) can be detected from the returned value alone,
    and an exception is a shape the real code produces only when something
    outside these four breaks.
    """
    try:
        data = get_ticker_sentiment(ticker)
    except Exception as e:
        return SentimentResult(
            0.0, False, f"sentiment lookup failed ({type(e).__name__}: {e})"
        )

    score = float(data.get("sentiment_score", 0.0))
    if data.get("available", True):
        return SentimentResult(score, True, None)
    return SentimentResult(score, False, data.get("reason") or "sentiment unavailable")


def get_sentiment_feature(ticker: str) -> float:
    """Get sentiment score for use as a feature.

    Returns a float between -1 and 1. Never raises — defaults to 0 on failure.
    The value-only view of :func:`get_sentiment_with_status`, which is what a
    caller needs when "unavailable" must not be presented as "neutral".
    """
    return get_sentiment_with_status(ticker).score


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="News sentiment analysis for LQ45 stocks")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols or --all")
    parser.add_argument("--all", action="store_true", help="All 45 tickers")
    parser.add_argument("--test", action="store_true", help="Quick test on BBCA.JK")
    parser.add_argument("--force", action="store_true", help="Force refresh (bypass cache)")
    args = parser.parse_args()

    if args.test:
        tickers = ["BBCA.JK"]
    elif args.all:
        tickers = TICKERS
    elif args.tickers:
        tickers = args.tickers
    else:
        parser.print_help()
        sys.exit(1)

    if not NEWSAPI_KEY:
        print("=" * 60)
        print("NewsAPI key not found! Set NEWSAPI_KEY environment variable.")
        print("Get a free key at: https://newsapi.org")
        print("")
        print("Testing with sample headlines instead (no API key)...")
        print("=" * 60)

    results = get_all_sentiments(tickers, force_refresh=args.force)

    print("\n=== Sentiment Results ===")
    print(f"{'Ticker':<12} {'Score':>8} {'Articles':>10}")
    print("-" * 32)
    for t, d in sorted(results.items(), key=lambda x: x[1]["sentiment_score"], reverse=True):
        print(f"{t:<12} {d['sentiment_score']:>8.3f} {d['n_articles']:>10}")

    if not NEWSAPI_KEY:
        print("\n⚠️  No NEWSAPI_KEY set. Showing cached/sample data.")
        print("   To use real news: set NEWSAPI_KEY env var and re-run.")
