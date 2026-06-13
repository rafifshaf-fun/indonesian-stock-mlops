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

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️ vaderSentiment not installed — run: pip install vaderSentiment")

from config import TICKERS, CACHE_CONFIG, get_logger

logger = get_logger(__name__)

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

def fetch_news(ticker: str, days_back: int = 3) -> list:
    """Fetch recent news headlines for a ticker via NewsAPI.

    Tries English keywords first, then falls back to ticker name search,
    then tries without language filter for Indonesian-language coverage.

    Args:
        ticker: Stock ticker (e.g. 'BBCA.JK')
        days_back: How many days of news to fetch

    Returns:
        List of headline text strings, or empty list on failure.
    """
    if not NEWSAPI_KEY:
        logger.debug("No NEWSAPI_KEY set — skipping news fetch for %s", ticker)
        return []

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

            if r.status_code == 200:
                data = r.json()
                articles = data.get("articles", [])
                headlines = [a.get("title", "") for a in articles if a.get("title")]
                if headlines:
                    logger.debug("Found %d articles for %s (strategy: %s)",
                                len(headlines), ticker, strategy)
                    return headlines
            elif r.status_code == 429:
                logger.warning("NewsAPI rate limited for %s", ticker)
                return []
            # Other errors: try next strategy
        except Exception:
            continue

    logger.debug("No articles found for %s with any strategy", ticker)
    return []


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
        dict with keys: ticker, sentiment_score, n_articles, n_headlines
    """
    if not force_refresh:
        cached = load_cache(ticker)
        if cached and "sentiment_score" in cached:
            return cached

    headlines = fetch_news(ticker)
    score = analyze_sentiment(headlines)
    n_headlines = len(headlines)

    result = {
        "ticker": ticker,
        "sentiment_score": score,
        "n_articles": n_headlines,
        "n_headlines": n_headlines,
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

def get_sentiment_feature(ticker: str) -> float:
    """Get sentiment score for use as a feature.

    Returns a float between -1 and 1. Never raises — defaults to 0 on failure.
    """
    try:
        data = get_ticker_sentiment(ticker)
        return data.get("sentiment_score", 0.0)
    except Exception:
        return 0.0


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
