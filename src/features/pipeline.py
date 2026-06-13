"""
Feature engineering pipeline: load_data, inject_macro_features, compute_target,
engineer_features_for_ticker, build_feature_set.

Orchestrates all feature modules to build the complete feature set.
"""
import os
import pandas as pd
import numpy as np
from ta.utils import dropna

# sys.path trick
import sys as _sys
_curr = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_curr)
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)

from config import (
    FEATURE_FLAGS, DATA_PROCESSED_PATH, get_logger,
)

logger = get_logger(__name__)

# ── Relative imports from sibling modules ─────────────────────────────────────
from .fetchers import (
    fetch_fundamentals, fetch_usdidr, fetch_fred_macro, fetch_bi_rate,
    fetch_google_trends, fetch_news_sentiment, fetch_idx_fundamentals,
)
from .indicators import compute_ta_features, compute_custom_features
from .enhanced_mas import compute_enhanced_mas
from .ict import compute_ict_features
from .volume_profile import compute_volume_profile_features
from .market import load_ihsg_data, compute_market_context, compute_cross_stock_features


def load_data(path: str) -> pd.DataFrame:
    """Load raw OHLCV data (yfinance MultiIndex format)."""
    return pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)


def inject_macro_features(df: pd.DataFrame, fundamentals: dict,
                          usdidr: pd.DataFrame, fred_macro: pd.DataFrame,
                          bi_rate: pd.Series) -> pd.DataFrame:
    """Inject external macro/fundamental features into the feature DataFrame.

    All external features are prepopulated as NaN and filled from the provided
    data. Missing data gracefully falls back to NaN/0.

    Args:
        df: Feature DataFrame (with datetime index)
        fundamentals: Dict of fundamental metrics from fetch_fundamentals()
        usdidr: USD/IDR rate DataFrame from fetch_usdidr()
        fred_macro: FRED macro DataFrame from fetch_fred_macro()
        bi_rate: BI rate Series from fetch_bi_rate()

    Returns:
        DataFrame with macro columns appended.
    """
    for col, val in (fundamentals or {}).items():
        df[col] = val

    if usdidr is not None and not usdidr.empty and "usdidr_rate" in usdidr.columns:
        df = df.join(usdidr, how="left")
        df["usdidr_rate"] = df["usdidr_rate"].ffill().bfill()
        df["usdidr_return"] = df["usdidr_return"].ffill().bfill().fillna(0)
    else:
        df["usdidr_rate"] = np.nan
        df["usdidr_return"] = 0.0

    if fred_macro is not None and not fred_macro.empty:
        df = df.join(fred_macro, how="left")
        for col in fred_macro.columns:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

    if bi_rate is not None and not bi_rate.empty:
        df = df.join(bi_rate, how="left")
        if "bi_rate_official" in df.columns:
            df["bi_rate_official"] = df["bi_rate_official"].ffill().bfill()
    else:
        df["bi_rate_official"] = np.nan

    return df


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute binary target: 1 if next day's Close > today's Close, else 0."""
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1]  # Drop last row (no next-day price)
    return df


def engineer_features_for_ticker(df, ticker, fundamentals, usdidr, fred_macro,
                                 bi_rate, ihsg=None, all_ticker_data=None,
                                 mode="full") -> pd.DataFrame:
    """Engineer features for a single ticker.

    Args:
        df: Raw multi-ticker OHLCV DataFrame (yfinance MultiIndex format)
        ticker: Ticker symbol
        fundamentals: Dict from fetch_fundamentals()
        usdidr: USD/IDR DataFrame
        fred_macro: FRED macro DataFrame
        bi_rate: BI rate Series
        ihsg: IHSG index DataFrame (optional, for market context)
        all_ticker_data: Dict of {ticker: DataFrame} for cross-stock features (optional)
        mode: 'full', 'incremental', or 'ci' (CI mode skips intraday)

    Returns:
        Feature-rich DataFrame or None if insufficient data.
    """
    ticker_df = df[ticker].copy()
    ticker_df = dropna(ticker_df)

    if len(ticker_df) < 50:
        return None

    # ── Shared: TA indicators ────────────────────────────────────────────
    if FEATURE_FLAGS.get("ta_indicators", True):
        ticker_df = compute_ta_features(ticker_df)

    # ── Shared: Custom features ──────────────────────────────────────────
    ticker_df = compute_custom_features(ticker_df)

    # ── Enhanced MAs ────────────────────────────────────────────────────
    if FEATURE_FLAGS.get("enhanced_mas", True):
        ticker_df = compute_enhanced_mas(ticker_df)

    # ── ICT Features ─────────────────────────────────────────────────────
    if FEATURE_FLAGS.get("ict_suite", True):
        ticker_df = compute_ict_features(ticker_df)

    # ── Shared: Macro injection ──────────────────────────────────────────
    ticker_df = inject_macro_features(ticker_df, fundamentals, usdidr, fred_macro, bi_rate)

    # ── News Sentiment ───────────────────────────────────────────────────
    if FEATURE_FLAGS.get("news_sentiment", True) and mode != "ci":
        ticker_df["news_sentiment"] = fetch_news_sentiment(ticker)
    else:
        ticker_df["news_sentiment"] = 0.0

    # ── Volume Profile (skip in CI mode) ─────────────────────────────────
    if FEATURE_FLAGS.get("volume_profile", True) and mode != "ci":
        try:
            vp_features = compute_volume_profile_features(ticker)
            for col, val in vp_features.items():
                ticker_df[col] = val
        except Exception:
            pass  # Volume profile is optional

    # ── Market Context ──────────────────────────────────────────────────
    if FEATURE_FLAGS.get("market_context", True) and ihsg is not None:
        ticker_df = compute_market_context(ticker_df, ihsg, ticker)

    # ── IDX Fundamentals ────────────────────────────────────────────────
    if FEATURE_FLAGS.get("idx_fundamentals", True):
        idx_fund = fetch_idx_fundamentals(ticker)
        for col, val in idx_fund.items():
            ticker_df[col] = val

    # ── Shared: Target ─────────────────────────────────────────────────
    ticker_df = compute_target(ticker_df)
    ticker_df["ticker"] = ticker

    return ticker_df


def build_feature_set(raw_path: str, output_path: str, tickers: list,
                      mode: str = "full"):
    """Build the complete feature set for all tickers.

    Args:
        raw_path: Path to raw OHLCV CSV
        output_path: Path for output Parquet/CSV
        tickers: List of ticker symbols
        mode: 'full', 'incremental', or 'ci'
    """
    df = load_data(raw_path)
    start = str(df.index.min().date())
    end = str(df.index.max().date())
    logger.info("Date range: %s → %s", start, end)

    # ── Fetch external data once (shared across all tickers) ─────────────────
    logger.info("Fetching USD/IDR rate...")
    usdidr = fetch_usdidr(start, end)

    logger.info("Fetching FRED macro + commodity data...")
    fred_macro = fetch_fred_macro(start, end)

    logger.info("Fetching BI Rate...")
    bi_rate = fetch_bi_rate(start, end)

    # ── IHSG data for market context ─────────────────────────────────────────
    ihsg = None
    if FEATURE_FLAGS.get("market_context", True):
        logger.info("Loading IHSG data...")
        ihsg = load_ihsg_data()

    # ── Pre-extract per-ticker Close data for cross-stock features ───────────
    all_ticker_data = {}
    if FEATURE_FLAGS.get("market_context", True):
        for t in tickers:
            try:
                t_df = df[t].copy()
                all_ticker_data[t] = dropna(t_df)
            except KeyError:
                pass

    # ── Process each ticker ──────────────────────────────────────────────────
    all_features = []
    for i, ticker in enumerate(tickers):
        logger.info("[%d/%d] Processing %s...", i+1, len(tickers), ticker)
        try:
            fundamentals = fetch_fundamentals(ticker)
            featured = engineer_features_for_ticker(
                df, ticker, fundamentals, usdidr, fred_macro, bi_rate,
                ihsg=ihsg, all_ticker_data=all_ticker_data, mode=mode,
            )

            if featured is None or len(featured) < 50:
                logger.warning("Skipping %s — insufficient data", ticker)
                continue

            # Google Trends (per ticker, rate-limited)
            if FEATURE_FLAGS.get("google_trends", True) and mode != "ci":
                trend = fetch_google_trends(ticker, start, end)
                if not trend.empty:
                    trend.index = pd.to_datetime(trend.index)
                    featured = featured.join(trend, how="left")
                featured["google_trend"] = featured["google_trend"].ffill().bfill().fillna(0) \
                    if "google_trend" in featured.columns else 0
            else:
                featured["google_trend"] = 0

            # Cross-stock features (computed after per-ticker processing)
            if FEATURE_FLAGS.get("market_context", True) and all_ticker_data:
                featured = compute_cross_stock_features(
                    featured, ticker, all_ticker_data, tickers
                )

            all_features.append(featured)
        except Exception as e:
            logger.error("Skipping %s: %s", ticker, e)

    if not all_features:
        logger.error("No features generated for any ticker!")
        return

    combined = pd.concat(all_features)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".parquet"):
        combined.to_parquet(output_path)
        logger.info("Saved %d rows × %d cols to %s", len(combined), len(combined.columns), output_path)
    else:
        combined.to_csv(output_path)
        logger.info("Saved %d rows × %d cols to %s", len(combined), len(combined.columns), output_path)
