"""
Feature engineering pipeline: load_data, inject_macro_features, compute_target,
build_features, engineer_features_for_ticker, build_feature_set.

Orchestrates all feature modules to build the complete feature set.

``build_features`` is the single, I/O-free feature builder: every external value
(fundamentals, macro frames, sentiment, volume profile, IDX fundamentals) is
passed in as an argument.  ``engineer_features_for_ticker`` is the training
entry point and is now a thin wrapper that fetches those values and delegates.
"""
import os
import warnings
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


class _NotSupplied:
    """Sentinel type for a keyword argument the caller never passed at all.

    ``None`` cannot carry this meaning: it is a *value*, and for the three
    family arguments of :func:`build_features` it means "this family is
    deliberately absent" — the serving path passes ``None`` on purpose, because
    it cannot obtain volume-profile, IDX-fundamental or Google-Trends values
    without per-request network I/O.  The sentinel instead means "the caller
    omitted the argument", which is the forgetful-caller case the omission
    warning exists for.

    One value cannot express both.  Conflating them made serving log two
    warnings on every healthy request, which trains operators to ignore
    warnings and so defeats the visibility the warning was added to create.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Readable in a log line or a debugger, unlike a bare ``object()``."""
        return "<not supplied>"


#: Default for the family keyword arguments of :func:`build_features`; see
#: :class:`_NotSupplied`.  Omitting an argument that defaults to this and whose
#: ``FEATURE_FLAGS`` entry is on is a degradation and is logged at ``warning``.
_NOT_SUPPLIED = _NotSupplied()


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


def build_features(df, ticker, *, fundamentals, usdidr, fred_macro, bi_rate,
                   ihsg=None, all_ticker_data=None, news_sentiment=0.0,
                   google_trend=_NOT_SUPPLIED, vp_features=_NOT_SUPPLIED,
                   idx_fundamentals=_NOT_SUPPLIED,
                   mode="full", with_target=False) -> pd.DataFrame:
    """Build the feature frame for one ticker from already-fetched inputs.

    Pure: this function performs no I/O.  It never fetches, reads or writes
    anything — every value it needs arrives as an argument, which is what lets
    the training path and the serving path share one implementation.

    The three family arguments ``google_trend``, ``vp_features`` and
    ``idx_fundamentals`` default to the module sentinel ``_NOT_SUPPLIED`` rather
    than to ``None``, so that two different absences stay distinguishable:

    * **the argument was omitted** — the caller never supplied this family.  The
      family's columns are omitted and the omission is logged at ``warning``
      when the family's ``FEATURE_FLAGS`` entry is on and ``mode != 'ci'``,
      because a family that quietly vanishes is a degradation; and
    * **the argument is an explicit ``None``** — the caller declares the absence
      deliberate (serving cannot obtain these families without per-request
      network I/O).  The columns are omitted and nothing is logged: an
      intentional omission is a design decision, not a fault.

    Neither case ever fabricates a column.

    Args:
        df: Raw multi-ticker OHLCV DataFrame (yfinance MultiIndex format)
        ticker: Ticker symbol
        fundamentals: Dict from fetch_fundamentals()
        usdidr: USD/IDR DataFrame
        fred_macro: FRED macro DataFrame
        bi_rate: BI rate Series
        ihsg: IHSG index DataFrame (optional, for market context)
        all_ticker_data: Dict of {ticker: DataFrame} for cross-stock features
            (optional) — accepted for call-site compatibility only.  Cross-stock
            features are applied by build_feature_set once every ticker has been
            engineered, because the sector/breadth columns need the full frame.
        news_sentiment: Sentiment score to store (used when the flag is on and
            mode is not 'ci'; otherwise the column is 0.0).  This keeps its
            ``0.0`` default deliberately: it is the value the column always
            carries, and the visibility of a *failed* sentiment fetch belongs to
            the serving overlay (``PredictResponse.degraded``), not here.
        google_trend: Trend value to materialise as a ``google_trend`` column.
            ``None`` means "deliberately no value": the column is not
            materialised and nothing is logged, which is what serving passes —
            a constant would fabricate a column the model consumes.  Omitting the
            argument means the caller did not supply the family: the column is
            still not materialised (never a ``0.0`` constant) and the omission is
            logged at ``warning``.
        vp_features: Dict of volume profile features from
            compute_volume_profile_features() (optional; applied only when the
            flag is on and mode is not 'ci').  ``None`` means "deliberately not
            supplied": the columns are omitted and nothing is logged.  Omitting
            the argument omits the same columns and logs the omission at
            ``warning`` (R3).
        idx_fundamentals: Dict of IDX fundamental fields (optional).  ``None``
            and omission behave exactly as for ``vp_features``.
        mode: 'full', 'incremental', or 'ci' (CI mode skips intraday)
        with_target: When True, compute the binary target and set the ``ticker``
            column.  Off by default because compute_target() drops the final row
            (it needs a next-day price), so the inference path must not call it.

    Returns:
        Feature-rich DataFrame or None if insufficient data.
    """
    ticker_df = df[ticker].copy()
    ticker_df = dropna(ticker_df)

    if len(ticker_df) < 50:
        return None

    # ── Shared: TA indicators ────────────────────────────────────────────
    if FEATURE_FLAGS.get("ta_indicators", True):
        with warnings.catch_warnings():
            # ta 0.11.0's PSARIndicator assigns by integer position instead of
            # .iloc (ta/trend.py:1030, every neighbouring line uses .iloc), and
            # pandas >= 2.2 deprecates positional __setitem__ on a label index.
            # The value still lands in the right place, so this is upstream
            # deprecation noise and not a feature defect — but the suite runs
            # with filterwarnings=["error"], which would otherwise abort every
            # test that walks the TA path.  Scoped to this one message and this
            # one call: every other warning still raises.
            warnings.filterwarnings(
                "ignore",
                message="Series.__setitem__ treating keys as positions is deprecated",
                category=FutureWarning,
            )
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
        ticker_df["news_sentiment"] = news_sentiment
    else:
        ticker_df["news_sentiment"] = 0.0

    # ── Volume Profile (skip in CI mode) ─────────────────────────────────
    if FEATURE_FLAGS.get("volume_profile", True) and mode != "ci":
        if vp_features is _NOT_SUPPLIED:
            # The caller never passed this family, so the 19 vp_* columns are
            # missing from a run whose flag says they belong there.  That is a
            # degradation, not a decision: named at warning, and still omitted —
            # fabricating them is the defect this task exists to remove.  An
            # explicit None takes the branch below and is silent, because a
            # caller that says "deliberately absent" has made a decision.
            logger.warning(
                "volume_profile family not supplied for %s: the vp_features "
                "argument was omitted while FEATURE_FLAGS['volume_profile'] is "
                "on and mode=%r; the 19 vp_* columns stay absent (pass None to "
                "declare the absence deliberate)", ticker, mode,
            )
        elif vp_features is not None:
            for col, val in vp_features.items():
                ticker_df[col] = val

    # ── Market Context ──────────────────────────────────────────────────
    if FEATURE_FLAGS.get("market_context", True) and ihsg is not None:
        ticker_df = compute_market_context(ticker_df, ihsg, ticker)

    # ── IDX Fundamentals ────────────────────────────────────────────────
    if FEATURE_FLAGS.get("idx_fundamentals", True):
        if idx_fundamentals is _NOT_SUPPLIED:
            # Same shape as volume_profile above: the flag is on and the argument
            # was never passed, so the ten idn_* columns are omitted invisibly.
            # Named at warning, omitted as before; an explicit None is silent.
            logger.warning(
                "idx_fundamentals family not supplied for %s: the "
                "idx_fundamentals argument was omitted while "
                "FEATURE_FLAGS['idx_fundamentals'] is on; the idn_* columns stay "
                "absent (pass None to declare the absence deliberate)", ticker,
            )
        elif idx_fundamentals is not None:
            for col, val in idx_fundamentals.items():
                ticker_df[col] = val

    # ── Google Trends ───────────────────────────────────────────────────
    # Materialised only when the caller supplies a value.  ``None`` is the
    # deliberate absence (serving cannot reproduce the fetch, and any constant
    # would fabricate a column the model consumes), while the sentinel is a
    # caller that forgot the family — named, and still not fabricated.  Before
    # the sentinel the default was ``0.0``, so "left at its default" was
    # indistinguishable from "the caller passed 0.0": a safe default
    # indistinguishable from a real value.
    if FEATURE_FLAGS.get("google_trends", True) and mode != "ci":
        if google_trend is _NOT_SUPPLIED:
            logger.warning(
                "google_trend not supplied for %s: the google_trend argument was "
                "omitted while FEATURE_FLAGS['google_trends'] is on and mode=%r; "
                "the column stays absent rather than defaulting to 0.0",
                ticker, mode,
            )
        elif google_trend is not None:
            ticker_df["google_trend"] = google_trend

    # ── Shared: Target ─────────────────────────────────────────────────
    if with_target:
        ticker_df = compute_target(ticker_df)
        ticker_df["ticker"] = ticker

    return ticker_df


def engineer_features_for_ticker(df, ticker, fundamentals, usdidr, fred_macro,
                                 bi_rate, ihsg=None, all_ticker_data=None,
                                 mode="full") -> pd.DataFrame:
    """Engineer features for a single ticker (training entry point).

    Fetches the per-ticker external inputs and delegates to build_features(),
    which owns all the feature logic.  Output is unchanged: the same columns,
    values and row count as before the extraction.

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
    # ── Short-history gate ───────────────────────────────────────────────
    # Same <50-usable-rows gate as build_features, checked here *first* so a
    # short-history ticker returns before any per-ticker fetch.  This is the
    # pre-refactor order: the extraction had moved the gate behind the
    # sentiment/volume-profile/fundamentals calls, which cost needless network
    # requests under CACHE_CONFIG["rate_limit_delay_seconds"].  build_features
    # keeps its own identical guard for callers that hand it data directly.
    if len(dropna(df[ticker].copy())) < 50:
        return None

    # ── News Sentiment ───────────────────────────────────────────────────
    news_sentiment = 0.0
    if FEATURE_FLAGS.get("news_sentiment", True) and mode != "ci":
        news_sentiment = fetch_news_sentiment(ticker)

    # ── Volume Profile (skip in CI mode) ─────────────────────────────────
    vp_features = {}
    if FEATURE_FLAGS.get("volume_profile", True) and mode != "ci":
        try:
            vp_features = compute_volume_profile_features(ticker)
        except Exception as e:
            logger.warning("Volume profile features unavailable for %s: %s", ticker, e)
            vp_features = {}

    # ── IDX Fundamentals ────────────────────────────────────────────────
    idx_fundamentals = {}
    if FEATURE_FLAGS.get("idx_fundamentals", True):
        idx_fundamentals = fetch_idx_fundamentals(ticker)

    # Every family is passed *explicitly* — including ``google_trend=None``,
    # which is a deliberate absence (build_feature_set fetches and joins its own
    # ``google_trend`` column, and two frames both carrying it make
    # ``DataFrame.join`` raise).  Nothing here is left to the sentinel default,
    # so the training path never triggers the omission warning: an omission it
    # warns about is a real caller mistake, not this wrapper.
    return build_features(
        df, ticker,
        fundamentals=fundamentals, usdidr=usdidr, fred_macro=fred_macro,
        bi_rate=bi_rate, ihsg=ihsg, all_ticker_data=all_ticker_data,
        news_sentiment=news_sentiment, google_trend=None,
        vp_features=vp_features, idx_fundamentals=idx_fundamentals,
        mode=mode, with_target=True,
    )


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
