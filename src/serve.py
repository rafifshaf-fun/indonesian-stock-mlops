import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from ta.utils import dropna
from datetime import date, timedelta
import asyncio
import traceback

from config import (
    TICKERS, MLFLOW_EXPERIMENT, FEATURE_FLAGS, get_logger,
)
from features import (
    fetch_fundamentals, fetch_usdidr, fetch_fred_macro, fetch_bi_rate,
    load_data, load_ihsg_data, compute_cross_stock_features,
)
# The one builder both paths share (Task 1.2).  Imported from its own module
# because features/__init__.py does not re-export it.
from features.pipeline import build_features

logger = get_logger(__name__)

# ── Dynamic path resolution ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

app = FastAPI(title="Indonesian Stock Prediction API", version="2.0.0")
Instrumentator().instrument(app).expose(app)

# ── Startup: load model index and warm up cache in background ─────────────
_model_index = {}

@app.on_event("startup")
async def startup_load_index():
    """Load model index at startup. Models are loaded lazily on first use."""
    global _model_index
    models_dir = os.path.join(MLRUNS_DIR, "1", "models")
    index_path = os.path.join(models_dir, "model_index.json")

    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                data = json.load(f)
            _model_index = data.get("ticker_to_model", {})
            logger.info("Loaded model index with %d tickers", len(_model_index))
        except Exception as e:
            logger.warning("Failed to load model index: %s", e)
    else:
        logger.warning("No model index found at %s", index_path)

    # Warm up cache in background (don't block startup)
    asyncio.create_task(_warmup_model_cache())
    # Also pre-warm prediction cache for fastest-first-hit
    asyncio.create_task(_warmup_prediction_cache())


async def _warmup_model_cache():
    """Gradually warm up model cache in background after startup."""
    models_dir = os.path.join(MLRUNS_DIR, "1", "models")
    logger.info("Background model warmup started (%d tickers)", len(_model_index))
    loaded = 0
    for ticker, info in list(_model_index.items())[:10]:  # Top 10 only
        if ticker in model_cache:
            continue
        model_folder = info["model_folder"]
        artifact_dir = os.path.join(models_dir, model_folder, "artifacts")
        mlmodel_file = os.path.join(artifact_dir, "MLmodel")
        if os.path.exists(mlmodel_file):
            try:
                model = mlflow.xgboost.load_model(artifact_dir)
                model_cache[ticker] = model
                loaded += 1
                logger.debug("Warmup: loaded %s", ticker)
            except Exception as e:
                logger.debug("Warmup failed for %s: %s", ticker, e)
    logger.info("Background warmup complete: %d models loaded", loaded)


async def _warmup_prediction_cache():
    """Pre-warm prediction cache for a few key tickers after startup.

    Warms only the top 5 most liquid tickers to avoid Yahoo Finance rate limits.
    Remaining tickers are cached on first user request.
    """
    priority = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]
    logger.info("Prediction cache pre-warm started (top %d tickers)...", len(priority))
    warmed = 0
    for ticker in priority:
        cached, _ = _cache_get(ticker)
        if cached is not None:
            warmed += 1
            continue
        try:
            df = await asyncio.to_thread(
                yf.download, ticker, period="250d", auto_adjust=True, progress=False
            )
            if df.empty or len(df) < 50:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = dropna(df)

            fundamentals = await asyncio.to_thread(fetch_fundamentals, ticker)
            latest = build_inference_features(df, ticker, fundamentals)

            model = load_best_model(ticker)
            trained_cols = get_feature_names(model)
            if not trained_cols:
                continue

            for col in trained_cols:
                if col not in latest.columns:
                    latest[col] = np.nan

            feature_vec = np.array(
                [float(latest[trained_cols].iloc[0].get(c, 0.0)) for c in trained_cols],
                dtype=np.float32
            )
            _cache_put(ticker, feature_vec)
            warmed += 1
            logger.debug("Pre-warmed: %s", ticker)
        except Exception as e:
            logger.debug("Pre-warm skipped for %s: %s", ticker, e)
            continue
    logger.info("Prediction cache pre-warm complete: %d/%d tickers cached",
                warmed, len(priority))

try:
    PREDICTION_COUNTER = Counter("stock_predictions_total", "Total predictions made", ["ticker", "signal"])
    CONFIDENCE_HISTOGRAM = Histogram("prediction_confidence", "Model confidence score", ["ticker"])
except ValueError:
    from prometheus_client import REGISTRY
    PREDICTION_COUNTER = REGISTRY._names_to_collectors.get("stock_predictions_total")
    CONFIDENCE_HISTOGRAM = REGISTRY._names_to_collectors.get("prediction_confidence")

model_cache = {}

# ── Prediction Cache (SQLite) — sub-second predictions after first hit ──────
import sqlite3
import pickle

PRED_CACHE_DB = os.path.join(BASE_DIR, "data", "prediction_cache.db")

def _get_cache_conn():
    os.makedirs(os.path.dirname(PRED_CACHE_DB), exist_ok=True)
    conn = sqlite3.connect(PRED_CACHE_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS cache "
                 "(ticker TEXT PRIMARY KEY, date TEXT, features BLOB, ohlcv BLOB)")
    conn.commit()
    return conn

def _cache_put(ticker: str, feature_vec: np.ndarray, ohlcv_df: pd.DataFrame | None = None):
    conn = _get_cache_conn()
    today = str(date.today())
    conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)",
                 (ticker, today,
                  pickle.dumps(feature_vec),
                  pickle.dumps(ohlcv_df) if ohlcv_df is not None else None))
    conn.commit()
    conn.close()

def _cache_get(ticker: str) -> tuple[np.ndarray | None, pd.DataFrame | None]:
    """Returns (feature_vector, ohlcv_dataframe) or (None, None) if stale/missing."""
    conn = _get_cache_conn()
    today = str(date.today())
    row = conn.execute("SELECT date, features, ohlcv FROM cache WHERE ticker = ?",
                       (ticker,)).fetchone()
    conn.close()
    if row and row[0] == today and row[1]:
        try:
            features = pickle.loads(row[1])
            ohlcv = pickle.loads(row[2]) if row[2] else None
            return features, ohlcv
        except Exception:
            pass
    return None, None

# ── Daily Caches ──────────────────────────────────────────────────────────────
_cache_date = None
_usdidr_cache = None
_fred_cache = None
_bi_cache = None
_fundamentals_cache = {}

def refresh_daily_cache():
    """Fetches macro data once per day to ensure fast API responses."""
    global _cache_date, _usdidr_cache, _fred_cache, _bi_cache
    today = str(date.today())

    if _cache_date == today:
        return

    logger.info("Refreshing Daily Macro Cache...")
    start_date = (date.today() - timedelta(days=10)).strftime("%Y-%m-%d")
    end_date = today

    try:
        _usdidr_cache = fetch_usdidr(start_date, end_date)
        _fred_cache = fetch_fred_macro(start_date, end_date)
        _bi_cache = fetch_bi_rate(start_date, end_date)
        _cache_date = today
        logger.info("Daily Macro Cache Updated")
    except Exception as e:
        logger.warning("Cache refresh failed: %s", e)

def load_best_model(ticker: str):
    """Load the best model for a ticker from the local models directory.

    Uses model index (loaded at startup) to map ticker -> model folder,
    bypassing the MLflow server entirely. Falls back to first available
    model if index is missing or ticker is not found.
    """
    if ticker in model_cache:
        return model_cache[ticker]

    models_dir = os.path.join(MLRUNS_DIR, "1", "models")
    if not os.path.isdir(models_dir):
        raise HTTPException(status_code=500, detail="Models directory not found")

    # Try model index first (loaded at startup)
    model_info = _model_index.get(ticker)

    if model_info:
        model_folder = model_info["model_folder"]
        artifact_dir = os.path.join(models_dir, model_folder, "artifacts")
        mlmodel_file = os.path.join(artifact_dir, "MLmodel")
        if os.path.exists(mlmodel_file):
            try:
                model = mlflow.xgboost.load_model(artifact_dir)
                logger.info("Loaded model for %s from index: %s", ticker, model_folder)
                model_cache[ticker] = model
                return model
            except Exception as e:
                logger.warning("Failed to load indexed model for %s: %s", ticker, e)

    # Fallback: use preloaded fallback model
    if "_fallback" in model_cache:
        logger.warning("Using fallback model for %s (not ideal)", ticker)
        model_cache[ticker] = model_cache["_fallback"]
        return model_cache[ticker]

    # Last resort: scan all models and use first valid one
    logger.warning("Model index miss for %s, scanning all models (slow!)", ticker)
    for model_folder in sorted(os.listdir(models_dir)):
        artifact_dir = os.path.join(models_dir, model_folder, "artifacts")
        mlmodel_file = os.path.join(artifact_dir, "MLmodel")
        if not os.path.exists(mlmodel_file):
            continue
        try:
            model = mlflow.xgboost.load_model(artifact_dir)
            logger.info("Fallback: loaded model for %s from %s", ticker, model_folder)
            model_cache[ticker] = model
            return model
        except Exception:
            continue

    raise HTTPException(status_code=404, detail=f"No model found for {ticker}")

def get_feature_names(model) -> list:
    """Extract expected feature names from a model (handles calibration wrappers)."""
    # XGBoost models store feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # CalibratedClassifierCV wraps the base estimator
    if hasattr(model, "estimator") and hasattr(model.estimator, "feature_names_in_"):
        return list(model.estimator.feature_names_in_)
    # Try base estimator attribute
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        return booster.feature_names if booster.feature_names else []
    return []

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED FEATURE BUILDING FOR INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

# The inference-time universe snapshot.  Module-level names rather than literals
# inside load_universe_context(), so a test can point them at a tmp_path without
# touching the real data/ tree.
UNIVERSE_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "stocks.csv")
IHSG_PATH = os.path.join(BASE_DIR, "data", "raw", "ihsg.csv")

# How much history the cross-stock (breadth/sector) pass is given at inference.
# It only ever contributes the latest row, whose every column needs at most six
# rows of history, so 30 leaves a five-fold margin while keeping the pass ~30x
# cheaper than walking the full 1500-row universe per request.  Configurable as
# a module attribute so the equivalence test can compare it against the full
# history.
CROSS_STOCK_WINDOW_ROWS = 30

_universe_context = None


def load_universe_context():
    """Load the daily universe snapshot that inference-time context needs.

    The market-context and sector features the builder computes belong to a
    ticker *relative to the universe* (IHSG, its sector peers, breadth), so a
    single request's OHLCV frame is not enough to build them — this is where
    serving gets the rest.

    Returns:
        ``(raw_ohlcv, ihsg, all_ticker_data)``: the multi-ticker OHLCV frame in
        yfinance ``group_by="ticker"`` layout, the IHSG index frame, and
        ``{ticker: dropna(raw_ohlcv[ticker])}`` for every configured ticker.

    Any source that cannot be read degrades to an empty frame/dict with a
    ``warning``, so a missing file yields absent/NaN context columns instead of
    an exception.  The result is cached for the life of the process, so a
    request pays the disk read at most once.
    """
    global _universe_context
    if _universe_context is not None:
        return _universe_context

    raw_ohlcv = pd.DataFrame()
    try:
        raw_ohlcv = load_data(UNIVERSE_RAW_PATH)
    except Exception as e:
        logger.warning("Universe raw data unavailable at %s: %s", UNIVERSE_RAW_PATH, e)

    ihsg = pd.DataFrame()
    try:
        ihsg = load_ihsg_data(IHSG_PATH)
    except Exception as e:
        logger.warning("IHSG data unavailable at %s: %s", IHSG_PATH, e)

    # Same per-ticker extraction build_feature_set does, so the cross-stock
    # features are computed from identical inputs on both paths.
    all_ticker_data = {}
    for ticker in TICKERS:
        try:
            all_ticker_data[ticker] = dropna(raw_ohlcv[ticker].copy())
        except KeyError:
            pass

    _universe_context = (raw_ohlcv, ihsg, all_ticker_data)
    return _universe_context


def universe_age_days() -> int:
    """Age in whole days of the raw universe file the features are built from.

    Reads the *universe* file's mtime (``UNIVERSE_RAW_PATH``) — not IHSG's, and
    not the process's.  The universe frame is what the market-context and sector
    features are computed from, so it is the file whose staleness makes those
    features stale.

    Returns:
        Whole days since that file was last written, or ``-1`` when the file is
        absent.  ``-1`` means *unknown*, never *fresh*: an absent file has no
        age, and reporting 0 would claim it had just been ingested.

    Note that this stats the file on disk, while the features actually served
    come from the snapshot cached at load_universe_context().  After a
    mid-process ingest the two disagree: this reports age 0 while serving still
    uses the older cached frame, so the number is evidence about the file, not
    about the features in the response.
    """
    try:
        mtime = os.path.getmtime(UNIVERSE_RAW_PATH)
    except OSError as e:
        logger.warning("Universe file %s unavailable — staleness unknown: %s",
                       UNIVERSE_RAW_PATH, e)
        return -1
    return int((time.time() - mtime) // 86400)


def build_inference_features(df: pd.DataFrame, ticker: str,
                             fundamentals: dict) -> pd.DataFrame:
    """Build the features for one inference request with the training builder.

    Delegates to the same ``build_features`` the training path uses, fed from
    the daily caches instead of the pipeline's per-run fetches, so a feature is
    either a real computed value or genuinely absent — never a placeholder.

    Args:
        df: OHLCV DataFrame for a single ticker (250 days), single-level columns
            as returned by ``yfinance.download(ticker)``
        ticker: Ticker symbol
        fundamentals: Dict of fundamental metrics from fetch_fundamentals()

    Returns:
        Latest row as a 1-row DataFrame, or an empty DataFrame when the ticker
        has fewer than the builder's 50 usable rows (the API layer rejects that
        case before calling).
    """
    # Three feature families are deliberately NOT supplied here, each for its own
    # measured reason:
    #
    # * vp_* (19 columns) is LIVE in training — no vp_* column is all-NaN in the
    #   committed data/processed/features.parquet — but it needs 5-minute
    #   intraday downloads, which serving cannot reproduce.  Excluded from the
    #   contract rather than faked.
    # * google_trend (1 column) is LIVE in training as well (build_feature_set
    #   fetches real values, and the committed parquet carries 98 distinct ones).
    #   Serving cannot reproduce the fetch, so passing any constant would
    #   fabricate a model-consumed column; None is passed, which makes the
    #   builder materialise no column at all.  Excluded from the contract too.
    # * idn_* (10 columns) is already DEAD in training: fetch_idx_fundamentals
    #   (src/features/fetchers.py) is a no-network stub that returns NaN for
    #   every field, and all ten columns are all-NaN in the committed parquet.
    #   There is nothing to reproduce.
    #
    # Because absence is a standing decision rather than a runtime failure, it is
    # not logged per request (Task 1.9 owns failure visibility).  The model's
    # feature contract (feature_schema.json, Task 1.4) is derived so that it
    # never expects a column serving cannot supply.
    raw_ohlcv, ihsg, all_ticker_data = load_universe_context()

    # build_features selects its ticker with df[ticker], i.e. it wants the
    # multi-ticker yfinance group_by="ticker" layout, while a serving request
    # carries one ticker with flat columns.  Widen the frame here instead of
    # loosening the builder's contract: one top-level entry for this ticker,
    # its fields underneath.
    multi = pd.concat({ticker: df}, axis=1)

    # Sentiment is fetched the way serving has always fetched it: flag-gated,
    # and 0.0 on any failure.  It is handed to the builder as a value rather
    # than written into the frame so the builder keeps ownership of the column.
    news_sentiment = 0.0
    if FEATURE_FLAGS.get("news_sentiment", True):
        try:
            from features.fetchers import fetch_news_sentiment
            news_sentiment = fetch_news_sentiment(ticker)
        except Exception:
            news_sentiment = 0.0

    featured = build_features(
        multi, ticker,
        fundamentals=fundamentals,
        usdidr=_usdidr_cache,
        fred_macro=_fred_cache,
        bi_rate=_bi_cache,
        ihsg=ihsg,
        all_ticker_data=all_ticker_data,
        news_sentiment=news_sentiment,
        # None, not a constant: a value here would materialise a fabricated
        # google_trend column at inference while training carries real fetched
        # values (see the family rationale above).
        google_trend=None,
        vp_features=None,
        idx_fundamentals=None,
        mode="full",
        with_target=False,
    )

    if featured is None:
        logger.warning("Fewer than 50 usable rows for %s — no features built", ticker)
        return pd.DataFrame()

    # Cross-stock features (breadth, sector return/rank/volume) are applied by
    # build_feature_set after every ticker is engineered, because they need the
    # whole universe.  Serving has the whole universe from
    # load_universe_context(), so it applies them at the same point for the same
    # reason.
    if FEATURE_FLAGS.get("market_context", True) and all_ticker_data:
        # compute_cross_stock_features walks every index label against every
        # ticker, so on the real universe (1553 rows x 45 tickers) the full
        # history costs ~29 s of CPU.  Serving returns only the latest row, and
        # every cross-stock column for that row is a function of at most the last
        # six rows (a 5-day return needs six), so the same function is applied to
        # a bounded tail window that still contains the row being served.  The
        # served row is identical either way — pinned by
        # tests/test_serve_parity.py::test_cross_stock_window_matches_full_history
        # and measured on the real universe in task-1.3-report.md.  Without the
        # window the first request per ticker stalls the event loop for ~30 s,
        # because this function is called synchronously from the async handler.
        cutoff = featured.index[-CROSS_STOCK_WINDOW_ROWS:][0]
        featured = compute_cross_stock_features(
            featured.loc[cutoff:].copy(),
            ticker,
            {t: d.loc[cutoff:] for t, d in all_ticker_data.items()},
            TICKERS,
        )

    numeric_df = featured.select_dtypes(include=[np.number])
    return numeric_df.iloc[[-1]].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# API MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    ticker: str

class BatchPredictRequest(BaseModel):
    tickers: List[str]

class PredictResponse(BaseModel):
    ticker: str
    prediction: int
    probability_up: float
    signal: str
    sentiment_score: float = 0.0
    probability_adjusted: float = 0.0
    signal_adjusted: str = ""
    drift_warning: bool = False
    drift_score: float = 0.0

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

# ── Sentiment overlay strength ────────────────────────────────────────────────
SENTIMENT_WEIGHT = 0.10
PSI_THRESHOLD = 0.25  # Population Stability Index threshold for drift warning

# ── PSI (Population Stability Index) — distribution drift check ─────────────
def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between expected (training) and actual (live) distributions.

    PSI > 0.25 indicates significant drift — predictions may be unreliable.
    """
    all_vals = np.concatenate([expected, actual])
    bin_edges = np.percentile(all_vals, np.linspace(0, 100, bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_hist, _ = np.histogram(expected, bins=bin_edges)
    act_hist, _ = np.histogram(actual, bins=bin_edges)

    exp_hist = (exp_hist / exp_hist.sum()).clip(0.001)
    act_hist = (act_hist / act_hist.sum()).clip(0.001)

    psi = np.sum((act_hist - exp_hist) * np.log(act_hist / exp_hist))
    return float(psi)

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Indonesian Stock Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/tickers")
def list_tickers():
    return {"tickers": TICKERS, "count": len(TICKERS)}

@app.get("/health")
def health():
    return {"status": "ok"}

def _apply_sentiment_overlay(ticker: str, prob: float, background_tasks=None) -> dict:
    """Adjust model probability with news sentiment overlay.

    Fetches latest sentiment and tweaks probability by up to SENTIMENT_WEIGHT.
    Falls back to 0.0 sentiment immediately on any error (no blocking).

    Returns dict with sentiment_score, probability_adjusted, signal_adjusted.
    """
    sentiment = 0.0
    try:
        from features.fetchers import fetch_news_sentiment
        sentiment = fetch_news_sentiment(ticker)
        sentiment = max(-1.0, min(1.0, sentiment))
    except Exception:
        pass  # sentiment stays 0.0 — never block prediction on sentiment failure

    adjustment = sentiment * SENTIMENT_WEIGHT
    prob_adj = prob + adjustment
    prob_adj = max(0.0, min(1.0, prob_adj))
    signal_adj = "BUY" if prob_adj >= 0.50 else "SELL"

    return {
        "sentiment_score": round(sentiment, 4),
        "probability_adjusted": round(prob_adj, 4),
        "signal_adjusted": signal_adj,
    }

@app.get("/cache")
def cache_status():
    """Show prediction cache status."""
    conn = _get_cache_conn()
    today = str(date.today())
    row = conn.execute("SELECT COUNT(*) FROM cache WHERE date = ?", (today,)).fetchone()
    cached_count = row[0] if row else 0
    total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    conn.close()
    return {
        "cached_today": cached_count,
        "total_cached": total,
        "total_tickers": len(TICKERS),
        "coverage": f"{cached_count}/{len(TICKERS)} ({cached_count*100//len(TICKERS)}%)",
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """Predict BUY/SELL signal for a single ticker.

    Checks daily prediction cache first — cached path returns in <0.2s.
    On cache miss, downloads data + computes features (~5-15s).
    """
    try:
        global _fundamentals_cache

        # 1. Trigger background cache refresh
        background_tasks.add_task(refresh_daily_cache)

        # ── Fast path: check prediction cache ──────────────────────────
        cached_features, cached_ohlcv = _cache_get(request.ticker)
        model = load_best_model(request.ticker)
        trained_cols = get_feature_names(model)
        if not trained_cols:
            raise HTTPException(status_code=500, detail="Model has no feature names")

        if cached_features is not None:
            logger.debug("Cache HIT for %s — sub-200ms prediction", request.ticker)
            # Align cached features to model columns
            latest_dict = {col: 0.0 for col in trained_cols}
            for i, col in enumerate(trained_cols):
                if i < len(cached_features):
                    latest_dict[col] = float(cached_features[i])
            latest_df = pd.DataFrame([latest_dict])

            prediction = int(model.predict(latest_df)[0])
            probability = float(model.predict_proba(latest_df)[0][1])
            signal = "BUY" if prediction == 1 else "SELL"
            overlay = _apply_sentiment_overlay(request.ticker, probability)
            PREDICTION_COUNTER.labels(ticker=request.ticker, signal=signal).inc()
            CONFIDENCE_HISTOGRAM.labels(ticker=request.ticker).observe(probability)
            return PredictResponse(
                ticker=request.ticker, prediction=prediction,
                probability_up=round(probability, 4), signal=signal,
                sentiment_score=overlay["sentiment_score"],
                probability_adjusted=overlay["probability_adjusted"],
                signal_adjusted=overlay["signal_adjusted"],
                drift_warning=False, drift_score=0.0,
            )

        # ── Slow path: full pipeline (download + features + cache) ──────
        logger.info("Cache MISS for %s — running full pipeline", request.ticker)

        # 2. Async OHLCV download
        df = await asyncio.to_thread(
            yf.download, request.ticker, period="250d", auto_adjust=True, progress=False
        )
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data on Yahoo Finance for {request.ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = dropna(df)

        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Not enough trading days for TA calculation")

        # 3. Fundamentals (async + cached daily)
        today = str(date.today())
        if request.ticker not in _fundamentals_cache or \
           _fundamentals_cache[request.ticker].get("date") != today:
            fundamentals = await asyncio.to_thread(fetch_fundamentals, request.ticker)
            _fundamentals_cache[request.ticker] = {"data": fundamentals, "date": today}
        else:
            fundamentals = _fundamentals_cache[request.ticker]["data"]

        # 4. Build features using shared functions
        latest = build_inference_features(df, request.ticker, fundamentals)

        # 5. Align features to trained columns
        for col in trained_cols:
            if col not in latest.columns:
                latest[col] = np.nan

        latest_row = latest[trained_cols].iloc[0]
        latest_df = pd.DataFrame([latest_row])

        # 6. Cache feature vector for next request (fast path)
        try:
            feature_vec = np.array([float(latest_row.get(c, 0.0)) for c in trained_cols],
                                   dtype=np.float32)
            _cache_put(request.ticker, feature_vec)
            logger.debug("Cached features for %s (%d cols)", request.ticker, len(trained_cols))
        except Exception as e:
            logger.debug("Failed to cache features for %s: %s", request.ticker, e)

        prediction = int(model.predict(latest_df)[0])
        probability = float(model.predict_proba(latest_df)[0][1])

        signal = "BUY" if prediction == 1 else "SELL"
        overlay = _apply_sentiment_overlay(request.ticker, probability)
        PREDICTION_COUNTER.labels(ticker=request.ticker, signal=signal).inc()
        CONFIDENCE_HISTOGRAM.labels(ticker=request.ticker).observe(probability)

        return PredictResponse(
            ticker=request.ticker,
            prediction=prediction,
            probability_up=round(probability, 4),
            signal=signal,
            sentiment_score=overlay["sentiment_score"],
            probability_adjusted=overlay["probability_adjusted"],
            signal_adjusted=overlay["signal_adjusted"],
            drift_warning=False, drift_score=0.0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Crash on %s: %s", request.ticker, e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest, background_tasks: BackgroundTasks):
    """Predict BUY/SELL signals for multiple tickers in one request.

    Fetches OHLCV data in parallel for all tickers.
    """
    if not request.tickers:
        raise HTTPException(status_code=400, detail="tickers list cannot be empty")

    # Trigger cache refresh
    background_tasks.add_task(refresh_daily_cache)

    async def predict_single(ticker: str) -> PredictResponse:
        """Predict for one ticker (used with asyncio.gather)."""
        try:
            df = await asyncio.to_thread(
                yf.download, ticker, period="250d", auto_adjust=True, progress=False
            )
            if df.empty:
                raise ValueError(f"No Yahoo Finance data for {ticker}")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df = dropna(df)
            if len(df) < 50:
                raise ValueError(f"Not enough data for {ticker}")

            fundamentals_today = str(date.today())
            if ticker not in _fundamentals_cache or \
               _fundamentals_cache[ticker].get("date") != fundamentals_today:
                fundamentals = await asyncio.to_thread(fetch_fundamentals, ticker)
                _fundamentals_cache[ticker] = {"data": fundamentals, "date": fundamentals_today}
            else:
                fundamentals = _fundamentals_cache[ticker]["data"]

            latest = build_inference_features(df, ticker, fundamentals)
            model = load_best_model(ticker)
            trained_cols = get_feature_names(model)

            for col in trained_cols:
                if col not in latest.columns:
                    latest[col] = np.nan

            latest_df = pd.DataFrame([latest[trained_cols].iloc[0]])
            prediction = int(model.predict(latest_df)[0])
            probability = float(model.predict_proba(latest_df)[0][1])
            signal = "BUY" if prediction == 1 else "SELL"
            overlay = _apply_sentiment_overlay(ticker, probability)

            PREDICTION_COUNTER.labels(ticker=ticker, signal=signal).inc()
            CONFIDENCE_HISTOGRAM.labels(ticker=ticker).observe(probability)

            return PredictResponse(
                ticker=ticker,
                prediction=prediction,
                probability_up=round(probability, 4),
                signal=signal,
                sentiment_score=overlay["sentiment_score"],
                probability_adjusted=overlay["probability_adjusted"],
                signal_adjusted=overlay["signal_adjusted"],
                drift_warning=False, drift_score=0.0,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("Batch predict failed for %s: %s", ticker, e)
            return PredictResponse(
                ticker=ticker,
                prediction=0,
                probability_up=0.0,
                signal=f"ERROR: {str(e)[:50]}",
            )

    tasks = [predict_single(t) for t in request.tickers]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    return BatchPredictResponse(predictions=list(results))

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)