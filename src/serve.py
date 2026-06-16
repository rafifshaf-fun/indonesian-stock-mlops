import sys, os, json
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
    compute_ta_features, compute_custom_features, compute_enhanced_mas,
    compute_ict_features, inject_macro_features,
    fetch_fundamentals, fetch_usdidr, fetch_fred_macro, fetch_bi_rate,
)

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

def build_inference_features(df: pd.DataFrame, ticker: str,
                             fundamentals: dict) -> pd.DataFrame:
    """Build features for a single inference request using shared functions.

    Mirrors the training feature pipeline but designed for real-time use:
    - Skips intraday volume profile (too slow)
    - Skips cross-stock features (not available at inference without full dataset)
    - Skips ICT features if flag disabled

    Args:
        df: OHLCV DataFrame for a single ticker (250 days)
        ticker: Ticker symbol
        fundamentals: Dict of fundamental metrics

    Returns:
        Latest row as a 1-row DataFrame with all features.
    """
    if FEATURE_FLAGS.get("ta_indicators", True):
        df = compute_ta_features(df)

    df = compute_custom_features(df)

    if FEATURE_FLAGS.get("enhanced_mas", True):
        df = compute_enhanced_mas(df)

    if FEATURE_FLAGS.get("ict_suite", True):
        df = compute_ict_features(df)

    # Inject macro from daily cache
    usdidr = _usdidr_cache if _usdidr_cache is not None else pd.DataFrame()
    fred_macro = _fred_cache if _fred_cache is not None else pd.DataFrame()
    bi_rate = _bi_cache if _bi_cache is not None else pd.Series(dtype=float)

    df = inject_macro_features(df, fundamentals, usdidr, fred_macro, bi_rate)
    df["google_trend"] = 0.0

    # News sentiment
    if FEATURE_FLAGS.get("news_sentiment", True):
        try:
            from features.fetchers import fetch_news_sentiment
            df["news_sentiment"] = fetch_news_sentiment(ticker)
        except Exception:
            df["news_sentiment"] = 0.0
    else:
        df["news_sentiment"] = 0.0

    # Placeholder columns — batch assign via dict to avoid fragmentation
    placeholders = {}
    for f in ["idn_der", "idn_roe", "idn_roa", "idn_current_ratio",
              "idn_gpm", "idn_npm", "idn_eps_growth", "idn_bvps",
              "idn_per", "idn_pbv"]:
        placeholders[f] = np.nan
    for f in ["vp_poc", "vp_poc_distance", "vp_vah", "vp_val",
              "vp_value_area_width", "vp_close_in_value_area", "vp_volume_skew",
              "vp_vwap_deviation", "vp_volume_at_close", "vp_profile_shape",
              "vp_open_drive", "vp_close_drive", "vp_intraday_volatility",
              "vp_intraday_range", "vp_intraday_trend", "vp_relative_volume",
              "vp_volume_trend_5d", "vp_high_volume_nodes", "vp_vwap_deviation_std"]:
        placeholders[f] = np.nan
    for f in ["ihsg_return_1d", "ihsg_return_5d", "relative_strength_1d",
              "relative_strength_5d", "beta_20d", "correlation_20d",
              "market_breadth", "market_advance_decline", "sector_return_1d",
              "sector_relative_strength", "sector_rank_5d", "sector_volume_ratio",
              "usdidr_stress"]:
        placeholders[f] = np.nan
    df = pd.concat([df, pd.DataFrame(placeholders, index=df.index)], axis=1)

    # Return latest row
    numeric_df = df.select_dtypes(include=[np.number])
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