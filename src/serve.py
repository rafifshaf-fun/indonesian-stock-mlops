import sys, os
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

try:
    PREDICTION_COUNTER = Counter("stock_predictions_total", "Total predictions made", ["ticker", "signal"])
    CONFIDENCE_HISTOGRAM = Histogram("prediction_confidence", "Model confidence score", ["ticker"])
except ValueError:
    from prometheus_client import REGISTRY
    PREDICTION_COUNTER = REGISTRY._names_to_collectors.get("stock_predictions_total")
    CONFIDENCE_HISTOGRAM = REGISTRY._names_to_collectors.get("prediction_confidence")

model_cache = {}

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

    Bypasses MLflow artifact URIs (which break with Windows→Docker path mismatch).
    Queries the tracking server only for run_id, then loads from filesystem.
    """
    if ticker in model_cache:
        return model_cache[ticker]

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.ticker = '{ticker}'",
        order_by=["metrics.avg_roc_auc DESC"],
        max_results=1,
    )

    if not runs:
        raise HTTPException(status_code=404, detail=f"No model found for ticker {ticker}")

    run_id = runs[0].info.run_id
    auc = runs[0].data.metrics.get("avg_roc_auc", "?")
    logger.info("Loading model for %s: run %s (AUC=%s)", ticker, run_id, auc)

    # Search models/ directory for artifact folder matching this run_id
    models_dir = os.path.join(MLRUNS_DIR, experiment.experiment_id, "models")
    if not os.path.isdir(models_dir):
        raise HTTPException(status_code=500, detail="Models directory not found")

    for model_folder in os.listdir(models_dir):
        artifact_dir = os.path.join(models_dir, model_folder, "artifacts")
        mlmodel_file = os.path.join(artifact_dir, "MLmodel")
        if not os.path.exists(mlmodel_file):
            continue
        try:
            with open(mlmodel_file, "r") as f:
                if run_id in f.read():
                    model = mlflow.xgboost.load_model(artifact_dir)
                    model_cache[ticker] = model
                    return model
        except Exception:
            continue

    raise HTTPException(status_code=500, detail=f"Model files not found for {ticker} (run {run_id})")

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

    # IDX fundamentals placeholders
    idx_fields = ["idn_der", "idn_roe", "idn_roa", "idn_current_ratio",
                  "idn_gpm", "idn_npm", "idn_eps_growth", "idn_bvps",
                  "idn_per", "idn_pbv"]
    for f in idx_fields:
        df[f] = np.nan

    # Volume profile placeholders (skip at inference)
    vp_fields = ["vp_poc", "vp_poc_distance", "vp_vah", "vp_val",
                 "vp_value_area_width", "vp_close_in_value_area", "vp_volume_skew",
                 "vp_vwap_deviation", "vp_volume_at_close", "vp_profile_shape",
                 "vp_open_drive", "vp_close_drive", "vp_intraday_volatility",
                 "vp_intraday_range", "vp_intraday_trend", "vp_relative_volume",
                 "vp_volume_trend_5d", "vp_high_volume_nodes", "vp_vwap_deviation_std"]
    for f in vp_fields:
        df[f] = np.nan

    # Market context placeholders
    ctx_fields = ["ihsg_return_1d", "ihsg_return_5d", "relative_strength_1d",
                  "relative_strength_5d", "beta_20d", "correlation_20d",
                  "market_breadth", "market_advance_decline", "sector_return_1d",
                  "sector_relative_strength", "sector_rank_5d", "sector_volume_ratio",
                  "usdidr_stress"]
    for f in ctx_fields:
        df[f] = np.nan

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

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

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

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """Predict BUY/SELL signal for a single ticker."""
    try:
        global _fundamentals_cache

        # 1. Trigger background cache refresh
        background_tasks.add_task(refresh_daily_cache)

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

        # 5. Load model and predict
        model = load_best_model(request.ticker)
        trained_cols = get_feature_names(model)

        if not trained_cols:
            raise HTTPException(status_code=500, detail="Model has no feature names")

        # Align features to trained columns
        for col in trained_cols:
            if col not in latest.columns:
                latest[col] = np.nan

        latest_df = pd.DataFrame([latest[trained_cols].iloc[0]])

        prediction = int(model.predict(latest_df)[0])
        probability = float(model.predict_proba(latest_df)[0][1])

        signal = "BUY" if prediction == 1 else "SELL"
        PREDICTION_COUNTER.labels(ticker=request.ticker, signal=signal).inc()
        CONFIDENCE_HISTOGRAM.labels(ticker=request.ticker).observe(probability)

        return PredictResponse(
            ticker=request.ticker,
            prediction=prediction,
            probability_up=round(probability, 4),
            signal=signal,
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

            PREDICTION_COUNTER.labels(ticker=ticker, signal=signal).inc()
            CONFIDENCE_HISTOGRAM.labels(ticker=ticker).observe(probability)

            return PredictResponse(
                ticker=ticker,
                prediction=prediction,
                probability_up=round(probability, 4),
                signal=signal,
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