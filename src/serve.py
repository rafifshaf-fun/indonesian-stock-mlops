import sys, os
sys.path.append(os.path.dirname(__file__))

import mlflow
import mlflow.xgboost 
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from ta import add_all_ta_features
from ta.utils import dropna
from features import fetch_fundamentals, fetch_usdidr, fetch_fred_macro, fetch_bi_rate
from datetime import date, timedelta
import asyncio
import traceback

# ── Dynamic path resolution ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

app = FastAPI(title="Indonesian Stock Prediction API", version="1.0.0")
Instrumentator().instrument(app).expose(app)

try:
    PREDICTION_COUNTER = Counter("stock_predictions_total", "Total predictions made", ["ticker", "signal"])
    CONFIDENCE_HISTOGRAM = Histogram("prediction_confidence", "Model confidence score", ["ticker"])
except ValueError:
    from prometheus_client import REGISTRY
    PREDICTION_COUNTER = REGISTRY._names_to_collectors.get("stock_predictions_total")
    CONFIDENCE_HISTOGRAM = REGISTRY._names_to_collectors.get("prediction_confidence")

MLFLOW_EXPERIMENT = "indonesian-stock-prediction"
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

    print("🔄 Refreshing Daily Macro Cache...")
    start_date = (date.today() - timedelta(days=10)).strftime("%Y-%m-%d")
    end_date = today

    try:
        _usdidr_cache = fetch_usdidr(start_date, end_date)
        _fred_cache = fetch_fred_macro(start_date, end_date)
        _bi_cache = fetch_bi_rate(start_date, end_date)
        _cache_date = today
        print("✅ Daily Macro Cache Updated")
    except Exception as e:
        print(f"⚠️ Cache refresh failed: {e}")

def load_best_model(ticker: str):
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
        max_results=1
    )

    if not runs:
        raise HTTPException(status_code=404, detail=f"No model found for ticker {ticker}")

    run_id = runs[0].info.run_id
    experiment_id = experiment.experiment_id

    models_dir = os.path.join(MLRUNS_DIR, experiment_id, "models")
    if not os.path.exists(models_dir):
        raise HTTPException(status_code=404, detail=f"No model artifacts found for {ticker}")

    model_path = None
    for model_folder in os.listdir(models_dir):
        mlmodel_file = os.path.join(models_dir, model_folder, "artifacts", "MLmodel")
        if os.path.exists(mlmodel_file):
            with open(mlmodel_file, "r") as f:
                content = f.read()
                if f"run_id: {run_id}" in content:
                    model_path = os.path.join(models_dir, model_folder, "artifacts")
                    break

    if not model_path:
        raise HTTPException(status_code=404, detail=f"Model files not found for {ticker}")

    model = mlflow.xgboost.load_model(model_path)
    model_cache[ticker] = model
    return model

class PredictRequest(BaseModel):
    ticker: str

class PredictResponse(BaseModel):
    ticker: str
    prediction: int
    probability_up: float
    signal: str

@app.get("/")
def root():
    return {"message": "Indonesian Stock Prediction API", "docs": "/docs", "health": "/health"}

@app.get("/tickers")
def list_tickers():
    return {"tickers": [
        "AADI.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK", "AMRT.JK", "ANTM.JK", "ARTO.JK", 
        "ASII.JK", "BBCA.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK", "BMRI.JK", "BREN.JK", "BRIS.JK", 
        "BRPT.JK", "CPIN.JK", "CTRA.JK", "EXCL.JK", "GOTO.JK", "ICBP.JK", "INCO.JK", "INDF.JK", 
        "INKP.JK", "ISAT.JK", "ITMG.JK", "JPFA.JK", "JSMR.JK", "KLBF.JK", "MAPA.JK", "MAPI.JK", 
        "MBMA.JK", "MDKA.JK", "MEDC.JK", "PGAS.JK", "PGEO.JK", "PTBA.JK", "SIDO.JK", "SMGR.JK", 
        "SMRA.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK",
    ]}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    try:
        global _fundamentals_cache
        
        # 1. Trigger background cache refresh safely
        background_tasks.add_task(refresh_daily_cache)

        # 2. Async OHLCV Download
        df = await asyncio.to_thread(
            yf.download, request.ticker, period="250d", auto_adjust=True, progress=False
        )
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data found on Yahoo Finance for {request.ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = dropna(df)
        
        if len(df) < 50: # Ensure we have enough data for 50-day moving averages
             raise HTTPException(status_code=400, detail=f"Not enough valid trading days for TA calculation")

        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )

        df["ret_1d"] = df["Close"].pct_change()
        df["ret_5d"] = df["Close"].pct_change(periods=5)

        if "trend_sma_fast" in df.columns:
            df["dist_from_sma50"] = df["Close"] / df["trend_sma_fast"] - 1
        if "trend_sma_slow" in df.columns:
            df["dist_from_sma200"] = df["Close"] / df["trend_sma_slow"] - 1

        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df) == 0:
            raise HTTPException(status_code=400, detail="Data extraction failed after TA calculation")
            
        latest = numeric_df.iloc[[-1]].copy()

        # 3. Async Fundamentals + Daily Cache
        today = str(date.today())
        if request.ticker not in _fundamentals_cache or _fundamentals_cache[request.ticker].get("date") != today:
            fundamentals = await asyncio.to_thread(fetch_fundamentals, request.ticker)
            _fundamentals_cache[request.ticker] = {"data": fundamentals, "date": today}
        else:
            fundamentals = _fundamentals_cache[request.ticker]["data"]

        for col, val in fundamentals.items():
            latest[col] = val

        # 4. Bulletproof Macro Caches Injection
        try:
            if _usdidr_cache is not None and not _usdidr_cache.empty and "usdidr_rate" in _usdidr_cache.columns:
                latest["usdidr_rate"] = _usdidr_cache["usdidr_rate"].iloc[-1]
                latest["usdidr_return"] = _usdidr_cache["usdidr_return"].iloc[-1]
            else:
                latest["usdidr_rate"] = np.nan
                latest["usdidr_return"] = 0.0
        except Exception:
            latest["usdidr_rate"] = np.nan
            latest["usdidr_return"] = 0.0

        try:
            if _fred_cache is not None and not _fred_cache.empty:
                for col in _fred_cache.columns:
                    if col in _fred_cache:
                        latest[col] = _fred_cache[col].iloc[-1]
        except Exception:
            pass 

        try:
            if _bi_cache is not None and not _bi_cache.empty:
                latest["bi_rate_official"] = _bi_cache.iloc[-1]
            else:
                latest["bi_rate_official"] = np.nan
        except Exception:
            latest["bi_rate_official"] = np.nan

        latest["google_trend"] = 0.0

        # 5. Align features and predict
        model = load_best_model(request.ticker)
        
        trained_cols = model.feature_names_in_
        for col in trained_cols:
            if col not in latest.columns:
                latest[col] = np.nan

        # Select only the expected columns in the exact order and convert to 2D
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
            signal=signal
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"CRASH ON {request.ticker}:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)