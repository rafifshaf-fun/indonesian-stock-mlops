import sys, os
sys.path.append(os.path.dirname(__file__))

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from ta import add_all_ta_features
from ta.utils import dropna

mlflow.set_tracking_uri("http://host.docker.internal:5000")

app = FastAPI(title="Indonesian Stock Prediction API", version="1.0.0")
Instrumentator().instrument(app).expose(app)

try:
    PREDICTION_COUNTER = Counter(
        "stock_predictions_total", "Total predictions made", ["ticker", "signal"]
    )
    CONFIDENCE_HISTOGRAM = Histogram(
        "prediction_confidence", "Model confidence score", ["ticker"]
    )
except ValueError:
    from prometheus_client import REGISTRY
    PREDICTION_COUNTER = REGISTRY._names_to_collectors.get("stock_predictions_total")
    CONFIDENCE_HISTOGRAM = REGISTRY._names_to_collectors.get("prediction_confidence")

MLFLOW_EXPERIMENT = "indonesian-stock-prediction"
model_cache = {}

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
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    model_cache[ticker] = model
    return model

class PredictRequest(BaseModel):
    ticker: str  # e.g. "BBCA.JK"

class PredictResponse(BaseModel):
    ticker: str
    prediction: int
    probability_up: float
    signal: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tickers")
def list_tickers():
    return {"tickers": [
        "AADI.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK",
        "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK", "BBCA.JK",
        "BBNI.JK", "BBRI.JK", "BBTN.JK", "BMRI.JK", "BREN.JK",
        "BRIS.JK", "BRPT.JK", "CPIN.JK", "CTRA.JK", "EXCL.JK",
        "GOTO.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INKP.JK",
        "ISAT.JK", "ITMG.JK", "JPFA.JK", "JSMR.JK", "KLBF.JK",
        "MAPA.JK", "MAPI.JK", "MBMA.JK", "MDKA.JK", "MEDC.JK",
        "PGAS.JK", "PGEO.JK", "PTBA.JK", "SIDO.JK", "SMGR.JK",
        "SMRA.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK",
    ]}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Fetch latest market data and compute features automatically
    df = yf.download(request.ticker, period="60d", auto_adjust=True, progress=False)

    if df.empty:
        raise HTTPException(status_code=400, detail=f"No data found for {request.ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = dropna(df)
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low",
        close="Close", volume="Volume", fillna=True
    )

    # Use the latest row as prediction input
    latest = df.iloc[[-1]].select_dtypes(include=[np.number])

    model = load_best_model(request.ticker)

    try:
        prediction = int(model.predict(latest)[0])
        probability = float(model.predict_proba(latest)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    signal = "BUY" if prediction == 1 else "SELL"

    PREDICTION_COUNTER.labels(ticker=request.ticker, signal=signal).inc()
    CONFIDENCE_HISTOGRAM.labels(ticker=request.ticker).observe(probability)

    return PredictResponse(
        ticker=request.ticker,
        prediction=prediction,
        probability_up=round(probability, 4),
        signal=signal
    )

@app.get("/")
def root():
    return {"message": "Indonesian Stock Prediction API", "docs": "/docs", "health": "/health"}

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)