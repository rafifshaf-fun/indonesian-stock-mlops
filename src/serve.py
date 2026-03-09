import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Indonesian Stock Prediction API", version="1.0.0")
Instrumentator().instrument(app).expose(app)

MLFLOW_EXPERIMENT = "indonesian-stock-prediction"
model_cache = {}

def load_best_model(ticker: str):
    """Load the latest MLflow run model for a given ticker."""
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
    model_cache[ticker] = model  # cache it so we don't reload every request
    return model

class PredictRequest(BaseModel):
    ticker: str
    features: dict  # key-value pairs of feature name → value

class PredictResponse(BaseModel):
    ticker: str
    prediction: int        # 0 = down, 1 = up
    probability_up: float
    signal: str            # "BUY" or "SELL"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tickers")
def list_tickers():
    from config import TICKERS
    return {"tickers": TICKERS}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = load_best_model(request.ticker)

    try:
        input_df = pd.DataFrame([request.features])
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    return PredictResponse(
        ticker=request.ticker,
        prediction=prediction,
        probability_up=round(probability, 4),
        signal="BUY" if prediction == 1 else "SELL"
    )

@app.get("/")
def root():
    return {
        "message": "Indonesian Stock Prediction API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)