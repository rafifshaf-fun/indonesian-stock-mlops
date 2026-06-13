"""
Prediction CLI — Predict signals from the command line.

Usage:
    python predict.py BBCA.JK                          # Via API (needs server running)
    python predict.py BBCA.JK --local                  # Via local model (no server)
    python predict.py BBCA.JK BBRI.JK TLKM.JK          # Multiple tickers
    python predict.py --list                           # List all tickers
    python predict.py BBCA.JK --json                   # JSON output (for scripts)
    python predict.py --all                            # All 45 tickers
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from config import TICKERS

API_URL = "http://127.0.0.1:8000"


def predict_via_api(ticker: str, timeout: int = 120) -> dict:
    """Predict via the running API server."""
    if not HAS_REQUESTS:
        return {"ticker": ticker, "error": "requests library not available"}

    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"ticker": ticker},
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()
        else:
            return {"ticker": ticker, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def predict_local(ticker: str) -> dict:
    """Predict using local model without API server."""
    try:
        import mlflow.xgboost
        import pandas as pd
        import numpy as np
        import yfinance as yf
        from ta import add_all_ta_features
        from ta.utils import dropna
        from features import (
            compute_ta_features, compute_custom_features,
            compute_enhanced_mas, compute_ict_features,
            inject_macro_features, fetch_fundamentals,
            fetch_usdidr, fetch_fred_macro, fetch_bi_rate,
        )
    except ImportError as e:
        return {"ticker": ticker, "error": f"Import error: {e}. Try without --local (uses API)."}

    try:
        # Download data
        df = yf.download(ticker, period="250d", auto_adjust=True, progress=False)
        if df.empty:
            return {"ticker": ticker, "error": "No data from Yahoo Finance"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = dropna(df)
        if len(df) < 50:
            return {"ticker": ticker, "error": "Not enough data after cleaning"}

        # Build features
        df = compute_ta_features(df)
        df = compute_custom_features(df)
        df = compute_enhanced_mas(df)
        df = compute_ict_features(df)

        # Macro
        start = str(df.index[0].date())
        end = str(df.index[-1].date())
        usdidr = fetch_usdidr(start, end)
        fred_macro = fetch_fred_macro(start, end)
        bi_rate = fetch_bi_rate(start, end)
        fundamentals = fetch_fundamentals(ticker)
        df = inject_macro_features(df, fundamentals, usdidr, fred_macro, bi_rate)
        df["google_trend"] = 0.0

        # Load model from filesystem using model index
        models_dir = os.path.join(os.path.dirname(__file__), "..", "mlruns", "1", "models")
        index_path = os.path.join(models_dir, "model_index.json")
        model = None
        if os.path.exists(index_path):
            try:
                with open(index_path) as f:
                    idx_data = json.load(f)
                ticker_map = idx_data.get("ticker_to_model", {})
                if ticker in ticker_map:
                    mf = ticker_map[ticker]["model_folder"]
                    artifact_dir = os.path.join(models_dir, mf, "artifacts")
                    if os.path.exists(os.path.join(artifact_dir, "MLmodel")):
                        model = mlflow.xgboost.load_model(artifact_dir)
            except Exception:
                pass
        if model is None:
            return {"ticker": ticker, "error": "No model found for this ticker. Run build_model_index.py first."}

        # Align features
        if hasattr(model, "feature_names_in_"):
            trained_cols = list(model.feature_names_in_)
        else:
            trained_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        numeric_df = df.select_dtypes(include=[np.number])
        latest = numeric_df.iloc[[-1]].copy()

        for col in trained_cols:
            if col not in latest.columns:
                latest[col] = np.nan
        latest_df = pd.DataFrame([latest[trained_cols].iloc[0]])

        prediction = int(model.predict(latest_df)[0])
        probability = float(model.predict_proba(latest_df)[0][1])
        signal = "BUY" if prediction == 1 else "SELL"

        return {
            "ticker": ticker,
            "prediction": prediction,
            "probability_up": round(probability, 4),
            "signal": signal,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Predict BUY/SELL signals for Indonesian stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py BBCA.JK              Single ticker via API
  python predict.py BBCA.JK BBRI.JK      Multiple tickers
  python predict.py --local BBCA.JK      Use local model (no API server)
  python predict.py --all                All 45 tickers
  python predict.py BBCA.JK --json       JSON output
        """
    )
    parser.add_argument("tickers", nargs="*", help="Ticker symbols (e.g. BBCA.JK)")
    parser.add_argument("--local", action="store_true", help="Use local model (no API server)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--all", action="store_true", help="Predict all 45 tickers")
    parser.add_argument("--timeout", type=int, default=120, help="API timeout in seconds")

    args = parser.parse_args()

    if args.all:
        tickers = TICKERS
    elif args.tickers:
        tickers = args.tickers
    else:
        parser.print_help()
        sys.exit(1)

    results = []
    for i, ticker in enumerate(tickers):
        if args.local:
            result = predict_local(ticker)
        else:
            result = predict_via_api(ticker, timeout=args.timeout)
        results.append(result)

        if not args.json:
            if "error" in result:
                print(f"[{i+1}/{len(tickers)}] {ticker}: ERROR - {result['error']}")
            else:
                arrow = "\U0001f7e2" if result["signal"] == "BUY" else "\U0001f534"
                print(f"[{i+1}/{len(tickers)}] {arrow} {ticker}: {result['signal']:4s} ({result['probability_up']*100:.1f}%)")

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
