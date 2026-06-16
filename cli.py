"""
Unified CLI — One entry point for all Indonesian Stock MLOps commands.

Usage:
    python cli.py predict BBCA.JK                 # Single prediction
    python cli.py predict BBCA.JK BBRI.JK --local # Local models
    python cli.py predict --all                    # All 45 tickers
    python cli.py backtest                         # Full backtest (0.50)
    python cli.py backtest --threshold 0.65        # Higher confidence
    python cli.py sentiment BBCA.JK                # News sentiment score
    python cli.py sentiment --all                  # All tickers
    python cli.py train                            # Train all models
    python cli.py train --ticker BBCA.JK --tune    # Train + tune single
    python cli.py serve                            # Start API server
    python cli.py status                           # Check API health
    python cli.py list                             # List all tickers
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
import json
import warnings
warnings.filterwarnings("ignore")


def cmd_predict(args):
    """Predict BUY/SELL signals."""
    tickers = _resolve_tickers(args)

    if args.local:
        results = _predict_local_all(tickers)
    else:
        results = _predict_api_all(tickers, args.timeout)

    _display_results(results, args.json)


def cmd_backtest(args):
    """Run walk-forward backtest."""
    print(f"Running backtest on all tickers (threshold={args.threshold})...")
    print("This may take 15-20 minutes.\n")
    os.environ["PYTHONUNBUFFERED"] = "1"
    import subprocess
    cmd = [
        sys.executable, "-u",
        os.path.join(os.path.dirname(__file__), "src", "backtest.py"),
        "--threshold", str(args.threshold),
    ]
    subprocess.run(cmd)


def cmd_sentiment(args):
    """Get news sentiment scores."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from features.fetchers import fetch_news_sentiment
    from config import TICKERS

    tickers = args.tickers if args.tickers else TICKERS
    for ticker in tickers:
        score = fetch_news_sentiment(ticker)
        sentiment = "😀 Bullish" if score > 0.15 else ("😟 Bearish" if score < -0.15 else "😐 Neutral")
        print(f"{ticker:12s}  {score:+.4f}  {sentiment}")


def cmd_train(args):
    """Train models."""
    import subprocess
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "src", "train.py")]
    if args.ticker:
        cmd.extend(["--ticker", args.ticker])
    if args.tune:
        cmd.append("--tune")
    if args.parallel:
        cmd.append("--parallel")
    subprocess.run(cmd)


def cmd_serve(args):
    """Start API server."""
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "src", "serve.py"),
    ]
    subprocess.run(cmd)


def cmd_status(args):
    """Check API health."""
    try:
        import requests
        r = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if r.status_code == 200:
            print("✅ API is RUNNING")
            r2 = requests.get("http://127.0.0.1:8000/tickers", timeout=5)
            if r2.status_code == 200:
                data = r2.json()
                print(f"   Tickers: {data.get('count', '?')}")
        else:
            print("❌ API returned status", r.status_code)
    except Exception as e:
        print(f"❌ API unreachable: {e}")


def cmd_list(args):
    """List all tickers."""
    from config import TICKERS, SECTORS
    print(f"\n{'Ticker':12s} {'Sector':20s}")
    print("-" * 35)
    for ticker in sorted(TICKERS):
        sector = "Unknown"
        for sec_name, sec_tickers in SECTORS.items():
            if ticker in sec_tickers:
                sector = sec_name
                break
        print(f"{ticker:12s} {sector:20s}")
    print(f"\nTotal: {len(TICKERS)} tickers in {len(SECTORS)} sectors")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_tickers(args) -> list:
    from config import TICKERS
    if args.all:
        return TICKERS
    if not args.tickers:
        print("No tickers specified. Use --all for all tickers or provide ticker symbols.")
        sys.exit(1)
    return args.tickers


def _predict_api_all(tickers: list, timeout: int) -> list:
    try:
        import requests
    except ImportError:
        return [{"ticker": t, "error": "requests not installed"} for t in tickers]

    results = []
    for t in tickers:
        try:
            r = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"ticker": t},
                timeout=timeout,
            )
            if r.status_code == 200:
                results.append(r.json())
            else:
                results.append({"ticker": t, "error": f"HTTP {r.status_code}"})
        except Exception as e:
            results.append({"ticker": t, "error": str(e)})
    return results


def _predict_local_all(tickers: list) -> list:
    import mlflow.xgboost
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from ta.utils import dropna
    from features import (
        compute_ta_features, compute_custom_features,
        compute_enhanced_mas, compute_ict_features,
        inject_macro_features, fetch_fundamentals,
        fetch_usdidr, fetch_fred_macro, fetch_bi_rate,
    )
    from config import FEATURE_FLAGS

    results = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="250d", auto_adjust=True, progress=False)
            if df.empty:
                results.append({"ticker": ticker, "error": "No Yahoo Finance data"})
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = dropna(df)
            if len(df) < 50:
                results.append({"ticker": ticker, "error": "Insufficient data"})
                continue

            df = compute_ta_features(df)
            df = compute_custom_features(df)
            df = compute_enhanced_mas(df)
            df = compute_ict_features(df)

            start = str(df.index[0].date())
            end = str(df.index[-1].date())
            usdidr = fetch_usdidr(start, end)
            fred_macro = fetch_fred_macro(start, end)
            bi_rate = fetch_bi_rate(start, end)
            fundamentals = fetch_fundamentals(ticker)
            df = inject_macro_features(df, fundamentals, usdidr, fred_macro, bi_rate)
            df["google_trend"] = 0.0

            if FEATURE_FLAGS.get("news_sentiment", True):
                from features.fetchers import fetch_news_sentiment
                df["news_sentiment"] = fetch_news_sentiment(ticker)
            else:
                df["news_sentiment"] = 0.0

            # Load model via index
            models_dir = os.path.join(os.path.dirname(__file__), "mlruns", "1", "models")
            index_path = os.path.join(models_dir, "model_index.json")
            model = None
            if os.path.exists(index_path):
                with open(index_path) as f:
                    ticker_map = json.load(f).get("ticker_to_model", {})
                if ticker in ticker_map:
                    mf = ticker_map[ticker]["model_folder"]
                    artifact_dir = os.path.join(models_dir, mf, "artifacts")
                    if os.path.exists(os.path.join(artifact_dir, "MLmodel")):
                        model = mlflow.xgboost.load_model(artifact_dir)
            if model is None:
                results.append({"ticker": ticker, "error": "No model found"})
                continue

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
            results.append({
                "ticker": ticker,
                "prediction": prediction,
                "probability_up": round(probability, 4),
                "signal": signal,
            })
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})
    return results


def _display_results(results: list, as_json: bool = False):
    if as_json:
        print(json.dumps(results, indent=2))
        return

    print(f"\n{'Ticker':12s} {'Signal':8s} {'Model':>7s} {'Sent':>7s}  {'Adj':>7s}  {'Final':8s}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['ticker']:12s} {'ERROR':8s} {'—':>7s} {'—':>7s}  {'—':>7s}  {r['error'][:40]}")
        else:
            model_signal = r.get("signal", "?")
            prob = r.get("probability_up", 0)
            sent = r.get("sentiment_score", 0)
            adj = r.get("probability_adjusted", prob)
            final_signal = r.get("signal_adjusted", model_signal)
            me = "🟢" if model_signal == "BUY" else "🔴"
            fe = "🟢" if final_signal == "BUY" else "🔴"
            print(f"{r['ticker']:12s} {me + ' ' + model_signal:8s} {prob:>6.1%}  {sent:>+5.2f}  {adj:>6.1%}  {fe + ' ' + final_signal:8s}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Indonesian Stock MLOps CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # predict
    p_pred = sub.add_parser("predict", help="Predict BUY/SELL signals")
    p_pred.add_argument("tickers", nargs="*", help="Ticker symbols")
    p_pred.add_argument("--local", action="store_true", help="Use local models (no API)")
    p_pred.add_argument("--json", action="store_true", help="JSON output")
    p_pred.add_argument("--all", action="store_true", help="All 45 tickers")
    p_pred.add_argument("--timeout", type=int, default=120, help="API timeout (seconds)")

    # backtest
    p_bt = sub.add_parser("backtest", help="Run walk-forward backtest")
    p_bt.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold (default: 0.50)")
    p_bt.add_argument("--ticker", type=str, help="Single ticker only")

    # sentiment
    p_sent = sub.add_parser("sentiment", help="Get news sentiment scores")
    p_sent.add_argument("tickers", nargs="*", help="Ticker symbols (omit for all)")

    # train
    p_train = sub.add_parser("train", help="Train models")
    p_train.add_argument("--ticker", type=str, help="Single ticker")
    p_train.add_argument("--tune", action="store_true", help="Enable Optuna hyperparameter tuning")
    p_train.add_argument("--tune-all", action="store_true", help="Tune all 45 tickers (slow!)")
    p_train.add_argument("--parallel", action="store_true", help="Parallel training (~4x faster)")

    # serve
    p_serve = sub.add_parser("serve", help="Start API server")

    # status
    p_status = sub.add_parser("status", help="Check API server health")

    # list
    p_list = sub.add_parser("list", help="List all tickers by sector")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "predict": cmd_predict,
        "backtest": cmd_backtest,
        "sentiment": cmd_sentiment,
        "train": cmd_train,
        "serve": cmd_serve,
        "status": cmd_status,
        "list": cmd_list,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
