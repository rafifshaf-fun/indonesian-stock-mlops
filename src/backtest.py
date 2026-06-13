"""
Backtesting Engine — Walk-Forward Trading Simulation
=====================================================
Simulates trading on historical data using trained models.
Tracks P&L, Sharpe ratio, max drawdown, win rate, and more.

Usage:
    python src/backtest.py                         # All tickers
    python src/backtest.py --ticker BBCA.JK        # Single ticker
    python src/backtest.py --threshold 0.65        # Min confidence to trade
    python src/backtest.py --plot                   # Generate equity curve plots

Output: Results logged to MLflow experiment "backtest"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
from datetime import datetime

from config import (
    TICKERS, DATA_PROCESSED_CSV_PATH, MLFLOW_EXPERIMENT,
    MLFLOW_DB, TRAINING_CONFIG, get_logger,
)
from train import load_features, prepare_xy, prune_features

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow_db_path = os.path.join(os.getcwd(), MLFLOW_DB)
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
BACKTEST_EXPERIMENT = "backtest"

# ── Model cache ───────────────────────────────────────────────────────────────
_model_cache = {}  # ticker -> model

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD = 0.5   # Only trade if prob > this
DEFAULT_INITIAL_CAPITAL = 1_000_000   # Starting virtual capital (IDR)
DEFAULT_TRADE_SIZE = 0.25            # Fraction of capital per trade
DEFAULT_COST_PER_TRADE = 0.0015      # 0.15% trading cost (broker fee)

# ═══════════════════════════════════════════════════════════════════════════════
# CORE: BACKTEST A SINGLE TICKER
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_ticker(ticker: str, df_features: pd.DataFrame,
                    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                    trade_size: float = DEFAULT_TRADE_SIZE,
                    cost_per_trade: float = DEFAULT_COST_PER_TRADE,
                    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                    plot: bool = False) -> dict:
    """Walk-forward backtest for a single ticker.

    Simulates trading through the entire feature history, using the
    trained model to predict on each day and trade accordingly.

    Args:
        ticker: Stock ticker
        df_features: Full feature DataFrame (all tickers)
        confidence_threshold: Minimum confidence to enter a trade
        trade_size: Fraction of capital deployed per trade (0-1)
        cost_per_trade: Trading cost as fraction (e.g. 0.0015 = 0.15%)
        initial_capital: Starting virtual capital
        plot: Generate equity curve plot

    Returns:
        dict of backtest metrics
    """
    # ── Load model from filesystem using model index ─────────────────
    global _model_cache
    if ticker in _model_cache:
        model = _model_cache[ticker]
        loaded = True
    else:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "mlruns", "1", "models")
        index_path = os.path.join(models_dir, "model_index.json")
        model = None
        loaded = False

        # Try model index first
        if os.path.exists(index_path):
            try:
                import json
                with open(index_path) as f:
                    index_data = json.load(f)
                ticker_map = index_data.get("ticker_to_model", {})
                if ticker in ticker_map:
                    mf = ticker_map[ticker]["model_folder"]
                    artifact_dir = os.path.join(models_dir, mf, "artifacts")
                    if os.path.exists(os.path.join(artifact_dir, "MLmodel")):
                        model = mlflow.xgboost.load_model(artifact_dir)
                        logger.info("Loaded model for %s from index: %s", ticker, mf)
                        loaded = True
            except Exception as e:
                logger.warning("Index load failed for %s: %s", ticker, e)

        # Fallback: scan all models
        if not loaded and os.path.isdir(models_dir):
            for mf in sorted(os.listdir(models_dir), reverse=True):
                mlf = os.path.join(models_dir, mf, "artifacts", "MLmodel")
                if os.path.exists(mlf):
                    try:
                        model = mlflow.xgboost.load_model(os.path.join(models_dir, mf, "artifacts"))
                        logger.info("Fallback: loaded model for %s from %s", ticker, mf)
                        loaded = True
                        break
                    except Exception as e:
                        logger.debug("Failed to load %s: %s", mf, e)
                        continue

        if loaded and model is not None:
            _model_cache[ticker] = model

    if not loaded or model is None:
        logger.warning("No model found for %s — skipping", ticker)
        return None

    # ── Prepare ticker data ───────────────────────────────────────────────
    ticker_df = df_features[df_features["ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_index()

    if len(ticker_df) < TRAINING_CONFIG["min_rows"]:
        logger.warning("Skipping %s — only %d rows", ticker, len(ticker_df))
        return None

    X, y = prepare_xy(ticker_df)
    X = prune_features(X, verbose=False)

    if X.shape[0] == 0 or X.shape[1] == 0:
        return None

    # Get expected feature names from model
    if hasattr(model, "feature_names_in_"):
        trained_cols = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        trained_cols = model.get_booster().feature_names
    else:
        trained_cols = X.columns.tolist()

    # Align features
    for col in trained_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[[c for c in trained_cols if c in X.columns]]
    X = X[trained_cols]

    # ── Walk-forward simulation ───────────────────────────────────────────
    n = len(X)
    positions = np.zeros(n, dtype=int)     # 1 = invested, 0 = cash
    capital = np.full(n, np.nan)           # Total capital (cash + stock value)
    cash = np.full(n, np.nan)              # Cash held
    stock_value = np.full(n, np.nan)       # Value of held shares
    trades_taken = []                       # List of trade dicts

    current_capital = initial_capital
    current_cash = initial_capital
    current_shares = 0
    entry_price = 0.0
    in_position = False

    # Use Close price from the feature DataFrame (available in raw but not in X)
    # Add recovery of Close price
    close_col = None
    for c in ["Close", "close"]:
        if c in ticker_df.columns:
            close_col = c
            break
    if close_col is None:
        # Try to find it in any available column
        for c in ticker_df.columns:
            if "close" in c.lower():
                close_col = c
                break
    if close_col is None:
        logger.warning("No Close price column found for %s", ticker)
        return None

    prices = ticker_df[close_col].values

    for i in range(n):
        proba = model.predict_proba(X.iloc[[i]])[:, 1][0]
        predicted_signal = 1 if proba >= confidence_threshold else 0
        price = prices[i]

        if np.isnan(price) or price <= 0:
            capital[i] = current_cash + current_shares * price if in_position else current_cash
            cash[i] = current_cash
            stock_value[i] = current_shares * price
            continue

        # BUY signal
        if predicted_signal == 1 and not in_position:
            invest_amount = current_cash * trade_size
            shares_bought = int(invest_amount / price)
            cost = shares_bought * price * cost_per_trade
            if shares_bought > 0 and invest_amount > cost:
                current_shares = shares_bought
                current_cash -= shares_bought * price + cost
                in_position = True
                entry_price = price
                trades_taken.append({
                    "date": str(ticker_df.index[i].date()),
                    "type": "BUY",
                    "price": price,
                    "shares": shares_bought,
                    "cost": cost,
                    "confidence": round(proba, 4),
                })

        # SELL signal (or exit)
        elif predicted_signal == 0 and in_position:
            proceeds = current_shares * price
            cost = proceeds * cost_per_trade
            pnl = proceeds - current_shares * entry_price - cost
            current_cash += proceeds - cost
            in_position = False
            trades_taken.append({
                "date": str(ticker_df.index[i].date()),
                "type": "SELL",
                "price": price,
                "shares": current_shares,
                "cost": cost,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / (current_shares * entry_price) * 100, 2) if current_shares > 0 else 0,
                "confidence": round(proba, 4),
            })
            current_shares = 0

        capital[i] = current_cash + current_shares * price if in_position else current_cash
        cash[i] = current_cash
        stock_value[i] = current_shares * price if in_position else 0

    # Close any open position on last day
    if in_position and n > 0:
        price = prices[-1]
        proceeds = current_shares * price
        cost = proceeds * cost_per_trade
        pnl = proceeds - current_shares * entry_price - cost
        current_cash += proceeds - cost
        in_position = False
        trades_taken.append({
            "date": str(ticker_df.index[-1].date()),
            "type": "SELL (forced close)",
            "price": price,
            "shares": current_shares,
            "cost": cost,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / (current_shares * entry_price) * 100, 2) if current_shares > 0 else 0,
        })
        capital[-1] = current_cash

    final_capital = capital[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    n_trades = len([t for t in trades_taken if t["type"].startswith("SELL")])

    # ── Calculate metrics ─────────────────────────────────────────────────
    returns = pd.Series(capital).pct_change().dropna()
    sharpe_ratio = np.nan
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252))

    # Max drawdown
    peak = np.maximum.accumulate(capital)
    drawdown = (peak - capital) / peak
    max_drawdown = float(np.nanmax(drawdown)) if not np.all(np.isnan(drawdown)) else 0.0

    # Win rate
    wins = [t for t in trades_taken if t.get("pnl", 0) > 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

    # Avg profit per winning/losing trade
    avg_win = np.mean([t["pnl"] for t in trades_taken if t.get("pnl", 0) > 0]) if wins else 0.0
    avg_loss = np.mean([t["pnl"] for t in trades_taken if t.get("pnl", 0) < 0]) if n_trades - len(wins) > 0 else 0.0

    # ── Plot equity curve ─────────────────────────────────────────────────
    img_path = None
    if plot and len(capital) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(ticker_df.index, capital, label="Portfolio Value", color="navy", linewidth=2)
        axes[0].axhline(y=initial_capital, color="gray", linestyle="--", alpha=0.7, label="Initial Capital")
        axes[0].fill_between(ticker_df.index, initial_capital, capital, where=(capital >= initial_capital),
                             color="green", alpha=0.15, label="Profit")
        axes[0].fill_between(ticker_df.index, initial_capital, capital, where=(capital < initial_capital),
                             color="red", alpha=0.15, label="Loss")
        axes[0].set_ylabel("Portfolio Value (IDR)")
        axes[0].set_title(f"Backtest: {ticker}  |  Return: {total_return*100:.1f}%  |  Sharpe: {sharpe_ratio:.2f}")
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.3)

        # Mark trades on equity curve
        buy_dates = [t["date"] for t in trades_taken if t["type"] == "BUY"]
        for idx, d in enumerate(buy_dates):
            dt = pd.Timestamp(d)
            if dt in ticker_df.index:
                axes[0].scatter(dt, capital[ticker_df.index.get_loc(dt)],
                               color="green", marker="^", s=60, zorder=5)

        # Drawdown panel
        axes[1].fill_between(ticker_df.index, 0, drawdown * 100, color="red", alpha=0.3)
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_xlabel("Date")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        img_path = f"{ticker}_backtest.png"
        plt.savefig(img_path, dpi=100)
        plt.close()

    metrics = {
        "ticker": ticker,
        "initial_capital": initial_capital,
        "final_capital": round(final_capital, 2),
        "total_return": round(total_return, 4),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 4) if not np.isnan(sharpe_ratio) else 0,
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate, 4),
        "n_trades": n_trades,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "confidence_threshold": confidence_threshold,
        "trade_size": trade_size,
    }

    logger.info(
        "%s | Return: %.1f%% | Sharpe: %.2f | MaxDD: %.1f%% | Win: %.0f%% (%d trades)",
        ticker, metrics["total_return_pct"], metrics["sharpe_ratio"],
        metrics["max_drawdown_pct"], metrics["win_rate"] * 100, n_trades
    )

    return {"metrics": metrics, "trades": trades_taken, "equity_curve": capital, "plot_path": img_path}


# ═══════════════════════════════════════════════════════════════════════════════
# RUN BACKTEST ON MULTIPLE TICKERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(tickers: list = None, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 trade_size: float = DEFAULT_TRADE_SIZE, plot: bool = False) -> pd.DataFrame:
    """Run backtest on multiple tickers and log results to MLflow.

    Args:
        tickers: List of tickers (default: all TICKERS)
        confidence_threshold: Minimum confidence to trade
        trade_size: Fraction of capital per trade
        plot: Generate equity curve plots

    Returns:
        DataFrame of backtest results per ticker
    """
    if tickers is None:
        tickers = TICKERS

    df = load_features(DATA_PROCESSED_CSV_PATH)
    logger.info("Loaded features: %s", df.shape)

    mlflow.set_experiment(BACKTEST_EXPERIMENT)
    all_results = []

    for ticker in tickers:
        result = backtest_ticker(
            ticker, df,
            confidence_threshold=confidence_threshold,
            trade_size=trade_size,
            plot=plot,
        )
        if result is None:
            continue

        metrics = result["metrics"]
        trades = result["trades"]
        all_results.append(metrics)

        # Log to MLflow
        with mlflow.start_run(run_name=f"bt_{ticker}"):
            mlflow.log_params({
                "ticker": ticker,
                "confidence_threshold": confidence_threshold,
                "trade_size": trade_size,
                "initial_capital": DEFAULT_INITIAL_CAPITAL,
            })
            mlflow.log_metrics({
                "total_return": metrics["total_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown_pct"] / 100,
                "win_rate": metrics["win_rate"],
                "n_trades": metrics["n_trades"],
                "avg_win": metrics["avg_win"],
                "avg_loss": metrics["avg_loss"],
            })
            # Log trades as artifact
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_csv = f"{ticker}_trades.csv"
                trades_df.to_csv(trades_csv, index=False)
                mlflow.log_artifact(trades_csv)
                os.remove(trades_csv)

            if result["plot_path"]:
                mlflow.log_artifact(result["plot_path"])
                os.remove(result["plot_path"])

    # Summary
    results_df = pd.DataFrame(all_results)
    if len(results_df) > 0:
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 70)
        logger.info("Avg Return: %.2f%% | Avg Sharpe: %.2f | Avg MaxDD: %.2f%% | Avg Win Rate: %.1f%%",
                    results_df["total_return_pct"].mean(),
                    results_df["sharpe_ratio"].mean(),
                    results_df["max_drawdown_pct"].mean(),
                    results_df["win_rate"].mean() * 100)
        logger.info("Tickers beating buy-hold: %d/%d",
                    (results_df["total_return"] > 0).sum(), len(results_df))
        logger.info("=" * 70)

        # Log summary to MLflow
        with mlflow.start_run(run_name="summary"):
            mlflow.log_metrics({
                "avg_return": results_df["total_return"].mean(),
                "avg_sharpe": results_df["sharpe_ratio"].mean(),
                "avg_max_drawdown": results_df["max_drawdown_pct"].mean() / 100,
                "avg_win_rate": results_df["win_rate"].mean(),
                "total_trades": results_df["n_trades"].sum(),
                "n_profitable_tickers": (results_df["total_return"] > 0).sum(),
            })

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk-Forward Backtesting Engine")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker to test")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help=f"Min confidence to trade (default: {DEFAULT_CONFIDENCE_THRESHOLD})")
    parser.add_argument("--trade-size", type=float, default=DEFAULT_TRADE_SIZE,
                        help=f"Fraction of capital per trade (default: {DEFAULT_TRADE_SIZE})")
    parser.add_argument("--plot", action="store_true", help="Generate equity curve plots")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else TICKERS
    logger.info("Running backtest on %d tickers (threshold=%.2f, trade_size=%.2f)...",
                len(tickers), args.threshold, args.trade_size)

    results = run_backtest(
        tickers=tickers,
        confidence_threshold=args.threshold,
        trade_size=args.trade_size,
        plot=args.plot,
    )

    if results is not None and len(results) > 0:
        print("\nTop Performers:")
        print(results.sort_values("sharpe_ratio", ascending=False).to_string(index=False))
