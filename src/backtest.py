"""
Backtesting Engine — Walk-Forward Trading Simulation
=====================================================
Simulates trading on historical data using trained models.
Tracks P&L, Sharpe ratio, max drawdown, win rate, and more.

Usage:
    python src/backtest.py                         # All tickers, production fleet
    python src/backtest.py --ticker BBCA.JK        # Single ticker
    python src/backtest.py --ticker BBCA.JK --ticker BBRI.JK   # Two tickers
    python src/backtest.py --threshold 0.65        # Min confidence to trade
    python src/backtest.py --slippage-bps 10       # 10 bps slippage per side
    python src/backtest.py --start 2024-01-01 --end 2024-12-31   # Bounded window
    python src/backtest.py --model-root mlruns/holdout-2025-06-12/models \
        --start 2025-06-16 --end 2026-06-12        # Out-of-sample holdout fleet
    python src/backtest.py --plot                   # Generate equity curve plots

The run reads ``data/processed/features.parquet`` — the full-mode artefact
``build_feature_set(mode="full")`` writes and the one the fleet was trained on —
not the CI-mode CSV, which omits three feature families and cannot satisfy the
models' persisted contracts.  Each ticker is aligned to its model's own
``feature_schema.json`` with no re-pruning: the contract is the model's, not the
backtest's.

**Which fleet is measured decides whether the number is evidence.**  With no
``--model-root`` the run reads the production fleet at ``mlruns/1/models``,
which was fitted on the full history up to the end of the artefact — so every
row it predicts on is a row the model saw in training and the figure is
**in-sample**: a computability check, not evidence of return.  ``--model-root``
points the run at a fleet trained by ``train.py --as-of <cutoff>``, and an
evaluation window that starts after that cutoff is the out-of-sample
measurement.  The store the run used is logged with every run.

The run names the universe it measured.  The 100-row minimum is applied to the
*evaluated* rows, so a narrow window can drop a ticker; every skip is reported
with its reason and the aggregate is stated to be over the tickers that ran.

Output: Results logged to MLflow experiment "backtest"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
from datetime import datetime

from config import (
    TICKERS, DATA_PROCESSED_PATH, MLFLOW_EXPERIMENT,
    MLFLOW_DB, TRAINING_CONFIG, get_logger,
)
from features.schema import FeatureSchemaError, align_to_schema
from model_store import (
    ModelNotFound, index_path, load_model_for, load_model_index, model_dir,
    set_model_root,
)
from train import load_features, prepare_xy

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow_db_path = os.path.join(os.getcwd(), MLFLOW_DB)
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
BACKTEST_EXPERIMENT = "backtest"

# ── Model cache ───────────────────────────────────────────────────────────────
_model_cache = {}  # ticker -> (model, feature_columns)

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD = 0.5   # Only trade if prob > this
DEFAULT_INITIAL_CAPITAL = 1_000_000   # Starting virtual capital (IDR)
DEFAULT_TRADE_SIZE = 0.25            # Fraction of capital per trade

# ── Transaction costs ─────────────────────────────────────────────────────────
# IDX is not symmetric.  Brokerage is paid on both legs, and the sell leg also
# carries the 0.1% final sales tax, so a round trip costs strictly more on the
# way out than on the way in.  A single flat fee charged at all three sites
# understated the exit by 0.1% of the notional; on a strategy that trades often
# that gap decides whether the strategy is viable at all (Phase 3).
DEFAULT_BUY_COST = 0.0015            # Brokerage on the buy leg
DEFAULT_SELL_COST = 0.0025           # Brokerage + 0.1% final sales tax on the sell leg


def costs_for(value: float, side: str, slippage_bps: int = 0) -> float:
    """Transaction cost in IDR for trading ``value`` IDR on one side.

    Args:
        value: Notional traded on this leg.  The buy site passes
            ``shares * price``, the two sell sites pass ``proceeds``.
        side: ``"buy"`` or ``"sell"`` — the sell leg carries the 0.1% tax.
        slippage_bps: Slippage in basis points, applied per side: 10 bps adds
            0.1% to the cost of this leg, and a round trip therefore pays it
            twice.  One basis point is one hundredth of a percent.

    Returns:
        The cost as a positive IDR amount.

    Raises:
        ValueError: If ``side`` is neither ``"buy"`` nor ``"sell"``.  A typo
            would otherwise be charged the buy rate silently.
    """
    if side == "buy":
        rate = DEFAULT_BUY_COST
    elif side == "sell":
        rate = DEFAULT_SELL_COST
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    return value * (rate + slippage_bps / 10_000)

# ═══════════════════════════════════════════════════════════════════════════════
# BOUNDED EVALUATION WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

def _window_bound(value, option: str):
    """Parse one ``--start``/``--end`` value into a timestamp, or ``None``.

    ``pd.Timestamp`` accepts a date, a datetime or a ``Timestamp``; anything it
    cannot parse is re-raised as a ``ValueError`` naming the option, so the
    refusal says which flag was wrong rather than "could not convert string".
    """
    if value is None:
        return None
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{option} must be a date such as 2024-01-01, got {value!r}"
        ) from exc


def filter_eval_window(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Restrict ``df`` to the rows dated within ``[start, end]``, inclusive.

    This is the seam a holdout needs: the caller names the first and last day to
    evaluate and only those rows reach the simulation.  Nothing is computed over
    the full history first, so a window cannot leak rows from outside itself
    into the run (Task 3.3 builds the cutoff on it).

    Args:
        df: The feature frame, indexed by date.  It may carry many tickers, in
            which case the index repeats — the mask is positional, so repeated
            dates are filtered identically.
        start: First date to evaluate, inclusive.  ``None`` leaves the beginning
            unbounded.
        end: Last date to evaluate, inclusive.  ``None`` leaves the end
            unbounded.  A date-only ``end`` includes that whole day, because
            this artefact is daily and its rows sit at midnight.

    Returns:
        The selected rows, in their original order.

    Raises:
        ValueError: a bound cannot be parsed, or the window selects **no rows**.
            An empty window is refused rather than simulated: an empty result is
            indistinguishable from a strategy that never traded, which is
            exactly the silence this backtest has to end.  The message names the
            requested window and the range the frame does cover.
    """
    start_at = _window_bound(start, "start")
    end_at = _window_bound(end, "end")
    if start_at is None and end_at is None:
        return df

    dates = df.index
    mask = np.ones(len(df), dtype=bool)
    if start_at is not None:
        mask &= dates >= start_at
    if end_at is not None:
        mask &= dates <= end_at

    bounded = df.loc[mask]
    named = ", ".join(
        f"{option}={value.date()}"
        for option, value in (("start", start_at), ("end", end_at))
        if value is not None
    )
    if bounded.empty:
        covered = (
            "the frame is empty"
            if len(df) == 0
            else f"the frame covers {df.index.min()} to {df.index.max()}"
        )
        raise ValueError(
            f"the requested evaluation window ({named}) selects no rows: "
            f"{covered} ({len(df)} row(s)) — refusing to simulate zero trades "
            f"on an empty window"
        )

    logger.info(
        "Evaluation window (%s): %d of %d row(s) kept", named, len(bounded), len(df)
    )
    return bounded


def load_eval_frame(start=None, end=None) -> pd.DataFrame:
    """Load the feature artefact **training** used, bounded to the window.

    Training reads ``config.DATA_PROCESSED_PATH``: the full-mode parquet that
    ``build_feature_set(mode="full")`` writes, carrying 193 columns including the
    ``vp_*``, news-sentiment and Google-Trends families.  The backtest used to
    read ``config.DATA_PROCESSED_CSV_PATH`` — the CI-mode artefact
    ``build_feature_set(mode="ci")`` writes, which omits three families (174
    columns) — and 12 of BBCA.JK's 60 contract columns were in that difference.
    A backtest that cannot build its model's own features measures nothing, so
    it reads the artefact the fleet was trained on.

    Args:
        start: First date to evaluate, inclusive (see :func:`filter_eval_window`).
        end: Last date to evaluate, inclusive.

    Returns:
        The feature frame, restricted to the requested window.

    Raises:
        ValueError: the window selects no rows — refused, not simulated.
        FileNotFoundError: the artefact is missing (``load_features``' own).
    """
    df = load_features(DATA_PROCESSED_PATH)
    logger.info("Loaded features: %s from %s", df.shape, DATA_PROCESSED_PATH)
    return filter_eval_window(df, start, end)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: BACKTEST A SINGLE TICKER
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_ticker(ticker: str, df_features: pd.DataFrame,
                    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                    trade_size: float = DEFAULT_TRADE_SIZE,
                    slippage_bps: int = 0,
                    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                    plot: bool = False,
                    skip_reasons: dict | None = None) -> dict | None:
    """Walk-forward backtest for a single ticker.

    Simulates trading through every row of the (already bounded) feature frame,
    using the trained model to predict on each day and trade accordingly.

    Args:
        ticker: Stock ticker
        df_features: Full feature DataFrame (all tickers)
        confidence_threshold: Minimum confidence to enter a trade
        trade_size: Fraction of capital deployed per trade (0-1)
        slippage_bps: Slippage in basis points, charged per side on top of the
            side's rate (see :func:`costs_for`)
        initial_capital: Starting virtual capital
        plot: Generate equity curve plot
        skip_reasons: Optional mapping the caller passes so a skipped ticker is
            named in the run's report with its reason.  ``None`` means the
            reason is only logged.  Every ``return None`` path records one, so
            the run can tell a shrunken universe from a flat one (R3).

    Returns:
        dict of backtest metrics, or ``None`` when the ticker cannot be
        measured (no model, too few rows in the window, an unsatisfiable
        feature contract, or no price) — in which case the reason is logged and,
        when ``skip_reasons`` is given, recorded in it.
    """
    reasons = {} if skip_reasons is None else skip_reasons

    def _skip(reason: str):
        """Record and return ``None``: the one shape of "not measured"."""
        reasons[ticker] = reason
        return None

    # ── Load the model through the one selector (Task 1.6) ────────────
    # model_store reads the same model_index.json the API reads, so the
    # backtester and the API resolve a ticker to the same folder by
    # construction (F9).  The old scan of sorted(os.listdir(...), reverse=True)
    # is gone: it answered an unmapped ticker with the alphabetically last
    # folder's model, which is not the folder the API would have chosen.
    #
    # The store itself is whatever model_store.model_dir() resolves: the
    # production fleet by default, or a holdout fleet when the run set the
    # override (Task 3.3).
    global _model_cache
    if ticker not in _model_cache:
        try:
            _model_cache[ticker] = load_model_for(ticker)
        except ModelNotFound as e:
            logger.warning("No model for %s — skipping (%s)", ticker, e)
            return _skip(f"no model is mapped in the model store ({e})")
    model, trained_cols = _model_cache[ticker]

    # ── Prepare ticker data ───────────────────────────────────────────────
    ticker_df = df_features[df_features["ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_index()

    if len(ticker_df) < TRAINING_CONFIG["min_rows"]:
        logger.warning("Skipping %s — only %d rows", ticker, len(ticker_df))
        return _skip(
            f"only {len(ticker_df)} row(s) in the evaluation window "
            f"(minimum {TRAINING_CONFIG['min_rows']})"
        )

    # ── The model's own contract, and nothing else ──────────────────────
    # Training computed pruning ONCE over the whole panel and persisted the
    # result as this model's ``feature_schema.json``.  The call that stood here
    # re-ran ``prune_features`` on a *single ticker's* frame, which dropped 10-17
    # of the 60 contracted columns — so ``align_to_schema`` raised and the ticker
    # was skipped.  It also divided by zero on this ticker's all-NaN columns
    # (``debt_to_equity`` for a bank), a warning that is an error under the
    # suite's policy.  The contract is the model's, not the backtest's: prune
    # nothing, align, and let the schema say what the model needs.
    X, _ = prepare_xy(ticker_df)

    if X.shape[0] == 0 or X.shape[1] == 0:
        return _skip("the ticker's frame has no feature columns after alignment")

    # Align features to the model's own contract.  align_to_schema raises for a
    # column the model expects and this frame cannot build — the loop that used
    # to insert NaN for exactly those columns is gone, because a padded column
    # is a feature the model never saw during training.  A ticker whose model
    # cannot be met is skipped with a warning naming it, rather than aborting
    # the other 44: the fleet is made consistent by the retrain, not by
    # weakening the contract here.
    try:
        X = align_to_schema(X, trained_cols)
    except FeatureSchemaError as e:
        logger.warning("Skipping %s — its model's feature contract cannot be met: %s",
                       ticker, e)
        return _skip(f"its model's feature contract cannot be met: {e}")

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
        return _skip("no Close price column")

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
            # Entry is a buy: brokerage, plus any slippage, against the notional.
            cost = costs_for(shares_bought * price, side="buy",
                             slippage_bps=slippage_bps)
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
            # Exit is a sell: brokerage plus the final sales tax, and slippage.
            cost = costs_for(proceeds, side="sell", slippage_bps=slippage_bps)
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
        # The forced close is a sell, and pays the sell side's costs like any
        # other exit — including the final sales tax it had been spared.
        cost = costs_for(proceeds, side="sell", slippage_bps=slippage_bps)
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

    # ── Buy-and-hold, over the same window ────────────────────────────────
    # R4: the summary line used to call ``(total_return > 0).sum()`` "tickers
    # beating buy-hold", which is a count of profitable tickers, not a
    # benchmark.  This is the comparison that label claimed to be: the
    # close-to-close price change of the window, 100% invested and before
    # costs.  It is deliberately not adjusted for the strategy's 25% position
    # size — the strategy's return is what it is, and the benchmark is stated.
    valid_prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(valid_prices) >= 2:
        buy_hold_return = float(valid_prices[-1] / valid_prices[0] - 1.0)
    else:
        buy_hold_return = float("nan")
    beats_buy_hold = bool(
        np.isfinite(buy_hold_return) and total_return > buy_hold_return
    )

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
        "slippage_bps": slippage_bps,
        # Every leg's cost, so a run can be compared across slippage levels by
        # what it actually paid rather than by the return alone.
        "total_costs": round(sum(t["cost"] for t in trades_taken), 2),
        # R4: the benchmark, and whether the strategy beat it — the comparison
        # the old "tickers beating buy-hold" line only appeared to make.
        "buy_hold_return_pct": (
            round(buy_hold_return * 100, 2) if np.isfinite(buy_hold_return) else np.nan
        ),
        "beats_buy_hold": beats_buy_hold,
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
                 trade_size: float = DEFAULT_TRADE_SIZE, slippage_bps: int = 0,
                 plot: bool = False, start=None, end=None,
                 model_root=None) -> pd.DataFrame:
    """Run backtest on multiple tickers and log results to MLflow.

    Args:
        tickers: List of tickers (default: all TICKERS)
        confidence_threshold: Minimum confidence to trade
        trade_size: Fraction of capital per trade
        slippage_bps: Slippage in basis points, charged per side
        plot: Generate equity curve plots
        start: First date to evaluate, inclusive (``YYYY-MM-DD``).  ``None``
            starts at the beginning of the feature history.
        end: Last date to evaluate, inclusive.
        model_root: Model store to resolve tickers through.  ``None`` is the
            production fleet (``mlruns/1/models``) — an in-sample run; a
            directory produced by ``train.py --as-of`` makes the run
            out-of-sample.  Selecting it here keeps the choice with the run
            rather than with a shell variable, and clears the per-process model
            cache so a second run in one process cannot keep the first fleet's
            models.

    Returns:
        DataFrame of backtest results per ticker, with the run's provenance in
        ``.attrs`` (``model_store``, ``eval_window``, ``skipped``).  Empty when
        no ticker could be backtested; the summary names every skipped ticker
        and why, so an empty result is never mistaken for a flat one.

    Raises:
        ValueError: the requested window contains no rows (see
            :func:`filter_eval_window`).
    """
    if tickers is None:
        tickers = TICKERS

    if model_root is not None:
        set_model_root(model_root)
        _model_cache.clear()

    # Which fleet produced this number is part of the result, not an ambient
    # detail: the same window reads as +X% in-sample and Y% out-of-sample, and a
    # report that cannot say which store ran is a report that cannot be
    # trusted.  The mapped-ticker count also catches an override that names an
    # empty or missing store before any simulation starts.
    store = model_dir()
    logger.info(
        "Model store: %s (%d ticker(s) mapped in %s)",
        store, len(load_model_index()), index_path(),
    )

    df = load_eval_frame(start=start, end=end)

    mlflow.set_experiment(BACKTEST_EXPERIMENT)
    all_results = []
    reasons: dict[str, str] = {}

    for ticker in tickers:
        result = backtest_ticker(
            ticker, df,
            confidence_threshold=confidence_threshold,
            trade_size=trade_size,
            slippage_bps=slippage_bps,
            plot=plot,
            skip_reasons=reasons,
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
                "slippage_bps": slippage_bps,
                # The window is part of what the run measured, so a logged run
                # can be reproduced without guessing which rows it saw.
                "eval_start": "history" if start is None else str(start),
                "eval_end": "history" if end is None else str(end),
                # ... and so is the fleet.  The production store means the
                # figure is in-sample; a holdout root means it is not.
                "model_store": str(store),
            })
            to_log = {
                "total_return": metrics["total_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown_pct"] / 100,
                "win_rate": metrics["win_rate"],
                "n_trades": metrics["n_trades"],
                "avg_win": metrics["avg_win"],
                "avg_loss": metrics["avg_loss"],
                "total_costs": metrics["total_costs"],
            }
            # A ticker whose prices were all unusable has no benchmark; MLflow
            # metrics cannot carry a NaN, and a fabricated 0.0 would read as a
            # flat market.
            if np.isfinite(metrics["buy_hold_return_pct"]):
                to_log["buy_hold_return"] = metrics["buy_hold_return_pct"] / 100
                to_log["beats_buy_hold"] = float(metrics["beats_buy_hold"])
            mlflow.log_metrics(to_log)
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
    measured = set(results_df["ticker"]) if "ticker" in results_df.columns else set()
    # R3: the row guard applies to the *windowed* rows, so a narrow window can
    # silently shrink the evaluated universe.  Every skip is named with its
    # reason, and the aggregate below says which universe it is over.
    skipped = {ticker: reason for ticker, reason in reasons.items() if ticker not in measured}
    for ticker in tickers:
        if ticker not in measured and ticker not in skipped:
            skipped[ticker] = "skipped without a recorded reason"

    results_df.attrs["model_store"] = str(store)
    results_df.attrs["eval_window"] = (
        "history" if start is None else str(start),
        "history" if end is None else str(end),
    )
    results_df.attrs["skipped"] = skipped

    if len(results_df) > 0:
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 70)
        logger.info("Model store: %s", store)
        # A run and a skip are different answers, and "0 of 45 ran" used to look
        # exactly like "45 ran and the strategy did nothing".  The counts are
        # logged before any average so the number below is read against them.
        logger.info("Tickers run: %d/%d (%d skipped)", len(results_df), len(tickers), len(skipped))
        logger.info(
            "Measured universe: %d of %d requested ticker(s) — every aggregate "
            "below is over the tickers that ran, not the whole request",
            len(results_df), len(tickers),
        )
        if skipped:
            logger.warning(
                "Evaluation universe shrank: %d of %d requested ticker(s) could "
                "not be measured — the aggregate below is over %d, not %d",
                len(skipped), len(tickers), len(results_df), len(tickers),
            )
            for ticker, reason in skipped.items():
                logger.warning("  skipped %s — %s", ticker, reason)
        logger.info("Avg Return: %.2f%% | Avg Sharpe: %.2f | Avg MaxDD: %.2f%% | Avg Win Rate: %.1f%%",
                    results_df["total_return_pct"].mean(),
                    results_df["sharpe_ratio"].mean(),
                    results_df["max_drawdown_pct"].mean(),
                    results_df["win_rate"].mean() * 100)
        # R4: the old line counted ``(total_return > 0).sum()`` — profitable
        # tickers, not a benchmark.  This compares each ticker's return to the
        # close-to-close price change of the same window (100% invested, before
        # costs), which is what the label says.
        logger.info(
            "Tickers beating buy-hold: %d/%d (strategy total return vs "
            "close-to-close buy-and-hold over the same window, 100%% invested, "
            "before costs)",
            int(results_df["beats_buy_hold"].sum()), len(results_df),
        )
        logger.info("=" * 70)

        # Log summary to MLflow
        with mlflow.start_run(run_name="summary"):
            summary = {
                "avg_return": results_df["total_return"].mean(),
                "avg_sharpe": results_df["sharpe_ratio"].mean(),
                "avg_max_drawdown": results_df["max_drawdown_pct"].mean() / 100,
                "avg_win_rate": results_df["win_rate"].mean(),
                "total_trades": results_df["n_trades"].sum(),
                "n_profitable_tickers": (results_df["total_return"] > 0).sum(),
                "n_beating_buy_hold": int(results_df["beats_buy_hold"].sum()),
                "n_measured": len(results_df),
                "n_requested": len(tickers),
                "n_skipped": len(skipped),
            }
            average_benchmark = results_df["buy_hold_return_pct"].mean()
            if np.isfinite(average_benchmark):
                summary["avg_buy_hold_return"] = average_benchmark / 100
            mlflow.log_metrics(summary)

    else:
        # 0 of 45 run is not a flat strategy: it is no measurement at all, and
        # it must not leave the process looking successful.
        logger.warning(
            "No ticker produced a backtest: 0 of %d ran (%d skipped) — no trade "
            "was simulated, so there is no return to report",
            len(tickers), len(skipped),
        )
        for ticker, reason in skipped.items():
            logger.warning("  skipped %s — %s", ticker, reason)

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk-Forward Backtesting Engine")
    parser.add_argument("--ticker", type=str, action="append", default=None,
                        help="Ticker to test; repeat the flag for several (default: all tickers)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help=f"Min confidence to trade (default: {DEFAULT_CONFIDENCE_THRESHOLD})")
    parser.add_argument("--trade-size", type=float, default=DEFAULT_TRADE_SIZE,
                        help=f"Fraction of capital per trade (default: {DEFAULT_TRADE_SIZE})")
    parser.add_argument("--slippage-bps", type=int, default=0,
                        help="Slippage in basis points, charged per side (default: 0)")
    parser.add_argument("--start", metavar="YYYY-MM-DD", default=None,
                        help="First date to evaluate, inclusive (default: start of the feature history)")
    parser.add_argument("--end", metavar="YYYY-MM-DD", default=None,
                        help="Last date to evaluate, inclusive (default: end of the feature history)")
    parser.add_argument("--model-root", metavar="DIR", default=None,
                        help="Model store to read (default: mlruns/1/models, the "
                             "production fleet — an in-sample run). Point it at a "
                             "fleet trained with train.py --as-of and start the "
                             "window after that cutoff for an out-of-sample run")
    parser.add_argument("--plot", action="store_true", help="Generate equity curve plots")
    args = parser.parse_args()

    tickers = args.ticker or TICKERS
    logger.info("Running backtest on %d tickers (threshold=%.2f, trade_size=%.2f, slippage=%d bps, "
                "window=%s..%s, model_root=%s)...",
                len(tickers), args.threshold, args.trade_size, args.slippage_bps,
                args.start or "history", args.end or "history",
                args.model_root or "default (production)")

    try:
        results = run_backtest(
            tickers=tickers,
            confidence_threshold=args.threshold,
            trade_size=args.trade_size,
            slippage_bps=args.slippage_bps,
            plot=args.plot,
            start=args.start,
            end=args.end,
            model_root=args.model_root,
        )
    except ValueError as e:
        # A window with no rows is a refused request, not a failed run: say so
        # and exit non-zero, with no traceback and no summary of zero trades.
        parser.error(str(e))

    if results is not None and len(results) > 0:
        print("\nTop Performers:")
        print(results.sort_values("sharpe_ratio", ascending=False).to_string(index=False))
