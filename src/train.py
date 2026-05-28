import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Docker/CI
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import warnings

from config import (
    TICKERS, DATA_PROCESSED_PATH, DATA_PROCESSED_CSV_PATH,
    MLFLOW_EXPERIMENT, MLFLOW_DB, TRAINING_CONFIG, get_logger,
)

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow_db_path = os.path.join(os.getcwd(), MLFLOW_DB)
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

FEATURES_PATH = DATA_PROCESSED_PATH  # Default: Parquet
MIN_ROWS = TRAINING_CONFIG["min_rows"]
IS_CI = bool(os.environ.get("CI"))

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE PRUNING
# ═══════════════════════════════════════════════════════════════════════════════

def prune_features(X: pd.DataFrame, corr_threshold: float = 0.95,
                   var_threshold: float = 0.001, verbose: bool = True) -> pd.DataFrame:
    """Remove redundant and low-variance features.

    1. Low-variance filter: drop features with near-zero variance after scaling
    2. Correlation filter: drop one of each highly correlated pair

    Args:
        X: Feature DataFrame
        corr_threshold: Pearson correlation threshold (drop if > this)
        var_threshold: Minimum variance threshold
        verbose: Log pruning summary

    Returns:
        Pruned feature DataFrame.
    """
    original_cols = X.columns.tolist()
    n_original = len(original_cols)

    # 1. Low-variance filter
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    variances = X_scaled.var()
    low_var_cols = variances[variances < var_threshold].index.tolist()
    X = X.drop(columns=low_var_cols, errors="ignore")

    # 2. Correlation filter
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper_tri.columns:
        high_corr = upper_tri[col][upper_tri[col] > corr_threshold].index.tolist()
        for hc in high_corr:
            # Keep the one with higher variance
            if hc not in to_drop and col not in to_drop:
                var_col = X[col].var()
                var_hc = X[hc].var()
                to_drop.add(col if var_col <= var_hc else hc)

    X = X.drop(columns=list(to_drop), errors="ignore")

    n_remaining = X.shape[1]
    if verbose:
        logger.info("Feature pruning: %d → %d features (dropped %d low-var, %d correlated)",
                    n_original, n_remaining, len(low_var_cols), len(to_drop))

    return X

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_features(path: str) -> pd.DataFrame:
    """Load feature DataFrame from Parquet or CSV."""
    if path.endswith(".parquet") and os.path.exists(path):
        return pd.read_parquet(path)
    elif os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    # Fallback: try CSV if parquet not found
    csv_path = DATA_PROCESSED_CSV_PATH
    if os.path.exists(csv_path):
        logger.warning("Parquet not found, using CSV fallback: %s", csv_path)
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    raise FileNotFoundError(f"No features found at {path} or {csv_path}")

def prepare_xy(df: pd.DataFrame):
    """Split features and target, keeping only numeric columns."""
    drop_cols = ["target", "ticker"]
    existing_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop).select_dtypes(include=[np.number])
    y = df["target"]
    return X, y

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(model, feature_names, ticker, top_n=20):
    """Generate bar chart of top features used by XGBoost."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances - {ticker}")
    plt.bar(range(top_n), importance[indices], align="center", color="teal")
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.tight_layout()

    img_path = f"{ticker}_importance.png"
    plt.savefig(img_path, dpi=100)
    plt.close()
    return img_path

def plot_confusion_matrix(y_true, y_pred, ticker):
    """Generate confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {ticker}")
    plt.colorbar()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    plt.tight_layout()

    img_path = f"{ticker}_confusion.png"
    plt.savefig(img_path, dpi=100)
    plt.close()
    return img_path

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: HYPERPARAMETER TUNING (Optuna)
# ═══════════════════════════════════════════════════════════════════════════════

def tune_hyperparameters(X, y, ticker: str, n_trials: int = 20) -> dict:
    """Tune XGBoost hyperparameters with Optuna and TimeSeriesSplit CV.

    Args:
        X: Feature matrix
        y: Target vector
        ticker: Ticker symbol (for logging)
        n_trials: Number of Optuna trials

    Returns:
        Best parameter dict.
    """
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna not installed — using default params. Run: pip install optuna")
        return TRAINING_CONFIG["xgb_params"].copy()

    logger.info("Tuning hyperparameters for %s (%d trials)...", ticker, n_trials)
    tscv = TimeSeriesSplit(n_splits=3, gap=TRAINING_CONFIG["cv_gap_days"])

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "logloss",
            "random_state": 42,
        }

        # Class imbalance
        neg, pos = np.bincount(y.astype(int))
        if neg > 0 and pos > 0:
            params["scale_pos_weight"] = neg / pos

        aucs = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if y_val.nunique() < 2:
                continue

            scaler = StandardScaler()
            X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
            X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X.columns)

            model = XGBClassifier(**params)
            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_val_s)[:, 1]
            aucs.append(roc_auc_score(y_val, proba))

        return np.mean(aucs) if aucs else 0.5

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["eval_metric"] = "logloss"
    best["random_state"] = 42
    logger.info("Best params for %s: AUC=%.4f | %s", ticker, study.best_value, best)
    return best

# ═══════════════════════════════════════════════════════════════════════════════
# CORE: TRAIN A SINGLE TICKER
# ═══════════════════════════════════════════════════════════════════════════════

def train(ticker: str = "BBCA.JK", tune: bool = False):
    """Train and evaluate an XGBoost model for a single ticker.

    Args:
        ticker: Stock ticker symbol
        tune: If True, run Optuna hyperparameter tuning
    """
    df = load_features(FEATURES_PATH)
    df = df[df["ticker"] == ticker].copy()
    df = df.sort_index()

    if len(df) < MIN_ROWS:
        logger.warning("Skipping %s — only %d rows (need %d)", ticker, len(df), MIN_ROWS)
        return

    X, y = prepare_xy(df)

    if X.shape[0] == 0 or X.shape[1] == 0:
        logger.warning("Skipping %s — no features remaining", ticker)
        return
    if y.nunique() < 2:
        logger.warning("Skipping %s — only one class (%d samples)", ticker, len(y))
        return

    # ── Feature Pruning ──────────────────────────────────────────────────
    corr_thresh = TRAINING_CONFIG["corr_threshold"]
    var_thresh = TRAINING_CONFIG["var_threshold"]
    X = prune_features(X, corr_threshold=corr_thresh, var_threshold=var_thresh)

    if X.shape[1] == 0:
        logger.warning("Skipping %s — all features pruned", ticker)
        return

    # ── Class Distribution ───────────────────────────────────────────────
    class_counts = y.value_counts().to_dict()
    neg_count = class_counts.get(0, 0)
    pos_count = class_counts.get(1, 0)
    imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0

    # ── CV Setup ─────────────────────────────────────────────────────────
    gap_days = TRAINING_CONFIG["cv_gap_days"]
    n_splits = TRAINING_CONFIG["cv_splits"]
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)

    # ── Hyperparameters ──────────────────────────────────────────────────
    if tune:
        params = tune_hyperparameters(X, y, ticker, n_trials=TRAINING_CONFIG["optuna_trials"])
    else:
        params = TRAINING_CONFIG["xgb_params"].copy()

    # Add class imbalance weight
    if neg_count > 0 and pos_count > 0:
        params["scale_pos_weight"] = neg_count / pos_count

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"xgb_{ticker}"):
        mlflow.log_params(params)
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("cv_gap_days", gap_days)
        mlflow.log_param("cv_splits", n_splits)
        mlflow.log_param("class_imbalance_ratio", round(imbalance_ratio, 2))
        mlflow.log_param("n_sell", neg_count)
        mlflow.log_param("n_buy", pos_count)
        mlflow.log_param("feature_pruning", True)

        accs, f1s, aucs, precs, recs = [], [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if y_val.nunique() < 2:
                continue

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
            X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns)

            model = XGBClassifier(**params)
            model.fit(X_train_scaled, y_train)

            preds = model.predict(X_val_scaled)
            proba = model.predict_proba(X_val_scaled)[:, 1]

            accs.append(accuracy_score(y_val, preds))
            f1s.append(f1_score(y_val, preds, zero_division=0))
            aucs.append(roc_auc_score(y_val, proba))
            precs.append(precision_score(y_val, preds, zero_division=0))
            recs.append(recall_score(y_val, preds, zero_division=0))

        if not accs:
            logger.warning("No valid folds for %s, skipping", ticker)
            return

        mlflow.log_metric("avg_accuracy", np.mean(accs))
        mlflow.log_metric("avg_f1", np.mean(f1s))
        mlflow.log_metric("avg_roc_auc", np.mean(aucs))
        mlflow.log_metric("avg_precision", np.mean(precs))
        mlflow.log_metric("avg_recall", np.mean(recs))
        mlflow.log_metric("f1_std", np.std(f1s))

        logger.info("%s | Acc: %.4f | F1: %.4f | AUC: %.4f | Prec: %.4f | Rec: %.4f | Features: %d",
                    ticker, np.mean(accs), np.mean(f1s), np.mean(aucs),
                    np.mean(precs), np.mean(recs), X.shape[1])

        # ── Final Model + Calibration ────────────────────────────────────
        scaler_final = StandardScaler()
        X_all_scaled = pd.DataFrame(scaler_final.fit_transform(X), columns=X.columns)

        base_model = XGBClassifier(**params)
        base_model.fit(X_all_scaled, y)

        # Calibrate probabilities (isotonic regression)
        try:
            calibrated_model = CalibratedCVClassifier(
                estimator=base_model, method="isotonic", cv="prefit"
            )
            calibrated_model.fit(X_all_scaled, y)
            final_model = calibrated_model
            mlflow.log_param("calibrated", True)
            logger.info("Model calibrated with isotonic regression for %s", ticker)
        except Exception:
            final_model = base_model
            mlflow.log_param("calibrated", False)
            logger.debug("Calibration skipped for %s", ticker)

        if not IS_CI:
            mlflow.xgboost.log_model(base_model, "model")

            # Feature importance plot
            img_path = plot_feature_importance(base_model, X.columns, ticker)
            mlflow.log_artifact(img_path, "feature_importance")
            os.remove(img_path)

            # Confusion matrix (from final fold or all-data prediction)
            all_preds = base_model.predict(X_all_scaled)
            cm_path = plot_confusion_matrix(y, all_preds, ticker)
            mlflow.log_artifact(cm_path, "confusion_matrix")
            os.remove(cm_path)

            logger.info("Model & plots logged for %s", ticker)
        else:
            logger.info("CI mode — metrics only for %s", ticker)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost models for stock prediction")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Train a single ticker (default: all tickers)")
    parser.add_argument("--tune", action="store_true",
                        help="Enable Optuna hyperparameter tuning")
    parser.add_argument("--features", type=str, default=FEATURES_PATH,
                        help="Path to features file (parquet or csv)")
    args = parser.parse_args()

    FEATURES_PATH = args.features

    ticker_list = [args.ticker] if args.ticker else TICKERS

    if args.tune:
        # Only tune top-N most liquid tickers
        ticker_list = ticker_list[:TRAINING_CONFIG["tune_top_n_tickers"]]
        logger.info("Tuning mode enabled for %d tickers", len(ticker_list))

    for ticker in ticker_list:
        train(ticker, tune=args.tune)