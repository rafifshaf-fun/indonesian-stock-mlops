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
from features.schema import (
    FeatureSchemaError, align_to_schema, candidate_columns, is_unproducible,
    log_feature_schema,
)

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# ── MLflow setup ──────────────────────────────────────────────────────────────
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"  # required for MLflow 3.13+
mlflow_db_path = os.path.join(os.getcwd(), MLFLOW_DB)
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{mlflow_db_path}"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

FEATURES_PATH = DATA_PROCESSED_PATH  # Default: Parquet
MIN_ROWS = TRAINING_CONFIG["min_rows"]
IS_CI = bool(os.environ.get("CI"))

#: How many features SHAP keeps for a run.  Named rather than repeated because
#: the value now belongs to one selection shared by every ticker (Task 1.4),
#: not to a per-ticker call.
SHAP_TOP_K = 60

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
# CORE: TRAIN A SINGLE TICKER (SHAP pruning + expanding CV + LGBM blend)
# ═══════════════════════════════════════════════════════════════════════════════

def _select_features_shap(X, y, top_k: int = SHAP_TOP_K) -> pd.DataFrame:
    """SHAP-based feature selection. Falls back to correlation pruning if SHAP unavailable."""
    try:
        import shap
    except ImportError:
        return X

    if X.shape[1] <= top_k:
        return X

    try:
        neg, pos = np.bincount(y.astype(int))
        sw = neg / pos if pos > 0 else 1
        model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                              eval_metric="logloss", random_state=42,
                              scale_pos_weight=sw, verbosity=0)
        sc = StandardScaler()
        X_s = pd.DataFrame(sc.fit_transform(X), columns=X.columns, index=X.index)
        model.fit(X_s, y)
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X_s.iloc[:min(1000, len(X_s))])
        if isinstance(vals, list):
            vals = vals[1]
        imp = np.abs(vals).mean(axis=0)
        top = np.argsort(imp)[::-1][:top_k]
        cols = X.columns[top].tolist()
        logger.info("SHAP pruning: %d → %d features", X.shape[1], len(cols))
        return X[cols]
    except Exception:
        return X


# ═══════════════════════════════════════════════════════════════════════════════
# ONE FEATURE SCHEMA FOR EVERY TICKER (Task 1.4, F5)
# ═══════════════════════════════════════════════════════════════════════════════

# The pooled selection is deterministic but not free — one SHAP pass over every
# ticker's rows — and `train()` is otherwise entered once per ticker, so the
# result is memoised.  Keyed by (features path, selection mode, top_k): pointing
# `--features` at another file inside one process must not reuse this file's
# contract.  At most one entry is kept, because one run has one contract.
_GLOBAL_SCHEMA_CACHE: dict[tuple[str, bool, int], list[str]] = {}


def select_global_feature_schema(features_path: str | None = None, *,
                                 use_shap: bool = True,
                                 top_k: int = SHAP_TOP_K) -> list[str]:
    """The one ordered feature list every ticker in this run is trained on.

    The audit found the 45 logged models carrying 45 *different* 60-name lists
    (union 132, intersection 1), because ``_select_features_shap`` ran once per
    ticker inside ``train()``.  Nothing then checked the column set a model was
    fed, so consumers padded the missing names with ``NaN``, and a change to any
    feature module invalidated all 45 models with no failure signal.

    This runs that same selection exactly **once**, over every ticker's rows at
    once, on the candidate set serving can actually reproduce
    (:func:`features.schema.candidate_columns`).  ``train()`` then fits, and
    logs, every ticker on exactly these columns in exactly this order, and
    ``feature_schema.json`` records the list beside each logged model.

    Returns the ordered column names; the caller gets a copy, so a schema in the
    cache cannot be mutated in place.
    """
    path = FEATURES_PATH if features_path is None else features_path
    key = (os.path.abspath(path), bool(use_shap), int(top_k))
    if key in _GLOBAL_SCHEMA_CACHE:
        return list(_GLOBAL_SCHEMA_CACHE[key])

    pooled = load_features(path)
    X_pooled, y_pooled = prepare_xy(pooled)

    # The exclusion is applied to the candidate set *before* the selection, and
    # it is the only mechanism available here: this function reads the already
    # engineered frame and has neither raw OHLCV nor IHSG data with which to
    # recompute what serving could produce.  A schema that named a column
    # serving cannot build would be exactly the silent-NaN defect F5 removes.
    candidates = candidate_columns(X_pooled.columns)
    X_candidates = X_pooled[candidates]

    if use_shap:
        selected = _select_features_shap(X_candidates, y_pooled, top_k=top_k)
    else:
        selected = prune_features(X_candidates, verbose=True)

    columns = selected.columns.tolist()
    if not columns:
        raise RuntimeError(
            f"global feature selection returned no columns for {path} "
            f"({len(candidates)} candidates, pooled shape {X_pooled.shape})"
        )

    logger.info(
        "Global feature schema: %d pooled numeric columns → %d candidates "
        "(serving-unproducible families excluded) → %d selected — once, for "
        "every ticker", X_pooled.shape[1], len(candidates), len(columns),
    )

    _GLOBAL_SCHEMA_CACHE.clear()  # one run, one contract
    _GLOBAL_SCHEMA_CACHE[key] = columns
    return list(columns)


def _expanding_cv_splits(X, n_splits=5, gap=10):
    """Expanding-window CV splits — each fold grows the training window."""
    n = len(X)
    ts = n // (n_splits + 1)
    for i in range(n_splits):
        te_start = n - (n_splits - i) * ts
        te_end = n - (n_splits - i - 1) * ts
        tr_end = te_start - gap
        if tr_end <= ts:
            continue
        yield np.arange(tr_end), np.arange(te_start, min(te_end, n))


def train(ticker: str = "BBCA.JK", tune: bool = False, use_shap: bool = True,
          expand_cv: bool = True, blend_lgbm: bool = True,
          feature_columns: list[str] | None = None):
    """Train an XGBoost (+ optional LGBM blend) model for a single ticker.

    Features: SHAP pruning, expanding walk-forward CV, LGBM soft-vote blending.

    ``feature_columns`` is the run's one shared schema (Task 1.4): every ticker
    is trained — and logged — on the same ordered list, selected once over the
    pooled frame rather than once per ticker.  When it is omitted the selection
    is made here, pooled and memoised, so a direct call still produces exactly
    the same contract as a full `--parallel` run.
    """
    df = load_features(FEATURES_PATH)
    df = df[df["ticker"] == ticker].copy().sort_index()

    if len(df) < MIN_ROWS:
        return logger.warning("Skipping %s — only %d rows", ticker, len(df))

    X, y = prepare_xy(df)
    if X.shape[0] == 0 or X.shape[1] == 0 or y.nunique() < 2:
        return

    # ── The run's one feature schema (Task 1.4) ───────────────────────────
    # One global selection replaces the per-ticker SHAP call that gave the 45
    # models 45 different feature lists and left nothing checking the column set
    # a model was fed.
    if feature_columns is None:
        feature_columns = select_global_feature_schema(use_shap=use_shap)
    else:
        # A caller-supplied schema is held to the same rule as a derived one: a
        # contract naming a column serving cannot build is the defect this task
        # removes, wherever the list came from.
        unproducible = [c for c in feature_columns if is_unproducible(c)]
        if unproducible:
            raise FeatureSchemaError(
                f"feature schema names serving-unproducible column(s): {unproducible}"
            )

    # Reorder by name and drop the intermediates the model never sees.  A column
    # the schema names but this frame lacks raises instead of being padded with
    # NaN, which is the silent failure the audit measured at ~0.10 of predicted
    # probability against the 0.50 decision boundary.
    X = align_to_schema(X, feature_columns)

    if X.shape[1] == 0:
        return

    neg_count, pos_count = (y == 0).sum(), (y == 1).sum()
    gap_days, n_splits = TRAINING_CONFIG["cv_gap_days"], TRAINING_CONFIG["cv_splits"]

    # ── CV splits ─────────────────────────────────────────────────────────
    if expand_cv:
        cv_splits = list(_expanding_cv_splits(X, n_splits, gap_days))
    else:
        cv_splits = list(TimeSeriesSplit(n_splits=n_splits, gap=gap_days).split(X))

    # ── Params ────────────────────────────────────────────────────────────
    if tune:
        params = tune_hyperparameters(X, y, ticker)
    else:
        params = TRAINING_CONFIG["xgb_params"].copy()
    if neg_count > 0 and pos_count > 0:
        params["scale_pos_weight"] = neg_count / pos_count
    params["nthread"] = 2

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"xgb_{ticker}"):
        mlflow.log_params(params)
        for k, v in {"ticker": ticker, "n_features": X.shape[1], "n_rows": len(df),
                     "shap_pruning": use_shap, "expanding_cv": expand_cv,
                     "lgbm_blend": blend_lgbm,
                     "class_imbalance": round(neg_count / max(pos_count, 1), 2)}.items():
            mlflow.log_param(k, v)

        accs, f1s, aucs, precs, recs = [], [], [], [], []

        for train_idx, val_idx in cv_splits:
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
            if y_vl.nunique() < 2:
                continue

            sc = StandardScaler()
            X_tr_s = pd.DataFrame(sc.fit_transform(X_tr), columns=X.columns)
            X_vl_s = pd.DataFrame(sc.transform(X_vl), columns=X.columns)

            model = XGBClassifier(**params)
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_vl_s)
            proba = model.predict_proba(X_vl_s)[:, 1]

            accs.append(accuracy_score(y_vl, preds))
            f1s.append(f1_score(y_vl, preds, zero_division=0))
            aucs.append(roc_auc_score(y_vl, proba))
            precs.append(precision_score(y_vl, preds, zero_division=0))
            recs.append(recall_score(y_vl, preds, zero_division=0))

        if not accs:
            return

        for n, v in [("avg_accuracy", accs), ("avg_f1", f1s), ("avg_roc_auc", aucs),
                     ("avg_precision", precs), ("avg_recall", recs),
                     ("f1_std", [np.std(f1s)])]:
            mlflow.log_metric(n, np.mean(v))

        logger.info("%s | Acc:%.4f F1:%.4f AUC:%.4f Feat:%d",
                    ticker, np.mean(accs), np.mean(f1s), np.mean(aucs), X.shape[1])

        # ── Final model ────────────────────────────────────────────────────
        # Fitted on the raw frame.  A tree is invariant to per-feature monotone
        # rescaling, so standardising here bought the model nothing and only
        # moved its split thresholds into z-score units — units no consumer
        # reconstructs, because no scaler was ever persisted and serve.py,
        # cli.py, predict.py and backtest.py all feed raw values.  That
        # train/serve mismatch is the audit's F1 finding (0 BUY across 1,500
        # historical rows).  Without a scaler there is also nothing to persist
        # and keep in sync.
        xgb = XGBClassifier(**params)
        xgb.fit(X, y)

        # ── LGBM blend (optional) ──────────────────────────────────────────
        proba_blend = None
        if blend_lgbm:
            try:
                from lightgbm import LGBMClassifier
                lgbm = LGBMClassifier(
                    n_estimators=params.get("n_estimators",200), learning_rate=params.get("learning_rate",0.05),
                    max_depth=params.get("max_depth",4), random_state=42, verbosity=-1,
                )
                # Same raw frame as the final fit: blending a scaler-fitted
                # learner with raw predictions is the same mismatch one step on.
                lgbm.fit(X, y)
                proba_blend = (xgb.predict_proba(X)[:,1] + lgbm.predict_proba(X)[:,1]) / 2
                mlflow.log_metric("blend_accuracy",
                                  accuracy_score(y, (proba_blend >= 0.5).astype(int)))
                logger.info("LGBM blended for %s", ticker)
            except ImportError:
                pass
            except Exception as e:
                logger.debug("LGBM blend failed for %s: %s", ticker, e)

        # ── Calibration ────────────────────────────────────────────────────
        # Unreachable as written — CalibratedCVClassifier is not a name that
        # exists, so this always raises and always logs calibrated=False.  It is
        # left alone deliberately (its removal is Task 1.9's), but it is on raw X
        # like every other call site: a dangling reference to a deleted frame
        # would be a defect here too, even on a line that never runs.
        try:
            final = CalibratedCVClassifier(estimator=xgb, method="isotonic", cv="prefit")
            final.fit(X, y)
            mlflow.log_param("calibrated", True)
        except Exception:
            final = xgb
            mlflow.log_param("calibrated", False)

        mlflow.set_tag("ticker", ticker)
        if not IS_CI:
            model_info = mlflow.xgboost.log_model(xgb, "model")
            # R4: `log_model` creates the logged-model directory itself, so the
            # contract is attached *after* the fact, to the model id MLflow just
            # returned.  It lands beside MLmodel in that model's artifact
            # directory — mlruns/1/models/<model_id>/artifacts/feature_schema.json
            # — which is the exact directory every consumer already hands to
            # mlflow.xgboost.load_model (src/serve.py, scripts/build_model_index.py).
            log_feature_schema(model_info.model_id, feature_columns)
            img = plot_feature_importance(xgb, X.columns, ticker)
            mlflow.log_artifact(img, "feature_importance"); os.remove(img)
            cm = plot_confusion_matrix(y, xgb.predict(X), ticker)
            mlflow.log_artifact(cm, "confusion_matrix"); os.remove(cm)
        else:
            logger.info("CI mode — metrics only for %s", ticker)


def _train_one(args):
    """Wrapper for parallel execution."""
    ticker, tune, feats, feature_columns = args
    global FEATURES_PATH
    FEATURES_PATH = feats
    try:
        train(ticker, tune=tune, feature_columns=feature_columns)
        return ticker, True
    except Exception as e:
        logger.error("%s failed: %s", ticker, e)
        return ticker, False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost (+LGBM blend) models")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--features", type=str, default=FEATURES_PATH)
    parser.add_argument("--parallel", action="store_true",
                        help="Parallel training across tickers (~4x speedup)")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--no-expand-cv", action="store_true")
    parser.add_argument("--no-blend", action="store_true")
    args = parser.parse_args()

    FEATURES_PATH = args.features
    tickers = [args.ticker] if args.ticker else TICKERS
    if args.tune:
        tickers = tickers[:TRAINING_CONFIG["tune_top_n_tickers"]]

    use_shap, expand_cv, blend = not args.no_shap, not args.no_expand_cv, not args.no_blend

    # One selection for the whole run, over the pooled frame, before any model is
    # trained (Task 1.4).  Computed in this process and handed to every worker so
    # the parallel branch shares one ordered list instead of letting each worker
    # process make its own selection.
    feature_columns = select_global_feature_schema(FEATURES_PATH, use_shap=use_shap)
    logger.info("Training %d tickers on one shared %d-column schema",
                len(tickers), len(feature_columns))

    if args.parallel and not args.ticker:
        from joblib import Parallel, delayed
        logger.info("Parallel training on %d tickers...", len(tickers))
        tasks = [(t, args.tune, FEATURES_PATH, feature_columns) for t in tickers]
        results = Parallel(n_jobs=-1, backend="loky")(delayed(_train_one)(t) for t in tasks)
        ok = sum(1 for _, s in results if s)
        logger.info("Done: %d/%d tickers trained", ok, len(tickers))
    else:
        for t in tickers:
            train(t, tune=args.tune, feature_columns=feature_columns,
                  use_shap=use_shap, expand_cv=expand_cv, blend_lgbm=blend)