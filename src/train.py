import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pathlib

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
from model_store import (
    MLMODEL_FILENAME, index_path, model_dir, set_model_root, write_model_index,
)

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

#: Environment variable a fleet run uses to name the MLflow experiment its
#: workers must log to.
#:
#: ``loky`` workers are separate processes that re-import this module, so a
#: global assigned in the parent does not reach them — but the parent's
#: environment does.  ``configure_fleet`` sets this variable before any worker
#: is spawned, and the module-level read below picks it up in each worker.  The
#: default leaves today's experiment untouched.
EXPERIMENT_ENV = "IDX_MLFLOW_EXPERIMENT"
EXPERIMENT_NAME = os.environ.get(EXPERIMENT_ENV, MLFLOW_EXPERIMENT)

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
# THE AS-OF CUTOFF — a training set that ends before the evaluation begins
# ═══════════════════════════════════════════════════════════════════════════════

def filter_as_of(df: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Keep only the rows dated at or before ``as_of``.

    This is the train side of the holdout.  ``backtest.py``'s ``--start`` names
    the first day *evaluated*; this names the last day *trained on*, so a model
    produced under it has never seen the window it is measured over.

    Args:
        df: A frame indexed by date — one ticker's rows, or the pooled panel.
        as_of: Last date to train on, inclusive.  A date, a datetime or a
            ``Timestamp``; ``None`` leaves the frame untouched.

    Returns:
        The rows dated at or before ``as_of``, in their original order.  The
        whole frame is returned unchanged when ``as_of`` is ``None``, so the
        default path cannot filter anything.

    Raises:
        ValueError: ``as_of`` cannot be parsed, or it selects **no rows**.  An
            empty training set is refused rather than trained on: a model
            fitted on nothing is not a weak model, it is no model, and the
            ticker would otherwise be skipped with a misleading row count.
    """
    if as_of is None:
        return df

    try:
        cutoff = pd.Timestamp(as_of)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"as-of must be a date such as 2025-06-12, got {as_of!r}"
        ) from exc

    mask = df.index <= cutoff
    bounded = df.loc[mask]
    if bounded.empty:
        covered = (
            "the frame is empty"
            if len(df) == 0
            else f"the frame covers {df.index.min()} to {df.index.max()}"
        )
        raise ValueError(
            f"the as-of cutoff ({cutoff.date()}) selects no rows: {covered} "
            f"({len(df)} row(s)) — refusing to train on nothing"
        )
    return bounded


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
_GLOBAL_SCHEMA_CACHE: dict[tuple, list[str]] = {}


def select_global_feature_schema(features_path: str | None = None, *,
                                 use_shap: bool = True,
                                 top_k: int = SHAP_TOP_K,
                                 as_of=None) -> list[str]:
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

    ``as_of`` applies :func:`filter_as_of` to the pooled frame **before** the
    selection.  Choosing the 60 columns over rows that include the evaluation
    window would leak the future into the contract even with every fit
    truncated, so the cutoff covers the selection too.  The cache key includes
    it: two cutoffs are two different contracts.

    Returns the ordered column names; the caller gets a copy, so a schema in the
    cache cannot be mutated in place.
    """
    path = FEATURES_PATH if features_path is None else features_path
    key = (
        os.path.abspath(path), bool(use_shap), int(top_k),
        None if as_of is None else str(as_of),
    )
    if key in _GLOBAL_SCHEMA_CACHE:
        return list(_GLOBAL_SCHEMA_CACHE[key])

    pooled = load_features(path)
    pooled = filter_as_of(pooled, as_of)
    if as_of is not None:
        logger.info(
            "As-of cutoff %s: %d row(s) available for the shared feature selection",
            pd.Timestamp(as_of).date(), len(pooled),
        )
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
          feature_columns: list[str] | None = None,
          as_of=None) -> str | None:
    """Train an XGBoost (+ optional LGBM blend) model for a single ticker.

    Features: SHAP pruning, expanding walk-forward CV, LGBM soft-vote blending.

    ``feature_columns`` is the run's one shared schema (Task 1.4): every ticker
    is trained — and logged — on the same ordered list, selected once over the
    pooled frame rather than once per ticker.  When it is omitted the selection
    is made here, pooled and memoised, so a direct call still produces exactly
    the same contract as a full `--parallel` run.

    ``as_of`` is the holdout cutoff: every row dated after it is removed before
    the row guard, the CV splits and the final fit, so the logged model has
    never seen the evaluation window.  ``None`` (the default) trains on the full
    history, exactly as before.

    Returns:
        The store folder of the model this call logged — the model's MLflow id,
        as reported by :func:`_logged_model_folder` — or ``None`` when nothing
        was logged (CI mode, a skipped ticker, or a model whose id could not be
        read).  The caller turns the pairs into the model index (F10); no folder
        name is ever derived here.
    """
    df = load_features(FEATURES_PATH)
    df = df[df["ticker"] == ticker].copy().sort_index()
    # The cutoff is applied *before* the row guard, so a ticker is judged on the
    # history it would actually train on: AADI.JK has 353 rows in the full
    # artefact but 114 before the 2025-06-12 cutoff.
    df = filter_as_of(df, as_of)

    if len(df) < MIN_ROWS:
        return logger.warning(
            "Skipping %s — only %d rows%s", ticker, len(df),
            f" at or before {as_of}" if as_of is not None else "",
        )

    X, y = prepare_xy(df)
    if X.shape[0] == 0 or X.shape[1] == 0 or y.nunique() < 2:
        return

    # ── The run's one feature schema (Task 1.4) ───────────────────────────
    # One global selection replaces the per-ticker SHAP call that gave the 45
    # models 45 different feature lists and left nothing checking the column set
    # a model was fed.
    if feature_columns is None:
        feature_columns = select_global_feature_schema(use_shap=use_shap, as_of=as_of)
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

    # The experiment, not the constant: a holdout fleet is logged to its own
    # experiment, whose artifact location is the fleet directory.  Calling with
    # MLFLOW_EXPERIMENT here would put a holdout model into the production
    # experiment's store — the exact split brain `configure_fleet` exists to
    # prevent, and the post-run guard below catches.
    mlflow.set_experiment(EXPERIMENT_NAME)

    # The store folder of the model this call logs, or ``None`` when nothing was
    # logged.  Initialised before the run because the early returns below are
    # "no model was logged" too, and the caller must not receive a folder for a
    # model that does not exist (R3/R5).
    model_folder: str | None = None

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

        # ── Calibration: removed, and it never ran ─────────────────────────
        # The block that stood here called ``CalibratedCVClassifier`` — a name
        # that exists in no module — so it raised ``NameError`` on its first
        # statement, its own bare ``except Exception`` swallowed that, and every
        # training run then logged ``calibrated=False`` and reported success.
        # That parameter was the only trace of the step, and it said "not
        # calibrated" while the README advertised isotonic calibration as a
        # feature (F8).
        #
        # It is deleted rather than repaired: repairing it means choosing a
        # reference distribution, a method and a CV split, each of which changes
        # the probabilities a model serves.  That is a separate, tested change.
        # The import of ``CalibratedClassifierCV`` went with it — nothing else
        # used it.

        mlflow.set_tag("ticker", ticker)
        if not IS_CI:
            model_info = mlflow.xgboost.log_model(xgb, "model")
            # R2/R5: the folder MLflow just created for this model, taken from
            # the id it returned and never guessed (see _logged_model_folder).  A
            # model whose id cannot be read contributes no index entry rather
            # than a folder name that points nowhere.
            model_folder = _logged_model_folder(model_info, ticker)
            # R4: `log_model` creates the logged-model directory itself, so the
            # contract is attached *after* the fact, to the model id MLflow just
            # returned.  It lands beside MLmodel in that model's artifact
            # directory — mlruns/1/models/<model_id>/artifacts/feature_schema.json
            # — which is the exact directory every consumer already hands to
            # mlflow.xgboost.load_model (src/serve.py, scripts/build_model_index.py).
            if model_folder is not None:
                log_feature_schema(model_folder, feature_columns)
            img = plot_feature_importance(xgb, X.columns, ticker)
            mlflow.log_artifact(img, "feature_importance"); os.remove(img)
            cm = plot_confusion_matrix(y, xgb.predict(X), ticker)
            mlflow.log_artifact(cm, "confusion_matrix"); os.remove(cm)
        else:
            logger.info("CI mode — metrics only for %s", ticker)

    return model_folder


def _logged_model_folder(model_info, ticker: str) -> str | None:
    """The store folder ``mlflow.xgboost.log_model`` created for ``ticker``.

    R2: MLflow names a logged model's directory after the ``model_id`` it
    returns, and that was **measured**, not assumed.  Logging a throwaway model
    into a scratch store laid out exactly like this repository's — tracking URI
    ``sqlite:///<root>/mlflow.db``, the experiment's artifact location
    ``<root>/mlruns/1``, which is what ``mlflow.db`` records for experiment 1 —
    produced one directory under ``<root>/mlruns/1/models/``, named
    ``m-<32 hex>``, and ``ModelInfo.artifact_path`` was
    ``<root>/mlruns/1/models/<that same name>/artifacts``.  The id is therefore
    used verbatim: no prefix, no scanning, no overlap test.

    R5: ``log_model`` runs only outside CI, so ``model_info`` may be absent or
    carry no id (MLflow does not guarantee one).  Either way this returns
    ``None`` **with a warning** rather than raising: the ticker is left out of
    the index and fails loudly at prediction time, which is what an index entry
    pointing at a folder nothing ever created would not do.
    """
    model_id = getattr(model_info, "model_id", None)
    if isinstance(model_id, str) and model_id:
        return model_id

    logger.warning(
        "%s: the logged model reported no model id (%r) — it is left out of the "
        "model index, so no prediction can resolve it until the next retrain",
        ticker, model_info,
    )
    return None


def _train_one(args):
    """Wrapper for parallel execution.

    Returns ``(ticker, ok, folder)``: ``ok`` is whether training raised (as
    before), and ``folder`` is the store folder of the model that was logged,
    or ``None``.  The parent needs the folder to build the index, and this
    tuple is the only thing that crosses the process boundary — the workers
    return it and never write the index themselves (F10: one writer, in the
    parent).

    The task tuple is ``(ticker, tune, features_path, feature_columns)`` as it
    has always been, with an optional trailing ``as_of`` cutoff.  The cutoff
    must travel here because a loky worker is a fresh interpreter: the parent's
    module globals do not cross, and a worker that defaulted it would train on
    the full history while the parent believed it had a holdout.
    """
    ticker, tune, feats, feature_columns, *optional = args
    as_of = optional[0] if optional else None
    global FEATURES_PATH
    FEATURES_PATH = feats
    try:
        folder = train(ticker, tune=tune, feature_columns=feature_columns, as_of=as_of)
        return ticker, True, folder
    except Exception as e:
        logger.error("%s failed: %s", ticker, e)
        return ticker, False, None


def _index_entries_from_results(results) -> dict[str, str]:
    """``ticker -> model folder`` from the tuples the loky workers returned.

    Only a worker that both succeeded and reported the folder MLflow created
    contributes an entry; a failed worker and a CI-mode worker (trained, nothing
    logged) contribute nothing.  No folder name is derived here — the index
    cannot name a model that was never logged.
    """
    return {
        ticker: folder
        for ticker, ok, folder in results
        if ok and folder
    }


def _train_sequentially(tickers, *, tune: bool, feature_columns: list[str] | None,
                        use_shap: bool, expand_cv: bool, blend_lgbm: bool,
                        as_of=None) -> dict[str, str]:
    """Train ``tickers`` one at a time, containing each ticker's failure.

    The sequential mirror of ``_train_one`` plus ``_index_entries_from_results``:
    a ticker whose training raises is logged at ``error`` and skipped, and the
    loop carries on, so the one writer at the end of ``main`` is still reached.
    Without that, a single ticker's exception propagates out of the loop and the
    process dies before ``write_model_index`` runs — leaving the index stale
    behind a run that logged models, which is the failure F10 exists to end.
    ``plot_feature_importance`` raising *after* ``log_model`` for a model with
    fewer than its ``top_n`` features is exactly that case; the parallel path
    already survives it, so the sequential path must too.  (The plot bug itself
    is pre-existing and not fixed here.)

    Returns:
        ``ticker -> model folder`` for the tickers that logged a model.  A
        skipped ticker, a CI-mode ticker and a failed ticker all contribute
        nothing; no folder name is ever derived here.
    """
    entries: dict[str, str] = {}
    for ticker in tickers:
        try:
            folder = train(ticker, tune=tune, feature_columns=feature_columns,
                           use_shap=use_shap, expand_cv=expand_cv, blend_lgbm=blend_lgbm,
                           as_of=as_of)
        except Exception as exc:
            logger.error("%s failed: %s", ticker, exc)
            continue
        if folder:
            entries[ticker] = folder
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# THE FLEET ROOT — a holdout fleet beside the production one (Task 3.3)
# ═══════════════════════════════════════════════════════════════════════════════

def configure_fleet(model_root=None, experiment=None) -> None:
    """Point this run's index and its MLflow experiment at one fleet directory.

    A retrain that is meant to be a *holdout* fleet has to keep the logged
    models and the ``ticker -> folder`` index in the same directory, and out of
    the production store.  MLflow decides where a logged model lands from its
    experiment's artifact location, while ``model_store`` decides where the
    index lands from :data:`model_store.MODEL_ROOT_ENV`; these are two
    independent mechanisms, so this is the one place that sets both from one
    path.

    Args:
        model_root: The fleet's ``models`` directory — what ``model_store``
            will resolve after this call.  The experiment's artifact location is
            its parent, because MLflow logs a model at
            ``<artifact location>/models/<model_id>``.
        experiment: The MLflow experiment to log this fleet's runs to.  Created
            with that artifact location when it does not exist; an existing one
            is reused, and the post-run check below catches it if it points
            somewhere else.

    ``None`` for both is the default production retrain and changes nothing.

    Raises:
        ValueError: exactly one of the two was given.  Either half alone writes
            a split brain: ``--model-root`` alone logs models into the
            production experiment while writing the index into the holdout
            root, and ``--experiment`` alone logs them elsewhere while writing
            the *production* index.  Both produce an index that names models the
            store does not have, or a production store quietly grown by an
            evaluation fleet.
    """
    if model_root is None and experiment is None:
        return
    if model_root is None or experiment is None:
        raise ValueError(
            "a fleet root needs both a --model-root and an --experiment, given "
            "together: the index is written beside the fleet whose models the run "
            "logs, and either one alone points the index and the models at "
            "different directories"
        )

    set_model_root(model_root)
    experiment_root = model_dir().parent

    client = mlflow.MlflowClient()
    existing = client.get_experiment_by_name(str(experiment))
    if existing is not None and getattr(existing, "lifecycle_stage", "active") == "deleted":
        # A deleted experiment still occupies its name, so creating it again
        # fails inside the first worker and the run ends with "no model was
        # logged" — a misleading message for a name collision.  Say what
        # happened instead of reusing a store MLflow has retired.
        raise ValueError(
            f"MLflow experiment {str(experiment)!r} exists but is deleted; "
            f"choose a different --experiment name — a deleted experiment "
            f"cannot be logged to"
        )

    global EXPERIMENT_NAME
    EXPERIMENT_NAME = str(experiment)
    os.environ[EXPERIMENT_ENV] = EXPERIMENT_NAME

    if existing is None:
        client.create_experiment(
            EXPERIMENT_NAME, artifact_location=experiment_root.as_uri()
        )
        logger.info(
            "Created MLflow experiment %r with artifact location %s",
            EXPERIMENT_NAME, experiment_root,
        )
    else:
        logger.info(
            "Using existing MLflow experiment %r (artifact location %s); the "
            "fleet root is %s",
            EXPERIMENT_NAME, getattr(existing, "artifact_location", "unknown"),
            model_dir(),
        )


def missing_model_folders(entries, root=None) -> list[str]:
    """Tickers whose logged-model folder is absent from the fleet directory.

    The post-run guard for the split brain :func:`configure_fleet` prevents: an
    index naming a folder that exists in the *other* fleet would look like a
    successful 45/45 retrain while every lookup for it fails.  Returns the
    tickers whose ``<root>/<folder>/artifacts/MLmodel`` is missing; an empty
    list means the index agrees with the fleet directory.
    """
    base = model_dir() if root is None else pathlib.Path(root).resolve()
    return [
        ticker
        for ticker, folder in entries.items()
        if not (base / str(folder) / "artifacts" / MLMODEL_FILENAME).is_file()
    ]


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
    parser.add_argument("--as-of", metavar="YYYY-MM-DD", default=None,
                        help="Train only on rows at or before this date, so the "
                             "resulting models never see the evaluation window "
                             "(default: the full history)")
    parser.add_argument("--model-root", metavar="DIR", default=None,
                        help="Write this run's model index beside a fleet in DIR "
                             "instead of mlruns/1/models. Requires --experiment; "
                             "the production fleet is left untouched")
    parser.add_argument("--experiment", metavar="NAME", default=None,
                        help="Log this run's models to the MLflow experiment NAME "
                             "(created with the fleet directory as its artifact "
                             "location). Requires --model-root")
    args = parser.parse_args()

    # The fleet root first: it decides where the index is written and which
    # experiment the workers log to.  A half-specified fleet is refused rather
    # than resolved to a default, because the default is the production store.
    try:
        configure_fleet(args.model_root, args.experiment)
    except ValueError as exc:
        parser.error(str(exc))

    FEATURES_PATH = args.features
    tickers = [args.ticker] if args.ticker else TICKERS
    if args.tune:
        tickers = tickers[:TRAINING_CONFIG["tune_top_n_tickers"]]

    use_shap, expand_cv, blend = not args.no_shap, not args.no_expand_cv, not args.no_blend

    if args.as_of is not None:
        logger.info(
            "As-of cutoff %s: training on rows at or before it, and selecting "
            "the shared feature schema from those rows only", args.as_of,
        )
    logger.info("Model store: %s (index %s)", model_dir(), index_path())

    # One selection for the whole run, over the pooled frame, before any model is
    # trained (Task 1.4).  Computed in this process and handed to every worker so
    # the parallel branch shares one ordered list instead of letting each worker
    # process make its own selection.  The cutoff applies here too (Task 3.3).
    try:
        feature_columns = select_global_feature_schema(
            FEATURES_PATH, use_shap=use_shap, as_of=args.as_of
        )
    except ValueError as exc:
        parser.error(str(exc))
    logger.info("Training %d tickers on one shared %d-column schema",
                len(tickers), len(feature_columns))

    if args.parallel and not args.ticker:
        from joblib import Parallel, delayed
        logger.info("Parallel training on %d tickers...", len(tickers))
        # The cutoff is the optional fifth element: loky workers are fresh
        # interpreters, so it has to cross the process boundary explicitly.
        tasks = [
            (t, args.tune, FEATURES_PATH, feature_columns, args.as_of) for t in tickers
        ]
        results = Parallel(n_jobs=-1, backend="loky")(delayed(_train_one)(t) for t in tasks)
        ok = sum(1 for _, s, _ in results if s)
        logger.info("Done: %d/%d tickers trained", ok, len(tickers))
        entries = _index_entries_from_results(results)
    else:
        # Contained per ticker, like the parallel path: one ticker's exception
        # must not abort the loop and skip the index write below.
        entries = _train_sequentially(
            tickers, tune=args.tune, feature_columns=feature_columns,
            use_shap=use_shap, expand_cv=expand_cv, blend_lgbm=blend,
            as_of=args.as_of,
        )

    # The index and the fleet must be the same directory.  Checking before the
    # write means a misconfigured fleet root leaves the previous index intact
    # instead of replacing it with one that names models this store does not
    # have — the state in which every prediction fails while the file looks
    # valid.
    if entries:
        missing = missing_model_folders(entries)
        if missing:
            logger.error(
                "Refusing to write an index that names %d model folder(s) absent "
                "from %s (%s%s): this run logged its models somewhere else — check "
                "--model-root and --experiment",
                len(missing), model_dir(), ", ".join(sorted(missing)[:5]),
                f", and {len(missing) - 5} more" if len(missing) > 5 else "",
            )
            raise SystemExit(1)

    # F10 — the index's single writer, in the parent, once, after every ticker
    # has been trained and logged.  No worker writes it: a worker returns its
    # ticker and folder in the tuple above and nothing else.  An empty mapping
    # is refused inside write_model_index rather than written, so a CI-mode run
    # (which logs no model at all) cannot empty an index predictions depend on.
    write_model_index(entries)