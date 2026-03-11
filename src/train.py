import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Switch to SQLite backend (fixes FutureWarning) ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")

FEATURES_PATH = "data/processed/features.csv"
MLFLOW_EXPERIMENT = "indonesian-stock-prediction"
MIN_ROWS = 50  # minimum rows needed to train


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def prepare_xy(df: pd.DataFrame):
    drop_cols = ["target", "ticker"]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y = df["target"]
    return X, y


def train(ticker: str = "BBCA.JK"):
    df = load_features(FEATURES_PATH)
    df = df[df["ticker"] == ticker].copy()
    df = df.sort_index()

    # ── GUARD: skip if not enough data ──
    if len(df) < MIN_ROWS:
        print(f"⚠️  Skipping {ticker} — only {len(df)} rows (need {MIN_ROWS})")
        return

    X, y = prepare_xy(df)

    # ── GUARD: skip if feature matrix is empty ──
    if X.shape[0] == 0 or X.shape[1] == 0:
        print(f"⚠️  Skipping {ticker} — empty feature matrix after preprocessing")
        return

    # ── GUARD: skip if only one class in target ──
    if y.nunique() < 2:
        print(f"⚠️  Skipping {ticker} — target has only one class")
        return

    tscv = TimeSeriesSplit(n_splits=5)

    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "eval_metric": "logloss",
        "random_state": 42
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"xgb_{ticker}"):
        mlflow.log_params(params)
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_rows", len(df))

        accs, f1s, aucs = [], [], []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # ── GUARD: skip fold if val set has only one class ──
            if y_val.nunique() < 2:
                print(f"  ⚠️  Skipping fold {fold} for {ticker} — single class in val")
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            proba = model.predict_proba(X_val)[:, 1]

            accs.append(accuracy_score(y_val, preds))
            f1s.append(f1_score(y_val, preds, zero_division=0))
            aucs.append(roc_auc_score(y_val, proba))

        if not accs:
            print(f"⚠️  No valid folds for {ticker}, skipping MLflow log")
            return

        mlflow.log_metric("avg_accuracy", np.mean(accs))
        mlflow.log_metric("avg_f1", np.mean(f1s))
        mlflow.log_metric("avg_roc_auc", np.mean(aucs))

        print(f"✅ {ticker} | Accuracy: {np.mean(accs):.4f} | F1: {np.mean(f1s):.4f} | AUC: {np.mean(aucs):.4f}")

        # Train final model on all data
        scaler_final = StandardScaler()
        X_all = scaler_final.fit_transform(X)
        final_model = XGBClassifier(**params)
        final_model.fit(X_all, y)

        mlflow.sklearn.log_model(final_model, name="model")
        print(f"  📦 Model logged to MLflow for {ticker}")


if __name__ == "__main__":
    TICKERS = [
        "AADI.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK",
        "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK", "BBCA.JK",
        "BBNI.JK", "BBRI.JK", "BBTN.JK", "BMRI.JK", "BREN.JK",
        "BRIS.JK", "BRPT.JK", "CPIN.JK", "CTRA.JK", "EXCL.JK",
        "GOTO.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INKP.JK",
        "ISAT.JK", "ITMG.JK", "JPFA.JK", "JSMR.JK", "KLBF.JK",
        "MAPA.JK", "MAPI.JK", "MBMA.JK", "MDKA.JK", "MEDC.JK",
        "PGAS.JK", "PGEO.JK", "PTBA.JK", "SIDO.JK", "SMGR.JK",
        "SMRA.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK",
    ]
    for ticker in TICKERS:
        train(ticker)
