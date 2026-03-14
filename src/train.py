import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import warnings
import mlflow.xgboost

warnings.filterwarnings("ignore")

# ── Use cwd-relative SQLite path ──
MLFLOW_DB = os.path.join(os.getcwd(), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

FEATURES_PATH = "data/processed/features.csv"
MLFLOW_EXPERIMENT = "indonesian-stock-prediction"
MIN_ROWS = 100  # Increased to handle embargo gap safely
IS_CI = bool(os.environ.get("CI")) 

def load_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)

def prepare_xy(df: pd.DataFrame):
    drop_cols = ["target", "ticker"]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y = df["target"]
    return X, y

def plot_feature_importance(model, feature_names, ticker, top_n=20):
    """Generates a bar chart of the top features used by XGBoost."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances - {ticker}")
    plt.bar(range(top_n), importance[indices], align="center", color="teal")
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.tight_layout()
    
    # Save temporarily to log to MLflow
    img_path = f"{ticker}_importance.png"
    plt.savefig(img_path)
    plt.close()
    return img_path

def train(ticker: str = "BBCA.JK"):
    df = load_features(FEATURES_PATH)
    df = df[df["ticker"] == ticker].copy()
    df = df.sort_index()

    if len(df) < MIN_ROWS:
        print(f"⚠️ Skipping {ticker} — only {len(df)} rows (need {MIN_ROWS})")
        return

    X, y = prepare_xy(df)

    if X.shape[0] == 0 or X.shape[1] == 0:
        return
    if y.nunique() < 2:
        return

    # ── Purged Time Series Cross Validation ───────────────────────────────────
    # Standard TimeSeriesSplit causes leakage because a 50-day moving average 
    # feature in the validation set overlaps with the end of the training set.
    # Enforce a "gap" (embargo) between train and validation indices.
    gap_days = 10 
    tscv = TimeSeriesSplit(n_splits=5, gap=gap_days)

    params = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8, 
        "eval_metric": "logloss",
        "random_state": 42
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"xgb_{ticker}"):
        mlflow.log_params(params)
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("cv_gap_days", gap_days)

        accs, f1s, aucs = [], [], []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if y_val.nunique() < 2:
                continue

            scaler = StandardScaler()
            # PRESERVE COLUMN NAMES for XGBoost (Fixes the 110 feature mismatch bug)
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
            X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns)

            model = XGBClassifier(**params)
            model.fit(X_train_scaled, y_train)

            preds = model.predict(X_val_scaled)
            proba = model.predict_proba(X_val_scaled)[:, 1]

            accs.append(accuracy_score(y_val, preds))
            f1s.append(f1_score(y_val, preds, zero_division=0))
            aucs.append(roc_auc_score(y_val, proba))

        if not accs:
            print(f"⚠️ No valid folds for {ticker}, skipping")
            return

        mlflow.log_metric("avg_accuracy", np.mean(accs))
        mlflow.log_metric("avg_f1", np.mean(f1s))
        mlflow.log_metric("avg_roc_auc", np.mean(aucs))

        print(f"✅ {ticker} | Accuracy: {np.mean(accs):.4f} | F1: {np.mean(f1s):.4f} | AUC: {np.mean(aucs):.4f}")

        # ── Final Model Training & Logging ────────────────────────────────────
        scaler_final = StandardScaler()
        X_all_scaled = pd.DataFrame(scaler_final.fit_transform(X), columns=X.columns)
        
        final_model = XGBClassifier(**params)
        final_model.fit(X_all_scaled, y)

        if not IS_CI:
            mlflow.xgboost.log_model(final_model, "model")

            # Generate and log feature importance plot
            img_path = plot_feature_importance(final_model, X.columns, ticker)
            mlflow.log_artifact(img_path, "feature_importance")
            os.remove(img_path) # Clean up local file
            
            print(f" 📦 Model & Importance plots logged for {ticker}")
        else:
            print(f" ⏭️ CI mode — metrics only, skipping model artifact")

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
