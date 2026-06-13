#!/usr/bin/env python
"""Build a ticker-to-model mapping index.

Scans all model folders in mlruns/1/models/, loads each model once,
extracts its expected feature names, matches them against ticker-specific
feature sets, and saves a model_index.json.

Usage:
    python scripts/build_model_index.py
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import mlflow.xgboost

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'mlruns', '1', 'models')
INDEX_FILE = os.path.join(MODELS_DIR, 'model_index.json')
FEATURES_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'features.csv')


def get_feature_names(model) -> list:
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    if hasattr(model, 'estimator') and hasattr(model.estimator, 'feature_names_in_'):
        return list(model.estimator.feature_names_in_)
    if hasattr(model, 'get_booster'):
        return model.get_booster().feature_names or []
    return []


def build_index():
    print(f"Loading features from {FEATURES_FILE}...")
    df = pd.read_csv(FEATURES_FILE, low_memory=False)
    tickers = df['ticker'].unique() if 'ticker' in df.columns else []
    print(f"Found {len(tickers)} tickers in features CSV")

    # Build per-ticker feature sets
    ticker_features = {}
    for ticker in tickers:
        tdf = df[df['ticker'] == ticker]
        if tdf.empty:
            continue
        numeric_cols = set(tdf.select_dtypes(include=[np.number]).columns) - {'target'}
        ticker_features[ticker] = numeric_cols

    model_folders = sorted(os.listdir(MODELS_DIR))
    print(f"Scanning {len(model_folders)} model folders...")

    index = {}
    errors = []

    for i, folder in enumerate(model_folders):
        artifact_dir = os.path.join(MODELS_DIR, folder, 'artifacts')
        mlmodel_file = os.path.join(artifact_dir, 'MLmodel')
        if not os.path.exists(mlmodel_file):
            continue

        try:
            model = mlflow.xgboost.load_model(artifact_dir)
            model_features = set(get_feature_names(model))

            if not model_features:
                continue

            # Match against each ticker's feature set
            best_ticker = None
            best_match = 0

            for ticker, tfeats in ticker_features.items():
                overlap = len(model_features & tfeats)
                if overlap > best_match:
                    best_match = overlap
                    best_ticker = ticker

            index[folder] = {
                'ticker': best_ticker,
                'match_score': best_match,
                'n_model_features': len(model_features),
            }
        except Exception as e:
            errors.append((folder, str(e)))
            continue

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(model_folders)}...")

    # Build ticker -> model path mapping
    ticker_to_model = {}
    for folder, info in index.items():
        ticker = info['ticker']
        if ticker and (ticker not in ticker_to_model or
                       info['match_score'] > ticker_to_model[ticker]['match_score']):
            ticker_to_model[ticker] = {
                'model_folder': folder,
                'match_score': info['match_score'],
                'n_features': info['n_model_features'],
            }

    output = {
        'total_models': len(model_folders),
        'indexed_models': len(index),
        'mapped_tickers': len(ticker_to_model),
        'errors': len(errors),
        'ticker_to_model': ticker_to_model,
    }

    with open(INDEX_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nIndex saved to {INDEX_FILE}")
    print(f"Mapped {len(ticker_to_model)}/{len(tickers)} tickers to models")
    if errors:
        print(f"Errors (first 5): {errors[:5]}")

    print(f"\nSample mappings:")
    for ticker in sorted(ticker_to_model.keys())[:10]:
        info = ticker_to_model[ticker]
        print(f"  {ticker}: {info['model_folder'][:20]}... (score={info['match_score']})")


if __name__ == '__main__':
    build_index()
