# 🇮🇩 Indonesian Stock Prediction — MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Async-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Native-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)
![Grafana](https://img.shields.io/badge/Grafana-Monitoring-F46800?logo=grafana&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)

An end-to-end MLOps system that predicts **BUY/SELL signals** for 45 Indonesian blue-chip stocks (IDX) using machine learning, real-time monitoring, and automated retraining. 

This project demonstrates production-grade MLOps practices including **Purged Time-Series Cross Validation**, **Async Model Serving**, and **Model Registry Management**.

---

## ✨ Key Features & Upgrades (v1.2)

- 🤖 **Native XGBoost Modeling** — Switched to `mlflow.xgboost` native flavor for better performance, explicit feature naming, and accurate feature importance logging.
- 📊 **Advanced Feature Engineering (200+)** — Includes 100+ technical indicators, real-time Yahoo Finance fundamentals, USD/IDR exchange rates, and custom `Google Trends` sentiment tracking.
- 🏦 **External Macro Data Caching** — Automatically fetches and caches WTI Oil, Gold, Coal, US10Y Yields, and Bank Indonesia Interest Rates via FRED API to enrich model context.
- ⚡ **Asynchronous API** — Completely overhauled `serve.py` using `asyncio` and FastAPI Background Tasks. The server no longer freezes during heavy concurrent requests or background data fetching.
- 🛡️ **Bulletproof Inference** — Implemented robust fallback mechanisms. If Yahoo Finance or FRED APIs go down, the model safely imputes missing data to guarantee 100% prediction uptime.
- 📡 **Real-time Observability** — Grafana dashboard built on Prometheus metrics tracking signal distributions, API latency, and model confidence drift.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        Data Sources                         │
│  yfinance (OHLCV + Fundamentals) │ FRED API │ Bank Indonesia │ Google Trends │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Engineering                      │
│     Technical Indicators (200+) │ Macro │ Sentiment         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              XGBoost Model Training                         │
│         TimeSeriesSplit CV │ MLflow Experiment Tracking     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI Prediction API                       │
│          /predict │ /metrics │ /health │ /docs              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Prometheus + Grafana Monitoring                │
│    Confidence Scores │ Signal Counts │ API Latency          │
└─────────────────────────────────────────────────────────────┘

## ✨ Features

- 🤖 **ML Model** — Per-ticker XGBoost classifiers with purged TimeSeriesSplit CV and isotonic calibration
- 📊 **172+ Features** — Technical indicators, ICT Smart Money Concepts, Enhanced MAs, macro, FX, fundamentals
- 🏦 **Multi-source Data** — Yahoo Finance, FRED API, Bank Indonesia, NewsAPI + VADER sentiment
- 📰 **Sentiment Overlay** — News sentiment adjusts prediction confidence by ±10% in real-time
- 📈 **MLflow Tracking** — Full experiment history, run comparison, model registry
- 📡 **Real-time Monitoring** — Grafana dashboard with BUY/SELL counts, confidence gauges, API latency
- 🐳 **Fully Dockerized** — `docker compose up -d` to spin up the entire stack
- ⚡ **Prediction Cache** — SQLite-backed daily cache for sub-second predictions after first hit
- 🔧 **Unified CLI** — `python cli.py predict|backtest|sentiment|train|serve|status|list`

---

## 🗂️ Project Structure

## 🗂️ Project Structure

```
├── src/
│   ├── features/              # Modular feature engineering package
│   │   ├── __init__.py         #   Re-exports all feature functions
│   │   ├── fetchers.py         #   fetch_fundamentals, fetch_usdidr, fetch_news_sentiment, etc.
│   │   ├── indicators.py       #   compute_ta_features, compute_custom_features
│   │   ├── enhanced_mas.py     #   compute_enhanced_mas (~15 features)
│   │   ├── ict.py              #   compute_ict_features (~25 ICT/Smart Money features)
│   │   ├── volume_profile.py   #   compute_volume_profile_features (~20 features)
│   │   ├── market.py           #   compute_market_context, compute_cross_stock_features
│   │   └── pipeline.py         #   engineer_features_for_ticker, build_feature_set
│   ├── ingest.py               # Fetch raw OHLCV data from yfinance
│   ├── train.py                # XGBoost training + MLflow logging + Optuna tuning
│   ├── serve.py                # FastAPI prediction server with sentiment overlay
│   ├── backtest.py             # Walk-forward backtesting engine
│   ├── news_sentiment.py       # NewsAPI + VADER sentiment analysis
│   └── config.py               # Single source of truth: tickers, sectors, feature flags
├── cli.py                      # Unified CLI (predict, backtest, sentiment, train, serve)
├── predict.py                  # Standalone prediction script (API or local mode)
├── data/
│   ├── raw/                    # Raw OHLCV CSVs
│   ├── processed/              # Engineered feature Parquet/CSV
│   └── prediction_cache.db     # Daily prediction cache (sub-second responses)
├── mlruns/                     # MLflow experiment artifacts and model registry
├── models/
│   └── by_ticker/              # Human-readable model directory (run scripts/link_models.py)
├── scripts/
│   ├── build_model_index.py    # Build ticker → model folder mapping
│   ├── link_models.py          # Create readable model dirs
│   ├── tune.py                 # Optuna hyperparameter tuning
│   └── seed_metrics.py         # Seed Prometheus metrics
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana-dashboard.json
│   └── grafana/                # Auto-provisioned Grafana datasources + dashboards
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
├── start.bat                   # Full launcher menu (train/retrain/tune/start)
├── run.bat                     # Quick Docker start (no retraining)
├── predict.bat                 # Quick prediction wrapper
├── cli.py                      # Unified CLI entry point
└── .env                        # API keys (never committed)
```

## 🚀 Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Python 3.11 (only needed for local training)

### 1. Clone the repo
```bash
git clone https://github.com/your-username/indonesian-stock-mlops.git
cd indonesian-stock-mlops
```

### 2. Set up environment variables
```bash
# Copy and edit .env (never commit this file)
# Add keys for optional features:
#   FRED_API_KEY    → macro/commodity data (https://fred.stlouisfed.org)
#   NEWSAPI_KEY     → news sentiment scores (https://newsapi.org)
```

### 3. Start the stack

**Windows — double-click or run:**
```bat
start.bat          # Full menu: train/retrain/tune/start
run.bat            # Quick start: just Docker + wait
predict.bat BBCA.JK  # Single prediction
```

**Linux / Mac:**
```bash
docker compose up -d
python cli.py status                    # Check if API is running
python cli.py predict BBCA.JK           # Single prediction
python cli.py predict --all             # All 45 tickers
python cli.py backtest                  # Full backtest (0.50 threshold)
python cli.py sentiment BBCA.JK         # News sentiment score
python cli.py list                      # List all tickers by sector
```

Training all 45 tickers takes ~15-20 minutes on first run (CPU). Models are cached after training.

---

## 🌐 Service URLs

| Service | URL | Notes |
|---|---|---|
| Prediction API | `http://127.0.0.1:8000` | Use 127.0.0.1 not localhost on Windows |
| API Docs | `http://127.0.0.1:8000/docs` | Interactive Swagger UI |
| Cache Status | `http://127.0.0.1:8000/cache` | Prediction cache coverage |
| MLflow UI | `http://localhost:5000` | Experiment tracking |
| Grafana | `http://localhost:3000` | admin/admin |
| Prometheus | `http://localhost:9090` | Raw metrics |

---

## 📡 API Usage

### Prediction (with sentiment overlay)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "BBCA.JK"}'
```

**Response (v2.2+):**
```json
{
  "ticker": "BBCA.JK",
  "prediction": 0,
  "probability_up": 0.4789,
  "signal": "SELL",
  "sentiment_score": 0.1523,
  "probability_adjusted": 0.4941,
  "signal_adjusted": "SELL"
}
```

| Field | Description |
|---|---|
| `prediction` | 0 = SELL, 1 = BUY |
| `probability_up` | Model confidence (0-1) |
| `sentiment_score` | VADER news sentiment (-1 to +1, 0 if unavailable) |
| `probability_adjusted` | Model confidence ±10% based on sentiment |
| `signal_adjusted` | Signal after sentiment overlay |

### Batch Prediction
```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["BBCA.JK", "BBRI.JK", "TLKM.JK"]}'
```

### List tickers
```bash
curl http://127.0.0.1:8000/tickers
```

---

## 🖥️ CLI Usage

```bash
# Prediction
python cli.py predict BBCA.JK                   # Via API
python cli.py predict BBCA.JK --local           # Local model
python cli.py predict --all --json              # All 45, JSON output

# Backtest (walk-forward simulation)
python cli.py backtest                           # Threshold 0.50
python cli.py backtest --threshold 0.65          # Higher confidence

# Sentiment
python cli.py sentiment BBCA.JK                 # Single ticker
python cli.py sentiment --all                    # All 45

# Training
python cli.py train                              # Train all models
python cli.py train --ticker BBCA.JK --tune      # Tune single ticker

# Status
python cli.py status                             # Check API health
python cli.py list                               # List all tickers
```

---

## 📊 Data Sources

| Source | Data | Update |
|---|---|---|
| Yahoo Finance | OHLCV, PE/PB, market cap, ROE, USD/IDR | Daily |
| FRED API | WTI oil, gold, coal, nickel, VIX, US 10Y | Daily |
| Bank Indonesia | BI 7-Day Reverse Repo Rate | Per meeting |
| NewsAPI | English + Indonesian news headlines | Per request |
| VADER | Lexicon-based sentiment on headlines | Per request |

---

## 🧠 Model Details

- **Algorithm** — XGBoost Classifier (per-ticker, 45 models)
- **Target** — Binary: 1 (next day close > today) / 0 (otherwise)
- **Validation** — Purged TimeSeriesSplit (5 folds, 10-day gap)
- **Calibration** — Isotonic regression for probability calibration
- **Features** — 172+ after pruning (correlation >0.95 + low-variance filter)
- **Tuning** — Optuna (20 trials, top-10 tickers via `--tune`)
- **Sentiment overlay** — ±10% adjustment from VADER news sentiment

## 🔧 Configuration (`src/config.py`)

| Flag | Default | Description |
|---|---|---|
| `ta_indicators` | true | 75 TA indicators from `ta` library |
| `enhanced_mas` | true | 15 enhanced moving average features |
| `ict_suite` | true | 25 ICT Smart Money Concepts features |
| `volume_profile` | true | 20 intraday volume profile features |
| `market_context` | true | 15 market/cross-stock features |
| `news_sentiment` | true | VADER sentiment from NewsAPI |
| `fred_macro` | true | FRED macro indicators |
| `bi_rate` | true | Bank Indonesia rate scraping |
| `google_trends` | true | Google Trends sentiment |

---

## 📸 Screenshots

![Grafana Dashboard](assets/grafana1.png)
![Grafana Dashboard](assets/grafana2.png)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
