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

- 🤖 **ML Model** — XGBoost classifier trained with TimeSeriesSplit cross-validation per ticker
- 📊 **200+ Features** — Technical indicators, fundamentals, macro data, FX rates, sentiment
- 🏦 **Multi-source Data** — Yahoo Finance, FRED API, Bank Indonesia, Google Trends
- 🔁 **Auto-retrain on Boot** — Fresh model every time you start the stack via `start.bat`
- 📈 **MLflow Tracking** — Full experiment history, run comparison, model registry
- 📡 **Real-time Monitoring** — Grafana dashboard with confidence scores, signal counts, API latency
- 🐳 **Fully Dockerized** — One command to spin up the entire stack
- ✅ **CI/CD Pipeline** — GitHub Actions runs tests and builds on every push

---

## 🗂️ Project Structure

```
├── src/
│   ├── ingest.py          # Fetch raw OHLCV data from yfinance
│   ├── features.py        # Feature engineering (TA + macro + trends)
│   ├── train.py           # XGBoost training + MLflow logging
│   ├── serve.py           # FastAPI prediction server
│   └── seed_metrics.py    # Seed Prometheus metrics on startup
├── data/
│   ├── raw/               # Raw OHLCV CSVs
│   └── processed/         # Engineered feature CSVs
├── mlruns/                # MLflow experiment artifacts
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI pipeline
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── start.bat              # Windows one-click startup
└── .env                   # API keys (never committed)
```

## 🚀 Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Windows OS (for `start.bat`); Linux/Mac users run commands manually

### 1. Clone the repo
```bash
git clone https://github.com/your-username/indonesian-stock-mlops.git
cd indonesian-stock-mlops
```

### 2. Set up environment variables
```bash
# Create .env file (never commit this)
cp .env.example .env
```
Edit `.env`:
```
FRED_API_KEY=your_fred_api_key_here
```
Get a free FRED API key at → https://fred.stlouisfed.org/docs/api/api_key.html

### 3. Start the stack
```bash
# Windows — double-click or run:
start.bat

# Linux / Mac
docker-compose up -d --build
docker-compose exec api python src/ingest.py
docker-compose exec api python src/features.py
docker-compose exec api python src/train.py
docker-compose exec api python src/seed_metrics.py
```

Training all 45 tickers takes ~5–10 minutes on first run.

---

## 🌐 Service URLs

| Service | URL | Description |
|---|---|---|
| Prediction API | http://localhost:8000 | FastAPI REST API |
| API Docs | http://localhost:8000/docs | Interactive Swagger UI |
| MLflow UI | http://localhost:5000 | Experiment tracking & model registry |
| Grafana | http://localhost:3000 | Real-time monitoring dashboard |
| Prometheus | http://localhost:9090 | Raw metrics scraper |

---

## 📡 API Usage

### Get a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "BBCA.JK"}'
```

**Response:**
```json
{
  "ticker": "BBCA.JK",
  "prediction": 1,
  "probability_up": 0.6371,
  "signal": "BUY"
}
```

### List available tickers
```bash
curl http://localhost:8000/tickers
```

---

## 📊 Data Sources

| Source | Data | Update Frequency |
|---|---|---|
| Yahoo Finance | OHLCV prices, PE/PB ratio, market cap, ROE | Daily |
| Yahoo Finance | USD/IDR exchange rate | Daily |
| FRED API | WTI oil, gold, coal, nickel, VIX, Fed rate, US 10Y, BI rate proxy | Daily/Monthly |
| Bank Indonesia | BI 7-Day Reverse Repo Rate | Per RDG meeting |
| Google Trends | Search interest per company (geo: ID) | Weekly |

---

## 🧠 Model Details

- **Algorithm** — XGBoost Classifier
- **Target** — Binary: 1 (next day close > today) / 0 (next day close ≤ today)
- **Validation** — TimeSeriesSplit (5 folds) to prevent data leakage
- **Features** — 210+ including all `ta` library indicators + macro + fundamental + sentiment
- **Tracked Metrics** — avg_accuracy, avg_f1, avg_roc_auc per ticker per run

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `FRED_API_KEY` | — | Required for macro + commodity features |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server address |

---

## 🛠️ Rebuilding After Dependency Changes

```bat
:: Windows
start.bat build

:: Linux/Mac
docker-compose up -d --build
```

---

## 📸 Screenshots

> MLflow experiment comparison, Grafana dashboard, and API Swagger UI screenshots here.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
