@echo off
setlocal enabledelayedexpansion
title Indonesian Stock MLOps v2.2
color 0A

echo =========================================
echo    INDONESIAN STOCK MLOPS v2.2
echo =========================================
echo.
echo [1] Start Server Only (fast - use cached models)
echo [2] Quick Retrain (CI mode - skip intraday/Trends)
echo [3] Full Retrain (all features + volume profile)
echo [4] Tune Models (Optuna hyperparameter tuning)
echo [5] Rebuild Docker + Start (fresh install)
echo [6] Predict a Ticker (quick prediction)
echo [7] Daily Pipeline (ingest ^> features ^> seed)
echo [8] Weekly Retrain (daily ^> train ^> seed)
echo.
set /p user_choice="Enter 1-8: "

echo.
echo [1/4] Cleaning up old containers...
docker compose down 2>nul
docker compose down api 2>nul

if "%user_choice%"=="5" (
    echo.
    echo [Rebuilding Docker image...]
    docker compose build --no-cache api
)

echo.
echo [2/4] Starting Docker services...
docker compose up -d

echo.
echo [3/4] Waiting for services to boot...

:: Use Python to health-check (more reliable than curl on Windows)
:WAIT_API
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" >nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 3 /nobreak >nul
    goto WAIT_API
)
echo API is ready!

:WAIT_MLFLOW
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000', timeout=3)" >nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 3 /nobreak >nul
    goto WAIT_MLFLOW
)
echo MLflow is ready!

:: Activate virtual environment if it exists (for local training)
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo.
echo [4/4] Pipeline Execution
if "%user_choice%"=="2" (
    echo === QUICK RETRAIN (CI mode) ===
    echo Step 1: Fetching stock data + IHSG...
    python src/ingest.py

    echo Step 2: Engineering features (CI mode)...
    python -c "import sys; sys.path.insert(0,'src'); from features.pipeline import build_feature_set; from config import TICKERS, DATA_PROCESSED_CSV_PATH; build_feature_set('data/raw/stocks.csv', DATA_PROCESSED_CSV_PATH, TICKERS, mode='ci')"

    echo Step 3: Training models...
    python src/train.py

    echo Step 4: Building model index...
    python scripts/build_model_index.py
)

if "%user_choice%"=="3" (
    echo === FULL RETRAIN (All features) ===
    echo This fetches intraday data for 45 tickers (~15-20 min).
    echo Step 1: Fetching stock data + IHSG...
    python src/ingest.py

    echo Step 2: Engineering features (full mode)...
    python -c "import sys; sys.path.insert(0,'src'); from features.pipeline import build_feature_set; from config import TICKERS, DATA_PROCESSED_PATH; build_feature_set('data/raw/stocks.csv', DATA_PROCESSED_PATH, TICKERS, mode='full')"

    echo Step 3: Training models...
    python src/train.py

    echo Step 4: Building model index...
    python scripts/build_model_index.py
)

if "%user_choice%"=="4" (
    echo === OPTUNA HYPERPARAMETER TUNING ===
    echo This will tune top-10 tickers (~30-60 min)...
    python src/train.py --tune
    echo Rebuilding model index...
    python scripts/build_model_index.py
)

if "%user_choice%"=="6" (
    echo === PREDICT A TICKER ===
    set /p pred_ticker="Enter ticker (e.g. BBCA.JK): "
    if exist "venv\Scripts\python.exe" (
        venv\Scripts\python.exe cli.py predict !pred_ticker!
    ) else (
        python cli.py predict !pred_ticker!
    )
)

if "%user_choice%"=="1" (
    echo Skipping training. Using existing models.
)

echo.
echo Step 4: Seeding metrics to Prometheus/Grafana (parallel async)...
python scripts/seed_metrics.py

echo Step 5: Pre-warming prediction cache for instant first request...
python -c "import asyncio, sys; sys.path.insert(0,'scripts'); from seed_metrics import seed_all; asyncio.run(seed_all(concurrency=4))"
goto END

:: ── OPTIONS 7 & 8 ──────────────────────────────────────────────────────
if "%user_choice%"=="7" (
    echo.
    echo === DAILY PIPELINE ===
    echo.
    echo Step 1: Fetching stock data + IHSG...
    python src/ingest.py || goto FAIL
    echo Step 2: Engineering features (CI mode)...
    python scripts/ci_features.py || goto FAIL
    echo Step 3: Building model index...
    python scripts/build_model_index.py
    echo Step 4: Seeding Grafana...
    python scripts/seed_metrics.py || goto FAIL
    echo.
    echo === DAILY PIPELINE COMPLETE! ===
    goto END
)

if "%user_choice%"=="8" (
    echo.
    echo === WEEKLY RETRAIN ===
    echo.
    echo Step 1: Fetching stock data + IHSG...
    python src/ingest.py || goto FAIL
    echo Step 2: Engineering features (CI mode)...
    python scripts/ci_features.py || goto FAIL
    echo Step 3: Training 45 models (parallel)...
    python src/train.py --parallel
    echo Step 4: Building model index...
    python scripts/build_model_index.py || goto FAIL
    echo Step 5: Restarting API...
    docker compose up -d --force-recreate api
    :WAIT_API2
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" >nul 2>&1
    if !errorlevel! neq 0 (
        timeout /t 3 /nobreak >nul
        goto WAIT_API2
    )
    echo API ready!
    echo Step 6: Seeding Grafana...
    python scripts/seed_metrics.py || goto FAIL
    echo.
    echo === WEEKLY RETRAIN COMPLETE! ===
    goto END
)

:FAIL
echo.
echo [ERROR] Pipeline step failed! Check output above.
pause
exit /b 1

:END
echo.
echo =========================================
echo         ALL SYSTEMS ARE LIVE!
echo =========================================
echo.
echo [ FastAPI    ]  http://127.0.0.1:8000/docs
echo [ MLflow     ]  http://localhost:5000
echo [ Prometheus ]  http://localhost:9090
echo [ Grafana    ]  http://localhost:3000 (admin/admin)
echo.
echo NOTE: Use 127.0.0.1 (not localhost) for API calls
echo       due to Docker Desktop IPv6 on Windows.
echo.
echo Press any key to close this window.
pause >nul