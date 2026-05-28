@echo off
setlocal enabledelayedexpansion
title Indonesian Stock MLOps v2.0
color 0A

echo =========================================
echo    INDONESIAN STOCK MLOPS STACK v2.0
echo =========================================
echo.

:: Ask the user what they want to do
echo [1] Start Server Only (Fast - uses cached models)
echo [2] Quick Retrain (CI mode - skip intraday/Trends)
echo [3] Full Retrain (All features incl. intraday volume)
echo [4] Rebuild Docker + Start
echo.
set /p user_choice="Enter 1-4: "

echo.
echo =========================================
echo [1/4] Cleaning up old containers...
docker compose down

if "%user_choice%"=="4" (
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

    echo Step 2: Engineering features (CI mode - fast)...
    python src/features.py --mode ci --format csv

    echo Step 3: Training models...
    python src/train.py
)

if "%user_choice%"=="3" (
    echo === FULL RETRAIN (All features) ===
    echo This will fetch intraday data for 45 tickers (~15-20 min).
    echo Step 1: Fetching stock data + IHSG...
    python src/ingest.py

    echo Step 2: Engineering features (full mode)...
    python src/features.py --mode full --format parquet

    echo Step 3: Training models...
    python src/train.py
)

if "%user_choice%"=="1" (
    echo Skipping training. Using existing models.
)

echo.
echo Step 4: Seeding metrics to Prometheus/Grafana...
python scripts/seed_metrics.py

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