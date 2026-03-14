@echo off
setlocal enabledelayedexpansion
title Indonesian Stock MLOps
color 0A

echo =========================================
echo       INDONESIAN STOCK MLOPS STACK
echo =========================================
echo.

:: Ask the user what they want to do
echo [1] Start Server Only (Fast)
echo [2] Start Server and Retrain Models (Takes 5-10 mins)
echo.
set /p user_choice="Enter 1 or 2: "

echo.
echo =========================================
echo [1/4] Cleaning up old containers...
docker-compose down

echo.
echo [2/4] Starting Docker services...
docker-compose up -d

echo.
echo [3/4] Waiting for services to boot...
:WAIT_API
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 2 /nobreak >nul
    goto WAIT_API
)
echo API is ready!

:WAIT_MLFLOW
curl -s http://localhost:5000 >nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 2 /nobreak >nul
    goto WAIT_MLFLOW
)
echo MLflow is ready!

:: Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo.
echo [4/4] Pipeline Execution
if "%user_choice%"=="2" (
    echo Step 1: Fetching stock data...
    python src/ingest.py
    
    echo Step 2: Engineering features...
    python src/features.py
    
    echo Step 3: Training models...
    python src/train.py
) else (
    echo Skipping training. Using existing models.
)

echo.
echo Step 4: Seeding today's metrics to Grafana...
python scripts/seed_metrics.py

echo.
echo =========================================
echo         ALL SYSTEMS ARE LIVE!
echo =========================================
echo.
echo [ FastAPI ]   http://localhost:8000/docs
echo [ MLflow  ]   http://localhost:5000
echo [ Grafana ]   http://localhost:3000
echo.
echo Press any key to close this window.
pause >nul