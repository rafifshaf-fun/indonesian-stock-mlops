@echo off
title Indonesian Stock MLOps
echo =========================================
echo  Starting Indonesian Stock MLOps Stack
echo =========================================

:: Start all services
docker-compose up -d
echo.
echo [1/4] Services starting...

:: Wait for MLflow to be healthy
echo [2/4] Waiting for MLflow...
:WAIT_MLFLOW
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 3 /nobreak >nul
    goto WAIT_MLFLOW
)
echo        MLflow is ready!

:: Wait for API to be healthy
echo [3/4] Waiting for API...
:WAIT_API
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 3 /nobreak >nul
    goto WAIT_API
)
echo        API is ready!

:: Run the full training pipeline
echo [4/4] Running training pipeline...
echo        Step 1: Fetching stock data...
docker-compose exec -T api python src/ingest.py

echo        Step 2: Engineering features...
docker-compose exec -T api python src/features.py

echo        Step 3: Training models (this takes a few minutes)...
docker-compose exec -T api python src/train.py

echo        Step 4: Seeding metrics...
docker-compose exec -T api python src/seed_metrics.py

echo.
echo =========================================
echo  All done! Stack is live:
echo.
echo   API        http://localhost:8000
echo   API Docs   http://localhost:8000/docs
echo   MLflow     http://localhost:5000
echo   Grafana    http://localhost:3000
echo =========================================
echo.
pause