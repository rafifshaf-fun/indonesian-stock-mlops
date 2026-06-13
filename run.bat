@echo off
title Indonesian Stock MLOps - Quick Start
color 0B

echo =========================================
echo    INDONESIAN STOCK MLOPS - Quick Start
echo =========================================
echo.

echo [1/2] Starting Docker services...
docker compose up -d

echo.
echo [2/2] Waiting for API to be ready...
:WAIT_API
curl.exe -s http://127.0.0.1:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo   Waiting for API... (this can take ~2 min on first start)
    timeout /t 5 /nobreak >nul
    goto WAIT_API
)
echo   API is ready!

echo.
echo =========================================
echo         SERVICES ARE LIVE!
echo =========================================
echo   API        http://127.0.0.1:8000/docs
echo   MLflow     http://localhost:5000
echo   Prometheus http://localhost:9090
echo   Grafana    http://localhost:3000 (admin/admin)
echo   CLI        python cli.py predict BBCA.JK
echo.
echo Press any key to open Grafana...
pause >nul
start http://localhost:3000
