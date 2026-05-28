@echo off
title Quick Prediction Test - 10 Tickers
color 0B

echo =========================================
echo   QUICK PREDICTION TEST (10 Tickers)
echo =========================================
echo.
echo Top 10 LQ45 stocks - ~5 minutes.
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

python scripts\quick_predict.py

echo.
echo =========================================
echo   DONE! Check Grafana at:
echo   http://localhost:3000
echo =========================================
pause
