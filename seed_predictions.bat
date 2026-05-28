@echo off
title Indonesian Stock Predictions - Seeding Grafana
color 0B

echo =========================================
echo   SEEDING PREDICTIONS FOR GRAFANA
echo =========================================
echo.
echo This will run predictions on all 45 LQ45
echo tickers and populate the Grafana dashboard.
echo.
echo Each prediction takes ~25 seconds.
echo Total time: ~20 minutes for all 45 tickers.
echo.
echo Press Ctrl+C to stop early.
echo =========================================
echo.

:: Activate venv if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

python scripts/seed_metrics.py

echo.
echo =========================================
echo   DONE! Check Grafana at:
echo   http://localhost:3000
echo   (login: admin / admin)
echo =========================================
pause
