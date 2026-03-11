@echo off
echo 🚀 Starting Indonesian Stock MLOps Stack...

echo 📥 Pulling latest from GitHub...
git pull

echo 🐳 Starting all services...
docker-compose up -d

echo ⏳ Waiting for API to be ready...
:waitloop
timeout /t 3 /nobreak >nul
curl -s -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo    Still waiting...
    goto waitloop
)
echo    API is ready!

echo 🌱 Seeding metrics...
call venv\Scripts\activate.bat && python scripts/seed_metrics.py

echo.
echo ✅ All services running!
echo    MLflow     → http://localhost:5000
echo    API        → http://localhost:8000
echo    Grafana    → http://localhost:3000
echo    Prometheus → http://localhost:9090
pause