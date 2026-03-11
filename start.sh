#!/bin/bash
echo "🚀 Starting Indonesian Stock MLOps Stack..."

# Pull latest mlruns + mlflow.db from GitHub
echo "📥 Pulling latest data from GitHub..."
git pull

# Activate virtual environment
source venv/Scripts/activate

# Start Docker services (Prometheus + Grafana)
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 5

# Seed Prometheus metrics
echo "🌱 Seeding metrics..."
python scripts/seed_metrics.py

# Start MLflow with SQLite backend
echo "📊 Starting MLflow UI..."
mlflow ui --backend-store-uri sqlite:///mlflow.db &

# Start FastAPI server (foreground — keeps terminal alive)
echo "⚡ Starting FastAPI server..."
uvicorn serve:app --reload