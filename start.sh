#!/bin/bash
echo "🚀 Starting Indonesian Stock MLOps Stack..."

# Pull latest mlruns from GitHub
echo "📥 Pulling latest mlruns from GitHub..."
git pull

# Activate virtual environment
source venv/Scripts/activate

# Start Docker services (Prometheus + Grafana)
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready before seeding
echo "⏳ Waiting for services to be ready..."
sleep 5

# Seed Prometheus metrics
echo "🌱 Seeding metrics..."
python scripts/seed_metrics.py

# Start MLflow in background
echo "📊 Starting MLflow UI..."
mlflow ui --backend-store-uri ./mlruns &

# Start FastAPI server (foreground)
echo "⚡ Starting FastAPI server..."
uvicorn serve:app --reload
