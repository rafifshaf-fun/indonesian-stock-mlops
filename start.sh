#!/bin/bash
echo "🚀 Starting Indonesian Stock MLOps Stack..."

# Pull latest code + mlflow.db from GitHub
echo "📥 Pulling latest from GitHub..."
git pull

# Activate venv (only needed for training scripts, not serving)
source venv/Scripts/activate

# Boot everything
echo "🐳 Starting all services..."
docker-compose up -d

# Wait for services to initialize
echo "⏳ Waiting for services..."
sleep 8

# Seed Prometheus metrics
echo "🌱 Seeding metrics..."
python scripts/seed_metrics.py

echo ""
echo "✅ All services running!"
echo "   MLflow  → http://localhost:5000"
echo "   API     → http://localhost:8000"
echo "   Grafana → http://localhost:3000"
echo "   Prometheus → http://localhost:9090"