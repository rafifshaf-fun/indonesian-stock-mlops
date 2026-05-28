FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (without DVC — not needed at serving time)
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY monitoring/ ./monitoring/
COPY data/ ./data/

# Create directories that will be mounted/used at runtime
RUN mkdir -p data/raw data/processed data/intraday mlruns

WORKDIR /app

EXPOSE 8000

CMD ["python", "src/serve.py"]
