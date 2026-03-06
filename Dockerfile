FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY src/ ./src/
COPY mlruns/ ./mlruns/

WORKDIR /app/src

EXPOSE 8000

CMD ["python", "serve.py"]