FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

WORKDIR /app/src

EXPOSE 8000

COPY src/ ./src/
COPY src/config.py ./config.py   

CMD ["python", "serve.py"]
