import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import requests
import time
from config import TICKERS, CACHE_CONFIG

API_URL = "http://127.0.0.1:8000/predict"

print("Fetching latest tickers data against training model and returning signal...")
for ticker in TICKERS:
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict", 
            json={"ticker": ticker}, 
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] {ticker}: {data['signal']} ({data['probability_up']*100:.2f}%)")
        else:
            print(f"[FAIL] {ticker}: HTTP {response.status_code}")
            
    except requests.exceptions.ReadTimeout:
        print(f"[TIMEOUT] {ticker} - Yahoo Finance might be rate-limiting.")
    except Exception as e:
        print(f"[ERROR] {ticker}: {type(e).__name__}")
        
    time.sleep(CACHE_CONFIG.get("rate_limit_delay_seconds", 2))