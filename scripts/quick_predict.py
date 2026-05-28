import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import requests
import time

# Top 10 most liquid LQ45 stocks
TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
    "ASII.JK", "UNVR.JK", "ADRO.JK", "ICBP.JK", "SMGR.JK",
]

print("Quick Prediction Test - 10 Tickers")
print("=" * 50)

for i, ticker in enumerate(TICKERS):
    try:
        r = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"ticker": ticker},
            timeout=120,
        )
        if r.status_code == 200:
            d = r.json()
            print(f"[{i+1:2d}/10] {ticker}: {d['signal']:4s} ({d['probability_up']*100:5.1f}%)")
        else:
            print(f"[{i+1:2d}/10] {ticker}: FAILED (HTTP {r.status_code})")
    except Exception as e:
        print(f"[{i+1:2d}/10] {ticker}: ERROR - {type(e).__name__}")

    if i < len(TICKERS) - 1:
        time.sleep(2)

print("=" * 50)
print("Done! Check Grafana at http://localhost:3000")