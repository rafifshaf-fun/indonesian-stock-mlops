"""Seed Prometheus metrics via parallel async requests with retry.

Turns a ~11-minute sequential loop into a ~20-second parallel burst.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
import aiohttp
from config import TICKERS

API_URL = "http://127.0.0.1:8000/predict"
MAX_RETRIES = 3
RETRY_DELAY = 2


async def predict_one(session: aiohttp.ClientSession, ticker: str, sem: asyncio.Semaphore) -> dict:
    """Predict with exponential backoff retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with sem:
                async with session.post(API_URL, json={"ticker": ticker}, timeout=120) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"[OK] {ticker}: {data.get('signal','?')} "
                              f"({data.get('probability_up',0)*100:.2f}%)")
                        return data
                    else:
                        print(f"[FAIL] {ticker}: HTTP {resp.status} (attempt {attempt})")
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] {ticker} (attempt {attempt})")
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")
        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY * attempt)
    return {}


async def seed_all(concurrency: int = 8):
    """Run all 45 predictions in parallel with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [predict_one(session, t, sem) for t in TICKERS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = sum(1 for r in results if isinstance(r, dict) and "signal" in r)
    print(f"\nDone: {ok}/{len(TICKERS)} predictions seeded to Grafana")
    return results


if __name__ == "__main__":
    print(f"Seeding {len(TICKERS)} tickers in parallel...")
    asyncio.run(seed_all())