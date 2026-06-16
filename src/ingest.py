import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from config import (
    TICKERS, IHSG_TICKER, DATA_RAW_PATH, START_DATE,
    FEATURE_FLAGS, CACHE_CONFIG, get_logger,
)

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

DATA_PATH = DATA_RAW_PATH
GLOBAL_START_DATE = START_DATE

def get_last_date(path: str) -> str:
    """Reads the existing dataset to find the last fetched date."""
    if not os.path.exists(path):
        return GLOBAL_START_DATE
        
    try:
        # Load with MultiIndex headers (yfinance format)
        df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        if df.empty:
            return GLOBAL_START_DATE
        
        last_date = df.index[-1]
        return last_date.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"⚠️ Could not read existing data ({e}). Defaulting to full history.")
        return GLOBAL_START_DATE

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Runs quality checks and warns about bad data."""
    if df.empty:
        return df
        
    # Group_by="ticker" means level 0 is Ticker, level 1 is OHLCV
    tickers_downloaded = set([col[0] for col in df.columns if isinstance(col, tuple)])
    
    for ticker in tickers_downloaded:
        try:
            ticker_df = df[ticker].copy()
            
            # Check 1: Zero Volume — fill from prior day (weekend/holiday/API gap)
            if 'Volume' in ticker_df.columns:
                zero_vol = ticker_df[ticker_df['Volume'] == 0]
                if not zero_vol.empty:
                    vol_fixed = ticker_df['Volume'].replace(0, np.nan).ffill().fillna(0)
                    df.loc[:, (ticker, 'Volume')] = vol_fixed.values
                    print(f" ⚠️  Warning: {ticker} had {len(zero_vol)} zero-volume days — filled from prior day.")
            
            # Check 2: Unrealistic Price Drops (Usually an unadjusted stock split)
            if 'Close' in ticker_df.columns:
                returns = ticker_df['Close'].pct_change()
                extreme_drops = returns[returns < -0.50]  # Drops > 50% in one day
                if not extreme_drops.empty:
                    bad_dates = extreme_drops.index.strftime('%Y-%m-%d').tolist()
                    print(f" 🚨 CRITICAL: {ticker} dropped >50% on {bad_dates}. Likely an unadjusted stock split!")
                    
        except KeyError:
            continue
            
    return df

def fetch_and_update(tickers: list, path: str):
    # yfinance 'end' date is exclusive, so we add 1 day to ensure we get today's data
    end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = get_last_date(path)
    
    print(f"Fetching data from {start_date} to {end_date}...")
        
    df_new = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        group_by="ticker", 
        auto_adjust=True, 
        progress=False
    )
    
    if df_new.empty:
        print(" ℹ️ No new data found on Yahoo Finance.")
        return

    print("Validating downloaded data...")
    df_new = validate_data(df_new)
    
    # Merge with existing data
    if os.path.exists(path) and start_date != GLOBAL_START_DATE:
        df_old = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        # Combine old and new, keeping the newest rows in case of overlapping dates
        df_combined = pd.concat([df_old, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined.sort_index(inplace=True)
        print(f" 🔄 Incremental update: Added {len(df_combined) - len(df_old)} new days.")
    else:
        df_combined = df_new
        print(f" ⬇️ Full download: {len(df_combined)} days.")
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_combined.to_csv(path)
    print(f" ✅ Saved {len(df_combined)} total rows to {path}")

def fetch_ihsg_index():
    """Fetch IHSG (^JKSE) index data for market context features."""
    path = "data/raw/ihsg.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = GLOBAL_START_DATE

    if os.path.exists(path):
        try:
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            if not existing.empty:
                start_date = existing.index[-1].strftime("%Y-%m-%d")
        except Exception:
            pass

    if start_date >= (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d"):
        logger.info("IHSG data already up to date.")
        return

    logger.info("Fetching IHSG (%s) from %s to %s...", IHSG_TICKER, start_date, end_date)
    try:
        df = yf.download(IHSG_TICKER, start=start_date, end=end_date, progress=False)
        if df.empty:
            logger.warning("No IHSG data returned.")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        if os.path.exists(path):
            old = pd.read_csv(path, index_col=0, parse_dates=True)
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)

        df.to_csv(path)
        logger.info("IHSG data saved: %d rows to %s", len(df), path)
    except Exception as e:
        logger.warning("Failed to fetch IHSG: %s", e)

def download_intraday_data(ticker: str, interval: str = "5m", period: str = "5d",
                           cache_dir: str = "data/intraday") -> pd.DataFrame:
    """Download intraday data for a ticker with caching.

    Args:
        ticker: Stock ticker (e.g. 'BBCA.JK')
        interval: Intraday interval ('1m', '5m', '15m', '60m')
        period: Lookback period ('1d', '5d', '1mo')
        cache_dir: Directory to cache intraday parquet files

    Returns:
        DataFrame with intraday OHLCV data, or empty DataFrame on failure.
    """
    import time as _time

    cache_file = os.path.join(cache_dir, f"{ticker}_{interval}.parquet")
    ttl = CACHE_CONFIG.get("intraday_ttl_hours", 4) * 3600

    # Check cache
    if os.path.exists(cache_file):
        try:
            cache_age = _time.time() - os.path.getmtime(cache_file)
            if cache_age < ttl:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    return df
        except Exception:
            pass  # Cache invalid, refetch

    # Download with rate limiting
    _time.sleep(CACHE_CONFIG.get("rate_limit_delay_seconds", 2))
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(cache_file)
        return df
    except Exception as e:
        logger.warning("Intraday download failed for %s: %s", ticker, e)
        return pd.DataFrame()

if __name__ == "__main__":
    print(f"Checking {len(TICKERS)} LQ45 tickers...")
    fetch_and_update(TICKERS, DATA_PATH)

    if FEATURE_FLAGS.get("market_context", True):
        print(f"Fetching IHSG index ({IHSG_TICKER})...")
        fetch_ihsg_index()