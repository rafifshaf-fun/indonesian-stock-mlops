import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings("ignore")

#LQ45 2 February 2026
TICKERS = [
    "AADI.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK",
    "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK", "BBCA.JK",
    "BBNI.JK", "BBRI.JK", "BBTN.JK", "BMRI.JK", "BREN.JK",
    "BRIS.JK", "BRPT.JK", "CPIN.JK", "CTRA.JK", "EXCL.JK",
    "GOTO.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INKP.JK",
    "ISAT.JK", "ITMG.JK", "JPFA.JK", "JSMR.JK", "KLBF.JK",
    "MAPA.JK", "MAPI.JK", "MBMA.JK", "MDKA.JK", "MEDC.JK",
    "PGAS.JK", "PGEO.JK", "PTBA.JK", "SIDO.JK", "SMGR.JK",
    "SMRA.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK",
]

DATA_PATH = "data/raw/stocks.csv"
GLOBAL_START_DATE = "2020-01-01"

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
            
            # Check 1: Zero Volume (Suspicious for liquid LQ45 stocks)
            if 'Volume' in ticker_df.columns:
                zero_vol = ticker_df[ticker_df['Volume'] == 0]
                if not zero_vol.empty:
                    print(f" ⚠️ Warning: {ticker} has {len(zero_vol)} recent days with 0 volume.")
            
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

if __name__ == "__main__":
    print(f"Checking {len(TICKERS)} LQ45 tickers...")
    fetch_and_update(TICKERS, DATA_PATH)