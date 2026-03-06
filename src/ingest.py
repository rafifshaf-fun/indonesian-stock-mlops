import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# LQ45 constituents 2 Feb 2026)
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

START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

def fetch_stock_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True)
    return df

def save_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Saved {len(df)} rows to {path}")

if __name__ == "__main__":
    print(f"Fetching {len(TICKERS)} LQ45 tickers from {START_DATE} to {END_DATE}...")
    df = fetch_stock_data(TICKERS, START_DATE, END_DATE)
    save_data(df, "data/raw/stocks.csv")
