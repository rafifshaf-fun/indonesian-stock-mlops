import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return df

def engineer_features_for_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # Extract single ticker from multi-ticker dataframe
    ticker_df = df[ticker].copy()
    ticker_df = dropna(ticker_df)

    # Add all technical indicators (momentum, trend, volatility, volume)
    ticker_df = add_all_ta_features(
        ticker_df,
        open="Open", high="High", low="Low",
        close="Close", volume="Volume",
        fillna=True
    )

    # Create target: 1 if next day's close is higher, 0 if lower
    ticker_df["target"] = (ticker_df["Close"].shift(-1) > ticker_df["Close"]).astype(int)

    # Drop last row (no target available)
    ticker_df = ticker_df.iloc[:-1]
    ticker_df["ticker"] = ticker

    return ticker_df

def build_feature_set(raw_path: str, output_path: str, tickers: list):
    df = load_data(raw_path)
    all_features = []

    for ticker in tickers:
        print(f"Processing {ticker}...")
        try:
            featured = engineer_features_for_ticker(df, ticker)
            all_features.append(featured)
        except Exception as e:
            print(f"Skipping {ticker}: {e}")

    combined = pd.concat(all_features)
    combined.to_csv(output_path)
    print(f"Feature set saved to {output_path}")

if __name__ == "__main__":
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
    build_feature_set(
        raw_path="data/raw/stocks.csv",
        output_path="data/processed/features.csv",
        tickers=TICKERS
    )
