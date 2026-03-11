import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from ta import add_all_ta_features
from ta.utils import dropna
from datetime import datetime, timedelta


# ─────────────────────────────────────────
# SECTION 1: OHLCV loader
# ─────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return df


# ─────────────────────────────────────────
# SECTION 2: NEW — yfinance fundamentals
# ─────────────────────────────────────────

def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental ratios from yfinance .info
    Returns a flat dict of scalar values — forward-filled across all dates
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio":       info.get("trailingPE", np.nan),
            "pb_ratio":       info.get("priceToBook", np.nan),
            "dividend_yield": info.get("dividendYield", np.nan) or 0.0,
            "market_cap":     info.get("marketCap", np.nan),
            "debt_to_equity": info.get("debtToEquity", np.nan),
            "return_on_equity": info.get("returnOnEquity", np.nan),
            "revenue_growth": info.get("revenueGrowth", np.nan),
            "profit_margins": info.get("profitMargins", np.nan),
        }
    except Exception as e:
        print(f"  ⚠️  Fundamentals failed for {ticker}: {e}")
        return {
            "pe_ratio": np.nan, "pb_ratio": np.nan,
            "dividend_yield": 0.0, "market_cap": np.nan,
            "debt_to_equity": np.nan, "return_on_equity": np.nan,
            "revenue_growth": np.nan, "profit_margins": np.nan,
        }


# ─────────────────────────────────────────
# SECTION 3: NEW — USD/IDR macro rate
# ─────────────────────────────────────────

def fetch_usdidr(start: str, end: str) -> pd.Series:
    """
    Fetch USD/IDR exchange rate from Yahoo Finance
    Falls back to empty series if unavailable
    """
    try:
        usdidr = yf.download("IDR=X", start=start, end=end, progress=False)
        series = usdidr["Close"].squeeze()
        series.name = "usdidr_rate"
        return series
    except Exception as e:
        print(f"  ⚠️  USD/IDR fetch failed: {e}")
        return pd.Series(name="usdidr_rate", dtype=float)


# ─────────────────────────────────────────
# SECTION 4: Core feature engineering
# ─────────────────────────────────────────

def engineer_features_for_ticker(
    df: pd.DataFrame,
    ticker: str,
    fundamentals: dict,
    usdidr: pd.Series
) -> pd.DataFrame:

    # Extract single ticker OHLCV
    ticker_df = df[ticker].copy()
    ticker_df = dropna(ticker_df)

    # Add all technical indicators (RSI, MACD, Bollinger, etc.)
    ticker_df = add_all_ta_features(
        ticker_df,
        open="Open", high="High", low="Low",
        close="Close", volume="Volume",
        fillna=True
    )

    # ── Attach fundamentals (same value across all rows — forward-filled) ──
    for col, val in fundamentals.items():
        ticker_df[col] = val

    # ── Attach USD/IDR rate aligned by date ──
    if not usdidr.empty:
        ticker_df = ticker_df.join(usdidr, how="left")
        ticker_df["usdidr_rate"] = ticker_df["usdidr_rate"].ffill().bfill()
    else:
        ticker_df["usdidr_rate"] = np.nan

    # ── Target: 1 if next day close is higher ──
    ticker_df["target"] = (ticker_df["Close"].shift(-1) > ticker_df["Close"]).astype(int)
    ticker_df = ticker_df.iloc[:-1]
    ticker_df["ticker"] = ticker

    return ticker_df


# ─────────────────────────────────────────
# SECTION 5: Build full feature set
# ─────────────────────────────────────────

def build_feature_set(raw_path: str, output_path: str, tickers: list):
    df = load_data(raw_path)

    # Determine date range for macro data
    start = str(df.index.min().date())
    end   = str(df.index.max().date())

    print(f"📅 Date range: {start} → {end}")
    print("💱 Fetching USD/IDR rate...")
    usdidr = fetch_usdidr(start, end)

    all_features = []

    for ticker in tickers:
        print(f"⚙️  Processing {ticker}...")
        try:
            print(f"  📊 Fetching fundamentals...")
            fundamentals = fetch_fundamentals(ticker)

            featured = engineer_features_for_ticker(df, ticker, fundamentals, usdidr)
            all_features.append(featured)
            print(f"  ✅ {ticker} done — {featured.shape[1]} features, {len(featured)} rows")
        except Exception as e:
            print(f"  ❌ Skipping {ticker}: {e}")

    combined = pd.concat(all_features)
    combined.to_csv(output_path)
    print(f"\n✅ Feature set saved → {output_path}")
    print(f"   Shape: {combined.shape}")
    print(f"   Features added: pe_ratio, pb_ratio, dividend_yield, market_cap,")
    print(f"                   debt_to_equity, return_on_equity, revenue_growth,")
    print(f"                   profit_margins, usdidr_rate")


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
