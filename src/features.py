import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna
import warnings
warnings.filterwarnings("ignore")

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("⚠️ fredapi not installed — run: pip install fredapi")

try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except ImportError:
    TRENDS_AVAILABLE = False
    print("⚠️ pytrends not installed — run: pip install pytrends")

# ── Configuration ─────────────────────────────────────────────────────────────
FRED_SERIES = {
    "wti_oil": "DCOILWTICO", 
    "gold_price": "GOLDAMGBD228NLBM", 
    "coal_price": "PCOALAUUSDM", 
    "nickel_price": "PNICKUSDM", 
    "fed_rate": "FEDFUNDS", 
    "vix": "VIXCLS", 
    "us10y": "DGS10", 
    "bi_rate_proxy": "IR3TIB01IDM156N", 
}

TICKER_SEARCH_TERMS = {
    "BBCA.JK": "Bank BCA", "BBRI.JK": "Bank BRI", "BMRI.JK": "Bank Mandiri",
    "BBNI.JK": "Bank BNI", "BBTN.JK": "Bank BTN", "TLKM.JK": "Telkom Indonesia",
    "ASII.JK": "Astra International", "GOTO.JK": "GoTo Gojek", "BREN.JK": "Barito Renewables",
    "ADRO.JK": "Adaro Energy", "ANTM.JK": "Aneka Tambang", "PTBA.JK": "Bukit Asam",
    "INDF.JK": "Indofood", "ICBP.JK": "Indofood CBP", "KLBF.JK": "Kalbe Farma",
    "UNVR.JK": "Unilever Indonesia", "UNTR.JK": "United Tractors",
    "AMMN.JK": "Amman Mineral", "INKP.JK": "Indah Kiat Pulp", "ISAT.JK": "Indosat Ooredoo",
    "EXCL.JK": "XL Axiata", "TOWR.JK": "Sarana Menara", "PGAS.JK": "Perusahaan Gas Negara",
    "JSMR.JK": "Jasa Marga", "CPIN.JK": "Charoen Pokphand", "JPFA.JK": "JAPFA Comfeed",
    "SMGR.JK": "Semen Indonesia", "CTRA.JK": "Ciputra Development",
    "SMRA.JK": "Summarecon Agung", "BRIS.JK": "Bank Syariah Indonesia",
    "ARTO.JK": "Bank Jago", "PGEO.JK": "Pertamina Geothermal",
}

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)

def fetch_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio": info.get("trailingPE", np.nan),
            "pb_ratio": info.get("priceToBook", np.nan),
            "dividend_yield": info.get("dividendYield", np.nan) or 0.0,
            "market_cap": info.get("marketCap", np.nan),
            "debt_to_equity": info.get("debtToEquity", np.nan),
            "return_on_equity": info.get("returnOnEquity", np.nan),
            "revenue_growth": info.get("revenueGrowth", np.nan),
            "profit_margins": info.get("profitMargins", np.nan),
        }
    except Exception:
        return {k: np.nan for k in ["pe_ratio", "pb_ratio", "dividend_yield", "market_cap", "debt_to_equity", "return_on_equity", "revenue_growth", "profit_margins"]}

def fetch_usdidr(start: str, end: str) -> pd.Series:
    try:
        usdidr = yf.download("IDR=X", start=start, end=end, progress=False)
        series = usdidr["Close"].squeeze()
        series.name = "usdidr_rate"
        # Calculate daily percentage change to remove absolute scaling issues
        ret_series = series.pct_change()
        ret_series.name = "usdidr_return"
        return pd.concat([series, ret_series], axis=1)
    except Exception:
        return pd.DataFrame(columns=["usdidr_rate", "usdidr_return"])

def fetch_fred_macro(start: str, end: str) -> pd.DataFrame:
    api_key = os.environ.get("FRED_API_KEY")
    if not FRED_AVAILABLE or not api_key:
        return pd.DataFrame()
    try:
        fred = Fred(api_key=api_key)
        frames = {}
        for name, series_id in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start, observation_end=end)
                s.name = name
                frames[name] = s
            except Exception:
                pass
        
        if not frames:
            return pd.DataFrame()
            
        df = pd.concat(frames.values(), axis=1)
        df.index = pd.to_datetime(df.index)
        
        # PREVENT LOOKAHEAD BIAS: 
        # Shift all macro indicators forward by 1 day to ensure we only trade on 
        # data that was publicly available yesterday.
        df = df.shift(1).resample("D").last().ffill().bfill()
        return df
    except Exception:
        return pd.DataFrame()

def fetch_bi_rate(start: str, end: str) -> pd.Series:
    import requests
    try:
        url = "https://www.bi.go.id/en/statistik/indikator/bi-rate.aspx"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(resp.text)
        for table in tables:
            if table.shape[1] >= 12:
                table.columns = [str(c) for c in table.columns]
                melted = table.melt(id_vars=table.columns[0], var_name="Month", value_name="bi_rate")
                melted.columns = ["Year", "Month", "bi_rate"]
                melted = melted.dropna(subset=["bi_rate"])
                melted["date"] = pd.to_datetime(melted["Year"].astype(str) + "-" + melted["Month"].astype(str) + "-01", errors="coerce")
                melted = melted.dropna(subset=["date"])
                
                series = melted.set_index("date")["bi_rate"]
                series = pd.to_numeric(series, errors="coerce").dropna()
                
                # PREVENT LOOKAHEAD BIAS: Shift monthly release to next month
                # Example: March 1st data is officially released ~March 20th. We delay it safely.
                series.index = series.index + pd.DateOffset(months=1)
                series = series.resample("D").last().ffill().bfill()
                series.name = "bi_rate_official"
                return series
    except Exception:
        pass
    return pd.Series(name="bi_rate_official", dtype=float)

def fetch_google_trends(ticker: str, start: str, end: str) -> pd.Series:
    if not TRENDS_AVAILABLE:
        return pd.Series(name="google_trend", dtype=float)
    search_term = TICKER_SEARCH_TERMS.get(ticker, ticker.replace(".JK", ""))
    try:
        pt = TrendReq(hl="id-ID", tz=420, timeout=(10, 25))
        pt.build_payload([search_term], cat=0, timeframe=f"{start} {end}", geo="ID")
        df = pt.interest_over_time()
        if df.empty:
            return pd.Series(name="google_trend", dtype=float)
            
        series = df[search_term]
        # Shift Google Trends by 1 week since weekly data is reported retroactively
        series.index = series.index + pd.DateOffset(days=7)
        series = series.resample("D").last().ffill().bfill()
        series.name = "google_trend"
        time.sleep(2)
        return series
    except Exception:
        return pd.Series(name="google_trend", dtype=float)

def engineer_features_for_ticker(df, ticker, fundamentals, usdidr, fred_macro, bi_rate) -> pd.DataFrame:
    ticker_df = df[ticker].copy()
    ticker_df = dropna(ticker_df)

    if len(ticker_df) < 50:
        return None

    ticker_df = add_all_ta_features(
        ticker_df, open="Open", high="High", low="Low",
        close="Close", volume="Volume", fillna=True
    )

    # ── ADVANCED: Relative Price Features ─────────────────────────────────────
    # Machine learning algorithms learn better from relative distance than raw numbers
    ticker_df["ret_1d"] = ticker_df["Close"].pct_change()
    ticker_df["ret_5d"] = ticker_df["Close"].pct_change(periods=5)
    
    # Distance from Moving Averages
    if "trend_sma_fast" in ticker_df.columns:
        ticker_df["dist_from_sma50"] = ticker_df["Close"] / ticker_df["trend_sma_fast"] - 1
    if "trend_sma_slow" in ticker_df.columns:
        ticker_df["dist_from_sma200"] = ticker_df["Close"] / ticker_df["trend_sma_slow"] - 1

    # ── External Data Merging ─────────────────────────────────────────────────
    for col, val in fundamentals.items():
        ticker_df[col] = val

    if not usdidr.empty:
        ticker_df = ticker_df.join(usdidr, how="left")
        ticker_df["usdidr_rate"] = ticker_df["usdidr_rate"].ffill().bfill()
        ticker_df["usdidr_return"] = ticker_df["usdidr_return"].ffill().bfill().fillna(0)
    else:
        ticker_df["usdidr_rate"] = np.nan
        ticker_df["usdidr_return"] = 0

    if not fred_macro.empty:
        ticker_df = ticker_df.join(fred_macro, how="left")
        for col in fred_macro.columns:
            ticker_df[col] = ticker_df[col].ffill().bfill()

    if not bi_rate.empty:
        ticker_df = ticker_df.join(bi_rate, how="left")
        ticker_df["bi_rate_official"] = ticker_df["bi_rate_official"].ffill().bfill()
    else:
        ticker_df["bi_rate_official"] = np.nan

    # ── Target label ──────────────────────────────────────────────────────────
    ticker_df["target"] = (ticker_df["Close"].shift(-1) > ticker_df["Close"]).astype(int)
    ticker_df = ticker_df.iloc[:-1]
    ticker_df["ticker"] = ticker

    return ticker_df

def build_feature_set(raw_path: str, output_path: str, tickers: list):
    df = load_data(raw_path)
    start = str(df.index.min().date())
    end = str(df.index.max().date())
    print(f"📅 Date range: {start} → {end}")

    print("💱 Fetching USD/IDR rate...")
    usdidr = fetch_usdidr(start, end)

    print("📊 Fetching FRED macro + commodity data...")
    fred_macro = fetch_fred_macro(start, end)

    print("🏦 Fetching BI Rate...")
    bi_rate = fetch_bi_rate(start, end)

    all_features = []
    for ticker in tickers:
        print(f"⚙️ Processing {ticker}...")
        try:
            fundamentals = fetch_fundamentals(ticker)
            featured = engineer_features_for_ticker(df, ticker, fundamentals, usdidr, fred_macro, bi_rate)

            if featured is None or len(featured) < 50:
                continue

            trend = fetch_google_trends(ticker, start, end)
            if not trend.empty:
                trend.index = pd.to_datetime(trend.index)
                featured = featured.join(trend, how="left")
                featured["google_trend"] = featured["google_trend"].ffill().bfill().fillna(0)
            else:
                featured["google_trend"] = 0

            all_features.append(featured)
        except Exception as e:
            print(f" ❌ Skipping {ticker}: {e}")

    combined = pd.concat(all_features)
    combined.to_csv(output_path)
    print(f"\n✅ Saved → {output_path} | Shape: {combined.shape}")

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
