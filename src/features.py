import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna
import warnings
warnings.filterwarnings("ignore")

# ── Optional dependencies (graceful fallback if not installed) ────────────────
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("⚠️  fredapi not installed — run: pip install fredapi")

try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except ImportError:
    TRENDS_AVAILABLE = False
    print("⚠️  pytrends not installed — run: pip install pytrends")

# ── FRED series to fetch ──────────────────────────────────────────────────────
FRED_SERIES = {
    "wti_oil":       "DCOILWTICO",       # WTI Crude Oil — daily
    "gold_price":    "GOLDAMGBD228NLBM", # Gold London Fix — daily
    "coal_price":    "PCOALAUUSDM",      # Australian coal — monthly (for ADRO/PTBA/ITMG)
    "nickel_price":  "PNICKUSDM",        # Nickel price — monthly (for ANTM/INCO)
    "fed_rate":      "FEDFUNDS",         # US Fed Funds Rate — monthly
    "vix":           "VIXCLS",           # VIX Fear Index — daily
    "us10y":         "DGS10",            # US 10Y Treasury yield — daily
    "bi_rate_proxy": "IR3TIB01IDM156N",  # Indonesia 3M interbank rate — monthly (BI rate proxy)
}

# ── Google Trends: ticker → Indonesian search term mapping ───────────────────
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
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return df


def fetch_fundamentals(ticker: str) -> dict:
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
        return {k: np.nan for k in [
            "pe_ratio", "pb_ratio", "dividend_yield", "market_cap",
            "debt_to_equity", "return_on_equity", "revenue_growth", "profit_margins"
        ]}


def fetch_usdidr(start: str, end: str) -> pd.Series:
    try:
        usdidr = yf.download("IDR=X", start=start, end=end, progress=False)
        series = usdidr["Close"].squeeze()
        series.name = "usdidr_rate"
        return series
    except Exception as e:
        print(f"  ⚠️  USD/IDR fetch failed: {e}")
        return pd.Series(name="usdidr_rate", dtype=float)


def fetch_fred_macro(start: str, end: str) -> pd.DataFrame:
    """Fetch macro + commodity features from FRED. Requires FRED_API_KEY env var."""
    api_key = os.environ.get("FRED_API_KEY")
    if not FRED_AVAILABLE or not api_key:
        print("  ⚠️  FRED_API_KEY not set — skipping FRED macro features")
        return pd.DataFrame()
    try:
        fred = Fred(api_key=api_key)
        frames = {}
        for name, series_id in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start, observation_end=end)
                s.name = name
                frames[name] = s
                print(f"  ✅ FRED: {name} ({series_id})")
            except Exception as e:
                print(f"  ⚠️  FRED {series_id} failed: {e}")
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames.values(), axis=1)
        # Resample all series to daily, forward-fill gaps (weekends, monthly series)
        df.index = pd.to_datetime(df.index)
        df = df.resample("D").last().ffill().bfill()
        return df
    except Exception as e:
        print(f"  ⚠️  FRED fetch failed entirely: {e}")
        return pd.DataFrame()


def fetch_bi_rate(start: str, end: str) -> pd.Series:
    """
    Fetch BI Rate (7-Day Reverse Repo) from Bank Indonesia public data.
    Falls back to FRED Indonesia 3M interbank proxy if BI API is unavailable.
    """
    import requests
    try:
        # BI public API — statistical table endpoint
        url = "https://www.bi.go.id/en/statistik/indikator/bi-rate.aspx"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        # Parse tables from the page
        tables = pd.read_html(resp.text)
        # BI rate table: Year x Month grid
        for table in tables:
            if table.shape[1] >= 12:  # Monthly columns
                # Melt from wide to long
                table.columns = [str(c) for c in table.columns]
                melted = table.melt(id_vars=table.columns[0], var_name="Month", value_name="bi_rate")
                melted.columns = ["Year", "Month", "bi_rate"]
                melted = melted.dropna(subset=["bi_rate"])
                melted["date"] = pd.to_datetime(
                    melted["Year"].astype(str) + "-" + melted["Month"].astype(str) + "-01",
                    errors="coerce"
                )
                melted = melted.dropna(subset=["date"])
                series = melted.set_index("date")["bi_rate"]
                series = pd.to_numeric(series, errors="coerce").dropna()
                series = series.resample("D").last().ffill().bfill()
                series.name = "bi_rate_official"
                print("  ✅ BI Rate: fetched from Bank Indonesia")
                return series
    except Exception as e:
        print(f"  ⚠️  BI Rate from BI website failed: {e} — will use FRED proxy")

    # Fallback: FRED IR3TIB01IDM156N (Indonesia 3M interbank — highly correlated to BI Rate)
    api_key = os.environ.get("FRED_API_KEY")
    if FRED_AVAILABLE and api_key:
        try:
            fred = Fred(api_key=api_key)
            s = fred.get_series("IR3TIB01IDM156N", observation_start=start, observation_end=end)
            s = s.resample("D").last().ffill().bfill()
            s.name = "bi_rate_official"
            print("  ✅ BI Rate: using FRED Indonesia interbank proxy")
            return s
        except Exception as e:
            print(f"  ⚠️  FRED BI proxy also failed: {e}")

    return pd.Series(name="bi_rate_official", dtype=float)


def fetch_google_trends(ticker: str, start: str, end: str) -> pd.Series:
    """
    Fetch weekly Google Trends interest score for the company (search term mapped
    from ticker). Returns a daily series via forward-fill. Rate-limited — adds
    a 2s sleep between calls automatically.
    """
    if not TRENDS_AVAILABLE:
        return pd.Series(name="google_trend", dtype=float)

    search_term = TICKER_SEARCH_TERMS.get(ticker, ticker.replace(".JK", ""))
    try:
        pt = TrendReq(hl="id-ID", tz=420, timeout=(10, 25))  # WIB = UTC+7
        pt.build_payload(
            [search_term],
            cat=0,
            timeframe=f"{start} {end}",
            geo="ID"
        )
        df = pt.interest_over_time()
        if df.empty:
            return pd.Series(name="google_trend", dtype=float)
        series = df[search_term].resample("D").last().ffill().bfill()
        series.name = "google_trend"
        time.sleep(2)  # Respect rate limit
        return series
    except Exception as e:
        print(f"  ⚠️  Google Trends failed for {ticker}: {e}")
        return pd.Series(name="google_trend", dtype=float)


def engineer_features_for_ticker(
    df: pd.DataFrame,
    ticker: str,
    fundamentals: dict,
    usdidr: pd.Series,
    fred_macro: pd.DataFrame,
    bi_rate: pd.Series,
) -> pd.DataFrame:

    ticker_df = df[ticker].copy()
    ticker_df = dropna(ticker_df)

    if len(ticker_df) < 50:
        print(f"  ⚠️  Skipping {ticker} — only {len(ticker_df)} rows after dropna")
        return None

    ticker_df = add_all_ta_features(
        ticker_df, open="Open", high="High", low="Low",
        close="Close", volume="Volume", fillna=True
    )

    # ── Yahoo Finance fundamentals ────────────────────────────────────────
    for col, val in fundamentals.items():
        ticker_df[col] = val

    # ── Yahoo Finance USD/IDR ─────────────────────────────────────────────
    if not usdidr.empty:
        ticker_df = ticker_df.join(usdidr, how="left")
        ticker_df["usdidr_rate"] = ticker_df["usdidr_rate"].ffill().bfill()
    else:
        ticker_df["usdidr_rate"] = np.nan

    # ── FRED macro + commodities ──────────────────────────────────────────
    if not fred_macro.empty:
        ticker_df = ticker_df.join(fred_macro, how="left")
        for col in fred_macro.columns:
            ticker_df[col] = ticker_df[col].ffill().bfill()

    # ── Bank Indonesia BI Rate ────────────────────────────────────────────
    if not bi_rate.empty:
        ticker_df = ticker_df.join(bi_rate, how="left")
        ticker_df["bi_rate_official"] = ticker_df["bi_rate_official"].ffill().bfill()
    else:
        ticker_df["bi_rate_official"] = np.nan

    # ── Google Trends (fetched per-ticker inside build_feature_set) ───────
    # Attached externally via join — see build_feature_set()

    # ── Target label ──────────────────────────────────────────────────────
    ticker_df["target"] = (ticker_df["Close"].shift(-1) > ticker_df["Close"]).astype(int)
    ticker_df = ticker_df.iloc[:-1]
    ticker_df["ticker"] = ticker

    return ticker_df


def build_feature_set(raw_path: str, output_path: str, tickers: list):
    df = load_data(raw_path)

    start = str(df.index.min().date())
    end   = str(df.index.max().date())
    print(f"📅 Date range: {start} → {end}")

    print("💱 Fetching USD/IDR rate...")
    usdidr = fetch_usdidr(start, end)

    print("📊 Fetching FRED macro + commodity data...")
    fred_macro = fetch_fred_macro(start, end)

    print("🏦 Fetching BI Rate...")
    bi_rate = fetch_bi_rate(start, end)

    all_features = []
    for ticker in tickers:
        print(f"⚙️  Processing {ticker}...")
        try:
            fundamentals = fetch_fundamentals(ticker)
            featured = engineer_features_for_ticker(
                df, ticker, fundamentals, usdidr, fred_macro, bi_rate
            )

            if featured is None or len(featured) < 50:
                print(f"  ❌ Skipping {ticker} — insufficient data after engineering")
                continue

            # ── Google Trends (per ticker, rate-limited) ──────────────────
            trend = fetch_google_trends(ticker, start, end)
            if not trend.empty:
                trend.index = pd.to_datetime(trend.index)
                featured = featured.join(trend, how="left")
                featured["google_trend"] = featured["google_trend"].ffill().bfill().fillna(0)
            else:
                featured["google_trend"] = 0

            all_features.append(featured)
            print(f"  ✅ Done — {featured.shape[1]} features, {len(featured)} rows")

        except Exception as e:
            print(f"  ❌ Skipping {ticker}: {e}")

    if not all_features:
        raise ValueError("No tickers produced valid features — check your raw data!")

    combined = pd.concat(all_features)
    combined.to_csv(output_path)
    print(f"\n✅ Saved → {output_path} | Shape: {combined.shape}")
    print(f"   New feature columns added:")
    base_cols = {"Open","High","Low","Close","Volume","target","ticker"}
    new_cols = [c for c in combined.columns if c not in base_cols
                and not c.startswith(("momentum_","trend_","volatility_","volume_","others_"))]
    for c in new_cols:
        print(f"   + {c}")


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
