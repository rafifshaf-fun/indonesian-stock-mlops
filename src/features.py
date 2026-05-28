import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volume import VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings("ignore")

from config import (
    TICKERS, SECTORS, IHSG_TICKER, FRED_SERIES, TICKER_SEARCH_TERMS,
    FEATURE_FLAGS, CACHE_CONFIG, DATA_RAW_PATH, DATA_PROCESSED_PATH,
    DATA_PROCESSED_CSV_PATH, INTRADAY_CACHE_DIR, get_logger,
)

logger = get_logger(__name__)

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logger.warning("fredapi not installed — run: pip install fredapi")

try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except ImportError:
    TRENDS_AVAILABLE = False
    logger.warning("pytrends not installed — run: pip install pytrends")

try:
    import pyarrow  # noqa: F401 — needed for parquet
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("pyarrow not installed — falling back to CSV. Run: pip install pyarrow")

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED FEATURE ENGINEERING FUNCTIONS
# Used by both batch training (build_feature_set) and inference (serve.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Load raw OHLCV data (yfinance MultiIndex format)."""
    return pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)

def compute_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical analysis indicators to OHLCV DataFrame.

    Args:
        df: DataFrame with columns: Open, High, Low, Close, Volume

    Returns:
        DataFrame with TA indicators added. Uses fillna=True to handle
        warm-up periods, then forward/backward fills any remaining NaN.
    """
    if len(df) < 50:
        return df

    df = add_all_ta_features(
        df, open="Open", high="High", low="Low",
        close="Close", volume="Volume", fillna=True
    )
    # Forward/backward fill any remaining NaN from indicator warm-up periods
    df = df.ffill().bfill().fillna(0)
    return df

def compute_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom returns-based features derived from daily OHLCV.

    Adds: ret_1d, ret_5d, dist_from_sma50, dist_from_sma200
    """
    df["ret_1d"] = df["Close"].pct_change()
    df["ret_5d"] = df["Close"].pct_change(periods=5)

    if "trend_sma_fast" in df.columns:
        df["dist_from_sma50"] = df["Close"] / df["trend_sma_fast"] - 1
    if "trend_sma_slow" in df.columns:
        df["dist_from_sma200"] = df["Close"] / df["trend_sma_slow"] - 1

    return df

def inject_macro_features(df: pd.DataFrame, fundamentals: dict,
                          usdidr: pd.DataFrame, fred_macro: pd.DataFrame,
                          bi_rate: pd.Series) -> pd.DataFrame:
    """Inject external macro/fundamental features into the feature DataFrame.

    All external features are prepopulated as NaN and filled from the provided
    data. Missing data gracefully falls back to NaN/0.

    Args:
        df: Feature DataFrame (with datetime index)
        fundamentals: Dict of fundamental metrics from fetch_fundamentals()
        usdidr: USD/IDR rate DataFrame from fetch_usdidr()
        fred_macro: FRED macro DataFrame from fetch_fred_macro()
        bi_rate: BI rate Series from fetch_bi_rate()

    Returns:
        DataFrame with macro columns appended.
    """
    for col, val in (fundamentals or {}).items():
        df[col] = val

    if usdidr is not None and not usdidr.empty and "usdidr_rate" in usdidr.columns:
        df = df.join(usdidr, how="left")
        df["usdidr_rate"] = df["usdidr_rate"].ffill().bfill()
        df["usdidr_return"] = df["usdidr_return"].ffill().bfill().fillna(0)
    else:
        df["usdidr_rate"] = np.nan
        df["usdidr_return"] = 0.0

    if fred_macro is not None and not fred_macro.empty:
        df = df.join(fred_macro, how="left")
        for col in fred_macro.columns:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

    if bi_rate is not None and not bi_rate.empty:
        df = df.join(bi_rate, how="left")
        if "bi_rate_official" in df.columns:
            df["bi_rate_official"] = df["bi_rate_official"].ffill().bfill()
    else:
        df["bi_rate_official"] = np.nan

    return df

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute binary target: 1 if next day's Close > today's Close, else 0."""
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1]  # Drop last row (no next-day price)
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED MOVING AVERAGE FEATURES (~15 features)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_enhanced_mas(df: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced moving average features: slopes, crossovers, ribbon, convergence.

    All computed from daily OHLCV. No extra API calls needed.
    """
    close = df["Close"]

    # MA slopes (rate of change — trend acceleration)
    if "trend_sma_fast" in df.columns:
        df["ma_slope_sma50"] = df["trend_sma_fast"].pct_change(periods=5)
    if "trend_sma_slow" in df.columns:
        df["ma_slope_sma200"] = df["trend_sma_slow"].pct_change(periods=10)

    # Build custom EMAs and SMAs if not present
    sma20 = close.rolling(20).mean()
    sma50 = df.get("trend_sma_fast", close.rolling(50).mean())
    sma200 = df.get("trend_sma_slow", close.rolling(200).mean())
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["ma_slope_sma20"] = sma20.pct_change(periods=5)
    df["ma_cross_20_50"] = (sma20 > sma50).astype(int)
    df["ma_cross_50_200"] = (sma50 > sma200).astype(int)
    df["ma_cross_ema12_26"] = (ema12 > ema26).astype(int)
    df["ma_ribbon_width"] = (sma20 - sma200) / close.replace(0, np.nan)

    # Hull Moving Average (HMA) approximation
    wma_half = close.rolling(window=10).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
    )
    wma_full = close.rolling(window=20).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
    )
    hma_raw = 2 * wma_half - wma_full
    df["hma"] = hma_raw.rolling(window=int(np.sqrt(20))).mean()
    df["hma_distance"] = (close - df["hma"]) / close.replace(0, np.nan)

    # KAMA slope
    if "momentum_kama" in df.columns:
        df["kama_slope"] = df["momentum_kama"].diff(periods=5) / close

    # MA alignment score: count of MAs in bullish order (shorter > longer)
    ma_list = [close.rolling(w).mean() for w in [5, 10, 20, 50, 200]]
    alignment = pd.Series(0, index=df.index, dtype=float)
    for i in range(len(ma_list) - 1):
        alignment += (ma_list[i] > ma_list[i + 1]).astype(float)
    df["ma_alignment_score"] = alignment / (len(ma_list) - 1)

    # % of MAs price is above
    df["price_vs_all_ma"] = sum((close > ma).astype(float) for ma in ma_list) / len(ma_list)

    # MA convergence (std of MAs normalized by price)
    ma_df = pd.concat([ma / close for ma in ma_list], axis=1)
    df["ma_convergence"] = ma_df.std(axis=1)

    # Price vs envelope (SMA20 ± 2σ)
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["price_vs_ma_envelope_upper"] = (close - upper) / close.replace(0, np.nan)
    df["price_vs_ma_envelope_lower"] = (close - lower) / close.replace(0, np.nan)

    return df

# ═══════════════════════════════════════════════════════════════════════════════
# ICT (INNER CIRCLE TRADER) SMART MONEY CONCEPTS (~25 features)
# All derived from daily OHLCV patterns. No extra API calls.
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ict_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ICT/Smart Money Concept features: FVG, Order Blocks, Liquidity, Market Structure.

    Based on Michael Huddleston's ICT methodology adapted for daily OHLCV.
    """
    if len(df) < 10:
        return df

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    open_ = df["Open"].values
    volume = df["Volume"].values

    # ATR (14) for normalization
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(14).mean().values

    n = len(df)
    fvg_bullish = np.zeros(n, dtype=int)
    fvg_bearish = np.zeros(n, dtype=int)
    fvg_bullish_size = np.zeros(n)
    fvg_bearish_size = np.zeros(n)

    # ── Fair Value Gaps (3-candle pattern) ─────────────────────────────────
    for i in range(2, n):
        # Bullish FVG: candle[i-2].high < candle[i].low (gap between)
        if high[i-2] < low[i]:
            fvg_bullish[i] = 1
            fvg_bullish_size[i] = (low[i] - high[i-2]) / atr[i] if atr[i] > 0 else 0
        # Bearish FVG: candle[i-2].low > candle[i].high
        if low[i-2] > high[i]:
            fvg_bearish[i] = 1
            fvg_bearish_size[i] = (low[i-2] - high[i]) / atr[i] if atr[i] > 0 else 0

    df["fvg_bullish"] = fvg_bullish
    df["fvg_bearish"] = fvg_bearish
    df["fvg_bullish_size"] = fvg_bullish_size
    df["fvg_bearish_size"] = fvg_bearish_size

    # ── Order Blocks (OB) — last opposite candle before >1.5 ATR move ──────
    ob_bullish_levels = np.full(n, np.nan)  # OB high level
    ob_bearish_levels = np.full(n, np.nan)  # OB low level

    for i in range(3, n):
        move = abs(close[i] - close[i-1])
        if move > 1.5 * atr[i] and atr[i] > 0:
            if close[i] > close[i-1]:  # Bullish move → look for prior bear candle
                for j in range(i-1, max(0, i-5), -1):
                    if close[j] < open_[j]:
                        ob_bullish_levels[i] = high[j]
                        break
            else:  # Bearish move → look for prior bull candle
                for j in range(i-1, max(0, i-5), -1):
                    if close[j] > open_[j]:
                        ob_bearish_levels[i] = low[j]
                        break

    df["ob_bullish_level"] = ob_bullish_levels
    df["ob_bearish_level"] = ob_bearish_levels

    # Count OBs in rolling window
    bull_ob_series = pd.Series((~np.isnan(ob_bullish_levels)).astype(int), index=df.index)
    bear_ob_series = pd.Series((~np.isnan(ob_bearish_levels)).astype(int), index=df.index)
    df["ob_bullish_count_20d"] = bull_ob_series.rolling(20).sum()
    df["ob_bearish_count_20d"] = bear_ob_series.rolling(20).sum()

    # Distance from current price to nearest OB
    df["ob_nearest_bullish_distance"] = (close - df["ob_bullish_level"].ffill()) / close
    df["ob_nearest_bearish_distance"] = (close - df["ob_bearish_level"].ffill()) / close

    # OB mitigated (breached) recently
    df["ob_mitigated_bullish"] = ((close > df["ob_bullish_level"].ffill()) &
                                   (df["ob_bullish_level"].notna())).astype(int)
    df["ob_mitigated_bearish"] = ((close < df["ob_bearish_level"].ffill()) &
                                   (df["ob_bearish_level"].notna())).astype(int)

    # ── Liquidity Sweeps ──────────────────────────────────────────────────
    high_5d = pd.Series(high).rolling(5).max().values
    low_5d = pd.Series(low).rolling(5).min().values

    df["liq_sweep_high"] = ((high > high_5d) & (close < open_)).astype(int)
    df["liq_sweep_low"] = ((low < low_5d) & (close > open_)).astype(int)

    # Equal highs/lows (within 0.1%)
    for i in range(2, n):
        for j in range(max(0, i-5), i):
            if j > 0 and high[j] > 0 and abs(high[i] / high[j] - 1) < 0.001:
                fvg_bullish[i] = max(fvg_bullish[i], 1)  # reuse var as flag
            if j > 0 and low[j] > 0 and abs(low[i] / low[j] - 1) < 0.001:
                fvg_bearish[i] = max(fvg_bearish[i], 1)

    df["liq_equal_highs"] = fvg_bullish
    df["liq_equal_lows"] = fvg_bearish

    # ── Market Structure — Swing Points & Break of Structure ───────────────
    swing_high = np.zeros(n, dtype=int)
    swing_low = np.zeros(n, dtype=int)
    lookback = 5

    for i in range(lookback, n - lookback):
        if all(high[i] >= high[i-j] for j in range(1, lookback+1)) and \
           all(high[i] >= high[i+j] for j in range(1, lookback+1)):
            swing_high[i] = 1
        if all(low[i] <= low[i-j] for j in range(1, lookback+1)) and \
           all(low[i] <= low[i+j] for j in range(1, lookback+1)):
            swing_low[i] = 1

    df["ms_swing_high"] = swing_high
    df["ms_swing_low"] = swing_low

    # Break of Structure (BOS) & Market Structure Shift (MSS)
    bos_bullish = np.zeros(n, dtype=int)
    bos_bearish = np.zeros(n, dtype=int)

    last_swing_high = np.nan
    last_swing_low = np.nan
    last_swing_high_idx = -1
    last_swing_low_idx = -1

    for i in range(n):
        if swing_high[i]:
            last_swing_high = high[i]
            last_swing_high_idx = i
        if swing_low[i]:
            last_swing_low = low[i]
            last_swing_low_idx = i

        if not np.isnan(last_swing_high) and close[i] > last_swing_high and i > last_swing_high_idx:
            bos_bullish[i] = 1
        if not np.isnan(last_swing_low) and close[i] < last_swing_low and i > last_swing_low_idx:
            bos_bearish[i] = 1

    df["ms_bos_bullish"] = bos_bullish
    df["ms_bos_bearish"] = bos_bearish

    # Market Structure Shift (MSS) — BOS after a trend
    df["ms_mss_bullish"] = ((bos_bullish == 1) &
                             (pd.Series(close).rolling(10).mean().diff().shift(1) < 0)).astype(int)
    df["ms_mss_bearish"] = ((bos_bearish == 1) &
                             (pd.Series(close).rolling(10).mean().diff().shift(1) > 0)).astype(int)

    # ── Premium / Discount Zones ───────────────────────────────────────────
    high_20d = pd.Series(high).rolling(20).max()
    low_20d = pd.Series(low).rolling(20).min()
    range_20d = high_20d - low_20d
    price_position = (pd.Series(close) - low_20d) / range_20d.replace(0, np.nan)

    df["pd_premium_zone"] = (price_position > 0.70).astype(int)
    df["pd_discount_zone"] = (price_position < 0.30).astype(int)
    df["pd_equilibrium_distance"] = abs(price_position - 0.50)

    # OTE zone (61.8%–79% retracement)
    df["pd_ote_zone"] = ((price_position >= 0.618) & (price_position <= 0.79)).astype(int)

    return df

# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME PROFILE FEATURES (~20 features)
# Uses 5-minute intraday data (free from Yahoo Finance) with local caching
# ═══════════════════════════════════════════════════════════════════════════════

def compute_volume_profile_features(ticker: str) -> dict:
    """Compute volume profile features from 5-minute intraday data.

    Downloads 5m data with caching. Falls back gracefully if download fails.

    Returns:
        dict of volume profile features for the most recent trading day.
    """
    try:
        from ingest import download_intraday_data
    except ImportError:
        return {}

    try:
        df = download_intraday_data(
            ticker, interval="5m", period="5d",
            cache_dir=INTRADAY_CACHE_DIR
        )
    except Exception:
        logger.debug("Volume profile download failed for %s", ticker)
        return {}

    if df.empty or len(df) < 10:
        return {}

    # Work with the most recent full trading day
    df = df.sort_index()
    latest_date = df.index[-1].date()
    day_data = df[df.index.date == latest_date]

    if len(day_data) < 10:
        return {}

    close = day_data["Close"].values
    high = day_data["High"].values
    low = day_data["Low"].values
    vol = day_data["Volume"].values
    typical_price = (high + low + close) / 3
    daily_close = close[-1]
    daily_vwap = np.average(typical_price, weights=vol) if vol.sum() > 0 else daily_close

    features = {}

    # VWAP deviation
    features["vp_vwap_deviation"] = (daily_close - daily_vwap) / daily_vwap if daily_vwap > 0 else 0.0

    # Point of Control (POC) — price bucket with most volume
    n_buckets = 50
    price_range = (low.min(), high.max())
    if price_range[1] > price_range[0]:
        buckets = np.linspace(price_range[0], price_range[1], n_buckets)
        vol_profile = np.zeros(n_buckets - 1)
        for i in range(len(vol_profile)):
            mask = (typical_price >= buckets[i]) & (typical_price < buckets[i+1])
            vol_profile[i] = vol[mask].sum()
        if vol_profile.sum() > 0:
            poc_idx = np.argmax(vol_profile)
            poc = (buckets[poc_idx] + buckets[poc_idx+1]) / 2
            features["vp_poc"] = poc
            features["vp_poc_distance"] = (daily_close - poc) / daily_close if daily_close > 0 else 0.0

            # Value Area (70%)
            cumulative = np.cumsum(np.sort(vol_profile)[::-1])
            total_vol = cumulative[-1]
            if total_vol > 0:
                threshold_70 = 0.70 * total_vol
                sorted_indices = np.argsort(vol_profile)[::-1]
                va_indices = sorted_indices[cumulative / total_vol <= 0.70]
                if len(va_indices) > 0:
                    vah = max(buckets[i+1] for i in va_indices if i < len(buckets)-1)
                    val = min(buckets[i] for i in va_indices if i < len(buckets)-1)
                    features["vp_vah"] = vah
                    features["vp_val"] = val
                    features["vp_value_area_width"] = (vah - val) / daily_vwap if daily_vwap > 0 else 0.0
                    features["vp_close_in_value_area"] = int(val <= daily_close <= vah)
                else:
                    for k in ["vp_vah", "vp_val", "vp_value_area_width", "vp_close_in_value_area"]:
                        features[k] = np.nan

            # Volume skew (above vs below VWAP)
            vol_above = vol[typical_price > daily_vwap].sum()
            vol_below = vol[typical_price < daily_vwap].sum()
            features["vp_volume_skew"] = vol_above / vol_below if vol_below > 0 else 1.0
        else:
            for k in ["vp_poc", "vp_poc_distance", "vp_vah", "vp_val",
                       "vp_value_area_width", "vp_close_in_value_area", "vp_volume_skew"]:
                features[k] = np.nan
    else:
        for k in ["vp_poc", "vp_poc_distance", "vp_vah", "vp_val",
                   "vp_value_area_width", "vp_close_in_value_area", "vp_volume_skew",
                   "vp_vwap_deviation"]:
            features[k] = np.nan

    # Volume at close level
    close_range = 0.005 * daily_close  # ±0.5%
    vol_near_close = vol[(typical_price >= daily_close - close_range) &
                          (typical_price <= daily_close + close_range)].sum()
    features["vp_volume_at_close"] = vol_near_close / vol.sum() if vol.sum() > 0 else 0.0

    # Profile shape
    features["vp_profile_shape"] = (features.get("vp_poc", daily_vwap) - daily_vwap) / daily_vwap \
        if daily_vwap > 0 else 0.0

    # Opening/Closing drives
    if len(day_data) >= 6:
        opening_vol = day_data.iloc[:6]["Volume"].sum()
        closing_vol = day_data.iloc[-6:]["Volume"].sum()
        total_vol = vol.sum()
        features["vp_open_drive"] = opening_vol / total_vol if total_vol > 0 else 0.0
        features["vp_close_drive"] = closing_vol / total_vol if total_vol > 0 else 0.0

    # Intraday volatility and trend
    returns_5m = pd.Series(close).pct_change().dropna()
    features["vp_intraday_volatility"] = returns_5m.std() if len(returns_5m) > 0 else 0.0
    features["vp_intraday_range"] = (high.max() - low.min()) / daily_close if daily_close > 0 else 0.0

    if len(close) > 5:
        x = np.arange(len(close))
        slope = np.polyfit(x, close, 1)[0]
        features["vp_intraday_trend"] = slope / daily_close if daily_close > 0 else 0.0
    else:
        features["vp_intraday_trend"] = 0.0

    # Daily volume context
    if len(df) > 20:
        daily_vol = df["Volume"].resample("D").sum()
        avg_vol_20d = daily_vol.rolling(20).mean().iloc[-1]
        today_vol = daily_vol.iloc[-1] if len(daily_vol) > 0 else vol.sum()
        features["vp_relative_volume"] = today_vol / avg_vol_20d if avg_vol_20d > 0 else 1.0
        features["vp_volume_trend_5d"] = daily_vol.pct_change(periods=5).iloc[-1] \
            if len(daily_vol) >= 6 else 0.0

        # High volume nodes
        vol_threshold = 2 * daily_vol.mean()
        features["vp_high_volume_nodes"] = int((daily_vol > vol_threshold).sum())
    else:
        features["vp_relative_volume"] = 1.0
        features["vp_volume_trend_5d"] = 0.0
        features["vp_high_volume_nodes"] = 0

    # VWAP std deviation
    vwap_series = pd.Series(typical_price * vol).expanding().sum() / vol.cumsum()
    features["vp_vwap_deviation_std"] = (daily_close - daily_vwap) / \
        vwap_series.std() if vwap_series.std() > 0 else 0.0

    # Prefix keys for clarity
    return features

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET CONTEXT & CROSS-STOCK FEATURES (~15 features)
# ═══════════════════════════════════════════════════════════════════════════════

def load_ihsg_data(path: str = "data/raw/ihsg.csv") -> pd.DataFrame:
    """Load IHSG index data."""
    if not os.path.exists(path):
        logger.warning("IHSG data not found at %s. Run ingest.py first.", path)
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, parse_dates=True)

def compute_market_context(df: pd.DataFrame, ihsg: pd.DataFrame,
                           ticker: str) -> pd.DataFrame:
    """Add market context features: IHSG returns, relative strength, beta, breadth.

    Args:
        df: Ticker feature DataFrame (datetime index, must have 'Close')
        ihsg: IHSG index DataFrame (datetime index, must have 'Close')
        ticker: Current ticker symbol

    Returns:
        DataFrame with market context columns appended.
    """
    if ihsg.empty or "Close" not in ihsg.columns:
        for col in ["ihsg_return_1d", "ihsg_return_5d", "relative_strength_1d",
                     "relative_strength_5d", "beta_20d", "correlation_20d"]:
            df[col] = np.nan
        return df

    ticker_close = df["Close"]
    ihsg_close = ihsg["Close"].reindex(df.index, method="ffill")

    df["ihsg_return_1d"] = ihsg_close.pct_change()
    df["ihsg_return_5d"] = ihsg_close.pct_change(periods=5)

    # Relative strength
    ticker_ret = ticker_close.pct_change()
    df["relative_strength_1d"] = ticker_ret - df["ihsg_return_1d"]
    df["relative_strength_5d"] = ticker_close.pct_change(periods=5) - df["ihsg_return_5d"]

    # Rolling beta (20-day)
    common_idx = df.index.intersection(ihsg.index)
    if len(common_idx) >= 21:
        t_ret = ticker_close.loc[common_idx].pct_change().dropna()
        i_ret = ihsg_close.loc[common_idx].pct_change().dropna()
        rolling_cov = t_ret.rolling(20).cov(i_ret)
        rolling_var = i_ret.rolling(20).var()
        beta = (rolling_cov / rolling_var.replace(0, np.nan)).reindex(df.index)
        df["beta_20d"] = beta.ffill().fillna(1.0)
        df["correlation_20d"] = t_ret.rolling(20).corr(i_ret).reindex(df.index).ffill().fillna(0.0)
    else:
        df["beta_20d"] = 1.0
        df["correlation_20d"] = 0.0

    # USD/IDR stress flag
    if "usdidr_rate" in df.columns and "usdidr_return" in df.columns:
        usdidr_sma20 = df["usdidr_rate"].rolling(20).mean()
        usdidr_std20 = df["usdidr_rate"].rolling(20).std()
        df["usdidr_stress"] = ((df["usdidr_rate"] > usdidr_sma20 + usdidr_std20) &
                               (df["usdidr_return"] > 0)).astype(int)
    else:
        df["usdidr_stress"] = 0

    return df

def compute_cross_stock_features(df: pd.DataFrame, ticker: str,
                                 all_ticker_data: dict,
                                 tickers: list) -> pd.DataFrame:
    """Add cross-stock features to an already-engineered feature DataFrame.

    Modifies df in-place by adding: market breadth, sector return, sector rank.

    Args:
        df: Already-engineered feature DataFrame for the ticker (must have 'Close')
        ticker: Current ticker symbol
        all_ticker_data: Dict of {ticker: DataFrame} with raw OHLCV for all tickers
        tickers: Full ticker list

    Returns:
        Same DataFrame with cross-stock columns appended.
    """
    if df is None or len(df) < 5:
        return df

    # Market breadth: % of tickers that closed up each day
    closes = {}
    for t in tickers:
        if t in all_ticker_data and "Close" in all_ticker_data[t].columns:
            closes[t] = all_ticker_data[t]["Close"]

    if len(closes) > 1:
        common_idx = df.index
        breadth = pd.Series(0.0, index=common_idx)
        for idx in common_idx:
            ups = 0
            for t in closes:
                if idx not in closes[t].index:
                    continue
                series_slice = closes[t].loc[:idx]
                if len(series_slice) >= 2 and series_slice.iloc[-1] > series_slice.iloc[-2]:
                    ups += 1
            total = len(closes)
            breadth.loc[idx] = ups / total if total > 0 else 0.5
        df["market_breadth"] = breadth
        adv = pd.Series(0.0, index=common_idx)
        dec = pd.Series(0.0, index=common_idx)
        for idx in common_idx:
            ups = 0
            downs = 0
            for t in closes:
                if idx not in closes[t].index:
                    continue
                series_slice = closes[t].loc[:idx]
                if len(series_slice) >= 2:
                    if series_slice.iloc[-1] > series_slice.iloc[-2]:
                        ups += 1
                    elif series_slice.iloc[-1] < series_slice.iloc[-2]:
                        downs += 1
            adv.loc[idx] = ups
            dec.loc[idx] = downs
        df["market_advance_decline"] = (adv - dec) / len(closes) if len(closes) > 0 else 0

    # Sector features
    sector = None
    for sec_name, sec_tickers in SECTORS.items():
        if ticker in sec_tickers:
            sector = sec_name
            break

    if sector:
        sec_tickers = [t for t in SECTORS[sector] if t in all_ticker_data]
        if len(sec_tickers) > 1:
            common_idx = df.index
            sector_return = pd.Series(0.0, index=common_idx)
            for idx in common_idx:
                rets = []
                for st in sec_tickers:
                    if st in all_ticker_data and "Close" in all_ticker_data[st].columns:
                        st_close = all_ticker_data[st]["Close"]
                        if idx in st_close.index:
                            loc_idx = st_close.index.get_loc(idx)
                            if loc_idx > 0:
                                rets.append(st_close.iloc[loc_idx] / st_close.iloc[loc_idx-1] - 1)
                sector_return.loc[idx] = np.mean(rets) if rets else 0.0
            df["sector_return_1d"] = sector_return

            # Ticker vs sector relative strength
            ticker_ret = df["Close"].pct_change()
            df["sector_relative_strength"] = ticker_ret - sector_return

            # Sector rank (5-day return percentile within sector)
            five_day_rets = {}
            for st in sec_tickers:
                if st in all_ticker_data:
                    st_close = all_ticker_data[st]["Close"]
                    ret_5d = st_close.pct_change(periods=5).reindex(common_idx)
                    five_day_rets[st] = ret_5d
            if five_day_rets:
                ret_df = pd.DataFrame(five_day_rets, index=common_idx)
                ranks = ret_df.rank(axis=1, pct=True)
                if ticker in ranks.columns:
                    df["sector_rank_5d"] = ranks[ticker]

            # Sector volume ratio
            if "Volume" in df.columns:
                sec_volumes = []
                for st in sec_tickers:
                    if st in all_ticker_data and "Volume" in all_ticker_data[st].columns:
                        sec_volumes.append(all_ticker_data[st]["Volume"].reindex(common_idx))
                if sec_volumes:
                    avg_sec_vol = pd.concat(sec_volumes, axis=1).mean(axis=1)
                    df["sector_volume_ratio"] = df["Volume"] / avg_sec_vol.replace(0, np.nan)

    return df

# ═══════════════════════════════════════════════════════════════════════════════
# IDX FUNDAMENTALS (supplementary Indonesian-specific data)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_idx_fundamentals(ticker: str) -> dict:
    """Fetch supplementary IDX fundamentals from community libraries.

    Uses noczero/idx-fundamental-analysis if available.
    Falls back to NaN on failure or if not installed.
    """
    try:
        # Attempt to use idx-fundamental-analysis if installed
        import importlib
        if importlib.util.find_spec("idx_fundamental_analysis"):
            # This library may have different API; attempt gracefully
            logger.debug("idx-fundamental-analysis found but API may vary")
    except Exception:
        pass

    # For now, return NaN placeholders. The library can be wired in when installed.
    fields = ["idn_der", "idn_roe", "idn_roa", "idn_current_ratio",
              "idn_gpm", "idn_npm", "idn_eps_growth", "idn_bvps",
              "idn_per", "idn_pbv"]
    return {f: np.nan for f in fields}

# ═══════════════════════════════════════════════════════════════════════════════
# CORE: ENGINEER FEATURES FOR A SINGLE TICKER (refactored to use shared funcs)
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features_for_ticker(df, ticker, fundamentals, usdidr, fred_macro,
                                 bi_rate, ihsg=None, all_ticker_data=None,
                                 mode="full") -> pd.DataFrame:
    """Engineer features for a single ticker.

    Args:
        df: Raw multi-ticker OHLCV DataFrame (yfinance MultiIndex format)
        ticker: Ticker symbol
        fundamentals: Dict from fetch_fundamentals()
        usdidr: USD/IDR DataFrame
        fred_macro: FRED macro DataFrame
        bi_rate: BI rate Series
        ihsg: IHSG index DataFrame (optional, for market context)
        all_ticker_data: Dict of {ticker: DataFrame} for cross-stock features (optional)
        mode: 'full', 'incremental', or 'ci' (CI mode skips intraday)

    Returns:
        Feature-rich DataFrame or None if insufficient data.
    """
    ticker_df = df[ticker].copy()
    ticker_df = dropna(ticker_df)

    if len(ticker_df) < 50:
        return None

    # ── Shared: TA indicators ────────────────────────────────────────────
    if FEATURE_FLAGS.get("ta_indicators", True):
        ticker_df = compute_ta_features(ticker_df)

    # ── Shared: Custom features ──────────────────────────────────────────
    ticker_df = compute_custom_features(ticker_df)

    # ── Enhanced MAs ────────────────────────────────────────────────────
    if FEATURE_FLAGS.get("enhanced_mas", True):
        ticker_df = compute_enhanced_mas(ticker_df)

    # ── ICT Features ─────────────────────────────────────────────────────
    if FEATURE_FLAGS.get("ict_suite", True):
        ticker_df = compute_ict_features(ticker_df)

    # ── Shared: Macro injection ──────────────────────────────────────────
    ticker_df = inject_macro_features(ticker_df, fundamentals, usdidr, fred_macro, bi_rate)

    # ── Volume Profile (skip in CI mode) ─────────────────────────────────
    if FEATURE_FLAGS.get("volume_profile", True) and mode != "ci":
        try:
            vp_features = compute_volume_profile_features(ticker)
            for col, val in vp_features.items():
                ticker_df[col] = val
        except Exception:
            pass  # Volume profile is optional

    # ── Market Context ──────────────────────────────────────────────────
    if FEATURE_FLAGS.get("market_context", True) and ihsg is not None:
        ticker_df = compute_market_context(ticker_df, ihsg, ticker)

    # ── IDX Fundamentals ────────────────────────────────────────────────
    if FEATURE_FLAGS.get("idx_fundamentals", True):
        idx_fund = fetch_idx_fundamentals(ticker)
        for col, val in idx_fund.items():
            ticker_df[col] = val

    # ── Shared: Target ─────────────────────────────────────────────────
    ticker_df = compute_target(ticker_df)
    ticker_df["ticker"] = ticker

    return ticker_df

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FULL FEATURE SET (batch pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_set(raw_path: str, output_path: str, tickers: list,
                      mode: str = "full"):
    """Build the complete feature set for all tickers.

    Args:
        raw_path: Path to raw OHLCV CSV
        output_path: Path for output Parquet/CSV
        tickers: List of ticker symbols
        mode: 'full', 'incremental', or 'ci'
    """
    df = load_data(raw_path)
    start = str(df.index.min().date())
    end = str(df.index.max().date())
    logger.info("Date range: %s → %s", start, end)

    # ── Fetch external data once (shared across all tickers) ─────────────────
    logger.info("Fetching USD/IDR rate...")
    usdidr = fetch_usdidr(start, end)

    logger.info("Fetching FRED macro + commodity data...")
    fred_macro = fetch_fred_macro(start, end)

    logger.info("Fetching BI Rate...")
    bi_rate = fetch_bi_rate(start, end)

    # ── IHSG data for market context ─────────────────────────────────────────
    ihsg = None
    if FEATURE_FLAGS.get("market_context", True):
        logger.info("Loading IHSG data...")
        ihsg = load_ihsg_data()

    # ── Pre-extract per-ticker Close data for cross-stock features ───────────
    all_ticker_data = {}
    if FEATURE_FLAGS.get("market_context", True):
        for t in tickers:
            try:
                t_df = df[t].copy()
                all_ticker_data[t] = dropna(t_df)
            except KeyError:
                pass

    # ── Process each ticker ──────────────────────────────────────────────────
    all_features = []
    for i, ticker in enumerate(tickers):
        logger.info("[%d/%d] Processing %s...", i+1, len(tickers), ticker)
        try:
            fundamentals = fetch_fundamentals(ticker)
            featured = engineer_features_for_ticker(
                df, ticker, fundamentals, usdidr, fred_macro, bi_rate,
                ihsg=ihsg, all_ticker_data=all_ticker_data, mode=mode,
            )

            if featured is None or len(featured) < 50:
                logger.warning("Skipping %s — insufficient data", ticker)
                continue

            # Google Trends (per ticker, rate-limited)
            if FEATURE_FLAGS.get("google_trends", True) and mode != "ci":
                trend = fetch_google_trends(ticker, start, end)
                if not trend.empty:
                    trend.index = pd.to_datetime(trend.index)
                    featured = featured.join(trend, how="left")
                featured["google_trend"] = featured["google_trend"].ffill().bfill().fillna(0) \
                    if "google_trend" in featured.columns else 0
            else:
                featured["google_trend"] = 0

            # Cross-stock features (computed after per-ticker processing)
            if FEATURE_FLAGS.get("market_context", True) and all_ticker_data:
                featured = compute_cross_stock_features(
                    featured, ticker, all_ticker_data, tickers
                )

            all_features.append(featured)
        except Exception as e:
            logger.error("Skipping %s: %s", ticker, e)

    if not all_features:
        logger.error("No features generated for any ticker!")
        return

    combined = pd.concat(all_features)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.endswith(".parquet"):
        combined.to_parquet(output_path)
    else:
        combined.to_csv(output_path)

    logger.info("Saved → %s | Shape: %s | Columns: %d",
                output_path, combined.shape, len(combined.columns))

    # Also save CSV for backward compatibility
    if output_path.endswith(".parquet"):
        csv_path = DATA_PROCESSED_CSV_PATH
        combined.to_csv(csv_path)
        logger.info("CSV fallback → %s", csv_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build feature set for stock prediction")
    parser.add_argument("--mode", choices=["full", "incremental", "ci"], default="full",
                        help="Pipeline mode: full (all features), ci (skip slow features)")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                        help="Output format")
    args = parser.parse_args()

    output_path = DATA_PROCESSED_PATH if args.format == "parquet" else DATA_PROCESSED_CSV_PATH

    logger.info("Building features for %d tickers (mode=%s, format=%s)...",
                len(TICKERS), args.mode, args.format)
    build_feature_set(
        raw_path=DATA_RAW_PATH,
        output_path=output_path,
        tickers=TICKERS,
        mode=args.mode,
    )
