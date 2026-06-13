"""
Volume Profile features (~20 features).
Uses 5-minute intraday data with local caching.
"""
import numpy as np
import pandas as pd

# sys.path trick
import sys as _sys
import os as _os
_curr = _os.path.dirname(_os.path.abspath(__file__))
_parent = _os.path.dirname(_curr)
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)

from config import INTRADAY_CACHE_DIR, get_logger

logger = get_logger(__name__)


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
                va_indices = sorted_indices[np.where(cumulative / total_vol <= 0.70)]
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

    return features
