"""
Enhanced Moving Average features (~15 features).
Slopes, crossovers, ribbon, convergence, alignment.
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


def compute_enhanced_mas(df: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced moving average features: slopes, crossovers, ribbon, convergence.

    Batches new columns into a dict and concatenates once to avoid
    DataFrame fragmentation warnings.
    """
    close = df["Close"]
    new = {}

    # MA slopes
    if "trend_sma_fast" in df.columns:
        new["ma_slope_sma50"] = df["trend_sma_fast"].pct_change(periods=5)
    if "trend_sma_slow" in df.columns:
        new["ma_slope_sma200"] = df["trend_sma_slow"].pct_change(periods=10)

    sma20 = close.rolling(20).mean()
    sma50 = df.get("trend_sma_fast", close.rolling(50).mean())
    sma200 = df.get("trend_sma_slow", close.rolling(200).mean())
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    new["ma_slope_sma20"] = sma20.pct_change(periods=5)
    new["ma_cross_20_50"] = (sma20 > sma50).astype(int)
    new["ma_cross_50_200"] = (sma50 > sma200).astype(int)
    new["ma_cross_ema12_26"] = (ema12 > ema26).astype(int)
    new["ma_ribbon_width"] = (sma20 - sma200) / close.replace(0, np.nan)

    # Hull Moving Average
    _w_sum = lambda w: np.arange(1, w + 1).sum()
    wma_half = close.rolling(10).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / _w_sum(len(x)), raw=True)
    wma_full = close.rolling(20).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / _w_sum(len(x)), raw=True)
    hma_raw = 2 * wma_half - wma_full
    new["hma"] = hma_raw.rolling(window=int(np.sqrt(20))).mean()
    new["hma_distance"] = (close - new["hma"]) / close.replace(0, np.nan)

    if "momentum_kama" in df.columns:
        new["kama_slope"] = df["momentum_kama"].diff(periods=5) / close

    # MA alignment
    ma_list = [close.rolling(w).mean() for w in [5, 10, 20, 50, 200]]
    alignment = pd.Series(0, index=df.index, dtype=float)
    for i in range(len(ma_list) - 1):
        alignment += (ma_list[i] > ma_list[i + 1]).astype(float)
    new["ma_alignment_score"] = alignment / (len(ma_list) - 1)
    new["price_vs_all_ma"] = sum((close > ma).astype(float) for ma in ma_list) / len(ma_list)

    ma_df = pd.concat([ma / close for ma in ma_list], axis=1)
    new["ma_convergence"] = ma_df.std(axis=1)

    # Price vs envelope
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    new["price_vs_ma_envelope_upper"] = (close - upper) / close.replace(0, np.nan)
    new["price_vs_ma_envelope_lower"] = (close - lower) / close.replace(0, np.nan)

    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df
