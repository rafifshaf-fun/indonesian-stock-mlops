"""
ICT (Inner Circle Trader) Smart Money Concepts features (~25 features).
FVG, Order Blocks, Liquidity, Market Structure. All from daily OHLCV.
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


def compute_ict_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ICT/Smart Money Concept features: FVG, Order Blocks, Liquidity, Market Structure.

    Collects new columns into a dict and concatenates once to avoid
    DataFrame fragmentation warnings.
    """
    if len(df) < 10:
        return df

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    open_ = df["Open"].values
    n = len(df)
    new = {}  # batch all new columns, then pd.concat once

    # ATR (14) for normalization
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(14).mean().values

    # ── Fair Value Gaps ───────────────────────────────────────────────────
    fvg_bullish = np.zeros(n, dtype=int)
    fvg_bearish = np.zeros(n, dtype=int)
    fvg_bullish_size = np.zeros(n)
    fvg_bearish_size = np.zeros(n)
    for i in range(2, n):
        if high[i-2] < low[i]:
            fvg_bullish[i] = 1
            fvg_bullish_size[i] = (low[i] - high[i-2]) / atr[i] if atr[i] > 0 else 0
        if low[i-2] > high[i]:
            fvg_bearish[i] = 1
            fvg_bearish_size[i] = (low[i-2] - high[i]) / atr[i] if atr[i] > 0 else 0
    new["fvg_bullish"] = fvg_bullish
    new["fvg_bearish"] = fvg_bearish
    new["fvg_bullish_size"] = fvg_bullish_size
    new["fvg_bearish_size"] = fvg_bearish_size

    # ── Order Blocks ─────────────────────────────────────────────────────
    ob_bull = np.full(n, np.nan)
    ob_bear = np.full(n, np.nan)
    for i in range(3, n):
        move = abs(close[i] - close[i-1])
        if move > 1.5 * atr[i] and atr[i] > 0:
            if close[i] > close[i-1]:
                for j in range(i-1, max(0, i-5), -1):
                    if close[j] < open_[j]:
                        ob_bull[i] = high[j]; break
            else:
                for j in range(i-1, max(0, i-5), -1):
                    if close[j] > open_[j]:
                        ob_bear[i] = low[j]; break
    new["ob_bullish_level"] = ob_bull
    new["ob_bearish_level"] = ob_bear
    new["ob_bullish_count_20d"] = pd.Series((~np.isnan(ob_bull)).astype(int), index=df.index).rolling(20).sum()
    new["ob_bearish_count_20d"] = pd.Series((~np.isnan(ob_bear)).astype(int), index=df.index).rolling(20).sum()

    # ── Liquidity Sweeps ──────────────────────────────────────────────────
    h5 = pd.Series(high).rolling(5).max().values
    l5 = pd.Series(low).rolling(5).min().values
    new["liq_sweep_high"] = ((high > h5) & (close < open_)).astype(int)
    new["liq_sweep_low"] = ((low < l5) & (close > open_)).astype(int)

    eq_highs = np.zeros(n, dtype=int)
    eq_lows = np.zeros(n, dtype=int)
    for i in range(2, n):
        for j in range(max(0, i-5), i):
            if j > 0 and high[j] > 0 and abs(high[i] / high[j] - 1) < 0.001:
                eq_highs[i] = 1
            if j > 0 and low[j] > 0 and abs(low[i] / low[j] - 1) < 0.001:
                eq_lows[i] = 1
    new["liq_equal_highs"] = eq_highs
    new["liq_equal_lows"] = eq_lows

    # ── Market Structure ──────────────────────────────────────────────────
    swing_high = np.zeros(n, dtype=int)
    swing_low = np.zeros(n, dtype=int)
    lb = 5
    for i in range(lb, n - lb):
        if all(high[i] >= high[i-j] for j in range(1, lb+1)) and \
           all(high[i] >= high[i+j] for j in range(1, lb+1)):
            swing_high[i] = 1
        if all(low[i] <= low[i-j] for j in range(1, lb+1)) and \
           all(low[i] <= low[i+j] for j in range(1, lb+1)):
            swing_low[i] = 1
    new["ms_swing_high"] = swing_high
    new["ms_swing_low"] = swing_low

    bos_bullish = np.zeros(n, dtype=int)
    bos_bearish = np.zeros(n, dtype=int)
    lsh, lsl, lshi, lsli = np.nan, np.nan, -1, -1
    for i in range(n):
        if swing_high[i]: lsh = high[i]; lshi = i
        if swing_low[i]: lsl = low[i]; lsli = i
        if not np.isnan(lsh) and close[i] > lsh and i > lshi: bos_bullish[i] = 1
        if not np.isnan(lsl) and close[i] < lsl and i > lsli: bos_bearish[i] = 1
    new["ms_bos_bullish"] = bos_bullish
    new["ms_bos_bearish"] = bos_bearish
    new["ms_mss_bullish"] = ((bos_bullish == 1) &
                              (pd.Series(close).rolling(10).mean().diff().shift(1) < 0)).astype(int)
    new["ms_mss_bearish"] = ((bos_bearish == 1) &
                              (pd.Series(close).rolling(10).mean().diff().shift(1) > 0)).astype(int)

    # ── Premium / Discount Zones ───────────────────────────────────────────
    h20 = pd.Series(high).rolling(20).max()
    l20 = pd.Series(low).rolling(20).min()
    pp = (pd.Series(close) - l20) / (h20 - l20).replace(0, np.nan)
    new["pd_premium_zone"] = (pp > 0.70).astype(int)
    new["pd_discount_zone"] = (pp < 0.30).astype(int)
    new["pd_equilibrium_distance"] = abs(pp - 0.50)
    new["pd_ote_zone"] = ((pp >= 0.618) & (pp <= 0.79)).astype(int)

    # ── Batch concat all at once ───────────────────────────────────────────
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # ── Dependent columns (refer to columns just added above) ──────────────
    df["ob_nearest_bullish_distance"] = (close - df["ob_bullish_level"].ffill()) / close
    df["ob_nearest_bearish_distance"] = (close - df["ob_bearish_level"].ffill()) / close
    df["ob_mitigated_bullish"] = ((close > df["ob_bullish_level"].ffill()) &
                                   (df["ob_bullish_level"].notna())).astype(int)
    df["ob_mitigated_bearish"] = ((close < df["ob_bearish_level"].ffill()) &
                                   (df["ob_bearish_level"].notna())).astype(int)
    return df
