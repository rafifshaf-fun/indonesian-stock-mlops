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

    Based on Michael Huddleston's ICT methodology adapted for daily OHLCV.
    """
    if len(df) < 10:
        return df

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    open_ = df["Open"].values

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
        if high[i-2] < low[i]:
            fvg_bullish[i] = 1
            fvg_bullish_size[i] = (low[i] - high[i-2]) / atr[i] if atr[i] > 0 else 0
        if low[i-2] > high[i]:
            fvg_bearish[i] = 1
            fvg_bearish_size[i] = (low[i-2] - high[i]) / atr[i] if atr[i] > 0 else 0

    df["fvg_bullish"] = fvg_bullish
    df["fvg_bearish"] = fvg_bearish
    df["fvg_bullish_size"] = fvg_bullish_size
    df["fvg_bearish_size"] = fvg_bearish_size

    # ── Order Blocks (OB) — last opposite candle before >1.5 ATR move ──────
    ob_bullish_levels = np.full(n, np.nan)
    ob_bearish_levels = np.full(n, np.nan)

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

    bull_ob_series = pd.Series((~np.isnan(ob_bullish_levels)).astype(int), index=df.index)
    bear_ob_series = pd.Series((~np.isnan(ob_bearish_levels)).astype(int), index=df.index)
    df["ob_bullish_count_20d"] = bull_ob_series.rolling(20).sum()
    df["ob_bearish_count_20d"] = bear_ob_series.rolling(20).sum()

    df["ob_nearest_bullish_distance"] = (close - df["ob_bullish_level"].ffill()) / close
    df["ob_nearest_bearish_distance"] = (close - df["ob_bearish_level"].ffill()) / close

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
    eq_highs = np.zeros(n, dtype=int)
    eq_lows = np.zeros(n, dtype=int)
    for i in range(2, n):
        for j in range(max(0, i-5), i):
            if j > 0 and high[j] > 0 and abs(high[i] / high[j] - 1) < 0.001:
                eq_highs[i] = 1
            if j > 0 and low[j] > 0 and abs(low[i] / low[j] - 1) < 0.001:
                eq_lows[i] = 1

    df["liq_equal_highs"] = eq_highs
    df["liq_equal_lows"] = eq_lows

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
    df["pd_ote_zone"] = ((price_position >= 0.618) & (price_position <= 0.79)).astype(int)

    return df
