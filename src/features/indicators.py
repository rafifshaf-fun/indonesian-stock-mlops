"""
Technical analysis indicators and custom features.
"""
import pandas as pd
import numpy as np
from ta import add_all_ta_features

# sys.path trick so submodules can import from src/
import sys as _sys
import os as _os
_curr = _os.path.dirname(_os.path.abspath(__file__))
_parent = _os.path.dirname(_curr)
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)


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

    # Copy to avoid fragmentation before ta library adds 75+ columns
    df = df.copy()
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low",
        close="Close", volume="Volume", fillna=True
    )
    df = df.ffill().bfill().fillna(0)
    return df


def compute_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom returns-based features derived from daily OHLCV.

    Adds: ret_1d, ret_5d, dist_from_sma50, dist_from_sma200
    """
    new = {
        "ret_1d": df["Close"].pct_change(),
        "ret_5d": df["Close"].pct_change(periods=5),
    }
    if "trend_sma_fast" in df.columns:
        new["dist_from_sma50"] = df["Close"] / df["trend_sma_fast"] - 1
    if "trend_sma_slow" in df.columns:
        new["dist_from_sma200"] = df["Close"] / df["trend_sma_slow"] - 1

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
