"""
Market context & cross-stock features (~15 features).
IHSG returns, relative strength, beta, breadth, sector features.
"""
import os
import numpy as np
import pandas as pd

# sys.path trick
import sys as _sys
_curr = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_curr)
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)

from config import SECTORS, get_logger

logger = get_logger(__name__)


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
