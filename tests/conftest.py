"""Shared test fixtures — fully offline and deterministic.

Every fixture here is synthetic: no fixture performs a network call and none
reads ``data/``, so the suite passes on a fresh clone.  Frames are generated
from the fixed :data:`SEED`, so repeated runs produce identical bytes and a
failure can be reproduced exactly.

Fixture contracts (consumed by this task and by later tasks in the rescue):

``raw_ohlcv``
    A 3-ticker, 300-row frame in yfinance ``group_by="ticker"`` layout: a
    2-level column ``MultiIndex`` of (``Ticker``, ``Field``) giving
    ``3 * 5 = 15`` columns, on a business-day ``DatetimeIndex``.
``ihsg_frame``
    A single-level OHLCV frame for the IHSG index, carrying the ``Close``
    column ``compute_market_context`` expects.  It shares ``raw_ohlcv``'s exact
    business-day index so a ticker frame sliced out of ``raw_ohlcv`` can be fed
    straight into ``compute_market_context`` without the ``reindex(method=
    "ffill")`` inside it producing a NaN tail.
``fundamentals_dict``
    A plain mapping of fundamental field names to floats, mirroring the shape
    returned by ``features.fetchers.fetch_fundamentals`` and
    ``features.fetchers.fetch_idx_fundamentals`` — the two dicts the pipeline
    splats into feature columns.
``tmp_model_dir``
    An isolated, empty directory for model artefacts, asserted to live outside
    the repository's ``mlruns/`` tree so no test can write into the tracked
    model store.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

# ── Determinism ───────────────────────────────────────────────────────────────
# Fixed seed and fixed end date: identical fixtures on every run, every machine.
SEED = 20261013
END_DATE = "2024-12-31"
N_ROWS = 300

OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")

# Three LQ45 tickers, hardcoded rather than imported from config so that these
# fixtures stay independent of the application's import chain.
RAW_TICKERS = ("BBCA.JK", "BBRI.JK", "BMRI.JK")

# ``fetch_fundamentals`` returns these eight keys.
FUNDAMENTAL_FIELDS = {
    "pe_ratio": 18.4,
    "pb_ratio": 3.1,
    "dividend_yield": 0.027,
    "market_cap": 1.15e12,
    "debt_to_equity": 0.62,
    "return_on_equity": 0.21,
    "revenue_growth": 0.083,
    "profit_margins": 0.34,
}

# ``fetch_idx_fundamentals`` returns these ten keys.
IDX_FUNDAMENTAL_FIELDS = {
    "idn_der": 0.58,
    "idn_roe": 0.19,
    "idn_roa": 0.031,
    "idn_current_ratio": 1.9,
    "idn_gpm": 0.46,
    "idn_npm": 0.28,
    "idn_eps_growth": 0.11,
    "idn_bvps": 742.0,
    "idn_per": 16.2,
    "idn_pbv": 2.7,
}

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_MLRUNS_DIR = _REPO_ROOT / "mlruns"


def _business_days(n: int = N_ROWS) -> pd.DatetimeIndex:
    """Business-day index ending on ``END_DATE``.

    Weekends are excluded but holidays are not: that keeps the index a pure
    function of ``END_DATE`` and ``n`` with no calendar data to depend on.
    """
    return pd.bdate_range(end=END_DATE, periods=n)


def _ohlcv_frame(start_price: float, rng: np.random.Generator,
                 index: pd.DatetimeIndex) -> pd.DataFrame:
    """A plausible, strictly positive OHLCV frame drawn from a seeded RNG."""
    n = len(index)

    # Geometric random walk for Close, ~1.8% daily vol, mild upward drift.
    returns = rng.normal(loc=0.0005, scale=0.018, size=n)
    close = np.round(start_price * np.exp(np.cumsum(returns)), 2)

    prev_close = np.concatenate([[start_price], close[:-1]])
    open_ = np.round(prev_close * (1.0 + rng.normal(0.0, 0.004, size=n)), 2)

    # High/Low bracket the open-close range, then are re-bounded so rounding to
    # two decimals can never invert the bar.
    oc_min = np.minimum(open_, close)
    oc_max = np.maximum(open_, close)
    high = np.round(oc_max * (1.0 + np.abs(rng.normal(0.0, 0.006, size=n))), 2)
    low = np.round(oc_min * (1.0 - np.abs(rng.normal(0.0, 0.006, size=n))), 2)
    high = np.maximum(high, oc_max)
    low = np.maximum(np.minimum(low, oc_min), 0.01)

    # Median ~2.4e7 shares/day, the order of magnitude for an LQ45 large cap.
    volume = rng.lognormal(mean=17.0, sigma=0.35, size=n).astype("int64")

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": volume},
        index=index,
    )


@pytest.fixture
def raw_ohlcv() -> pd.DataFrame:
    """3 tickers x 300 business days of synthetic OHLCV, yfinance layout."""
    index = _business_days()

    frames = {}
    for offset, ticker in enumerate(RAW_TICKERS):
        rng = np.random.default_rng(SEED + offset)
        frames[ticker] = _ohlcv_frame(
            start_price=2_000.0 + 500.0 * offset, rng=rng, index=index
        )

    df = pd.concat(frames, axis=1)
    df.columns = df.columns.set_names(["Ticker", "Field"])
    return df


@pytest.fixture
def ihsg_frame() -> pd.DataFrame:
    """Single-level IHSG OHLCV frame sharing ``raw_ohlcv``'s index."""
    rng = np.random.default_rng(SEED + 100)
    return _ohlcv_frame(start_price=7_200.0, rng=rng, index=_business_days())


@pytest.fixture
def fundamentals_dict() -> dict[str, float]:
    """Fundamental field name -> float, as produced by the fetchers."""
    return {**FUNDAMENTAL_FIELDS, **IDX_FUNDAMENTAL_FIELDS}


@pytest.fixture
def tmp_model_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """An empty, isolated directory for model artefacts — never ``mlruns/``."""
    model_dir = (tmp_path / "models").resolve()
    model_dir.mkdir()
    assert _MLRUNS_DIR.resolve() not in model_dir.parents, (
        "tmp_model_dir must not be inside the tracked mlruns/ tree"
    )
    return model_dir
