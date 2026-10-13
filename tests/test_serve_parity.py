"""Task 1.3 — serving builds its features with the training builder.

The audit's most damaging finding was a serving path that *manufactured* its
market-context, volume-profile and IDX-fundamental columns as hardcoded ``NaN``
placeholders instead of computing them, while training computed them for real.
Six of the faked columns were ones the trained model actually uses, which moved
the predicted probability by ~0.10 against a 0.50 decision boundary.

These tests drive serving's own entry point, ``serve.build_inference_features``,
against the synthetic Task 1.1 fixtures, and they are offline by construction:

* an autouse fixture replaces ``requests.get``/``post`` and
  ``yfinance.download``/``Ticker`` with a function that raises, so passing tests
  prove no request was attempted;
* the same fixture points ``serve.UNIVERSE_RAW_PATH`` and ``serve.IHSG_PATH`` at
  paths that do not exist, so nothing can silently fall back to the real
  ``data/raw/`` tree: the only universe snapshot in play is the one injected
  through ``serve.load_universe_context``;
* the universe snapshot is injected through that module-level name — which
  ``build_inference_features`` looks up in its module globals, so the monkeypatch
  replaces what the function actually calls;
* the three daily macro caches are injected as module attributes, and
  ``features.fetchers.fetch_news_sentiment`` is stubbed (see ``served_frame``).
"""
import os
import time

import numpy as np
import pandas as pd
import pytest

# ``serve`` imports mlflow, which imports mlflow.tracing, which imports
# opentelemetry, whose runtime-context loader still calls the removed
# ``importlib.metadata`` dict interface.  On Python 3.11 that emits
# ``DeprecationWarning: SelectableGroups dict interface is deprecated. Use
# select.`` from opentelemetry/util/_importlib_metadata.py:32 — upstream, not
# this repository.
#
# The second message comes from FastAPI, raised by src/serve.py's own
# ``@app.on_event("startup")`` decorator, which is deprecated in favour of a
# lifespan handler.  Migrating it is a structural change to the app, outside
# this task's `src/serve.py:287-356` range and unverifiable without running the
# server, so it is reported as a finding rather than silently restructured here.
# That message begins with a newline, hence the inline DOTALL flag.
#
# The suite runs with ``filterwarnings = ["error"]`` (Task 1.1's deliberate
# choice), so without these two filters the import aborts every test in this
# file.  Both are scoped to their exact message and to this one module: every
# other warning — including any warning the feature code itself raises — still
# escalates.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:SelectableGroups dict interface is deprecated:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:(?s).*on_event is deprecated.*:DeprecationWarning"
    ),
]

# The tickers the Task 1.1 fixtures carry.  All three sit in
# ``config.SECTORS["banking"]``, so handing all three to the serving path is what
# makes the sector-relative columns computable at all: with one ticker
# ``compute_cross_stock_features`` has no peer to compare against and emits no
# sector column, exactly as it does in training.
FIXTURE_TICKERS = ("BBCA.JK", "BBRI.JK", "BMRI.JK")

# The six features the audit named as fabricated at inference although the
# trained model uses them.
AUDIT_NAMED_MARKET_FEATURES = (
    "correlation_20d",
    "sector_relative_strength",
    "ihsg_return_5d",
    "relative_strength_1d",
    "sector_return_1d",
    "beta_20d",
)

# Columns that are all-NaN in the served row for reasons that predate this task
# and lie outside it, so that any *other* empty column still fails the test:
#
# * the six ICT columns are all-NaN because of an index-alignment bug at
#   src/features/ict.py:117-129 — ``pd.DataFrame(new, index=df.index)``
#   reindexes a RangeIndex-ed ``pd.Series`` onto the DatetimeIndex, producing
#   NaN.  The same six are all-NaN in the committed
#   data/processed/features.parquet, i.e. training's frame carries them too;
# * ``ob_bullish_level`` / ``ob_bearish_level`` are legitimately NaN on this
#   particular row: an order-block level is not known until a block is set, and
#   training has the same per-row NaN.
PRE_EXISTING_NAN_COLUMNS = frozenset({
    "ms_mss_bullish",
    "ms_mss_bearish",
    "pd_premium_zone",
    "pd_discount_zone",
    "pd_equilibrium_distance",
    "pd_ote_zone",
    "ob_bullish_level",
    "ob_bearish_level",
})


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Make any network access or real-``data/`` read a loud failure.

    The network stubs raise *and* record the attempt, and the record is asserted
    empty after the test: the application wraps every fetch in ``try/except``, so
    a raising stub alone could be swallowed and the test would pass anyway.
    """
    import serve
    import requests
    import yfinance as yf

    attempts = []

    def explode(*args, **kwargs):
        attempts.append(f"{args[:1]!r}{kwargs!r}"[:120])
        raise AssertionError("serving parity tests must never use the network")

    monkeypatch.setattr(requests, "get", explode)
    monkeypatch.setattr(requests, "post", explode)
    monkeypatch.setattr(yf, "download", explode)
    monkeypatch.setattr(yf, "Ticker", explode)

    # Deliberately absent files: if a code path read the real universe instead of
    # the injected context its features would come out empty and the tests below
    # would fail, rather than quietly depending on this machine's data/raw/.
    monkeypatch.setattr(serve, "UNIVERSE_RAW_PATH", str(tmp_path / "absent-stocks.csv"))
    monkeypatch.setattr(serve, "IHSG_PATH", str(tmp_path / "absent-ihsg.csv"))

    yield
    assert attempts == [], f"network calls attempted during this test: {attempts}"


@pytest.fixture
def injected(raw_ohlcv, ihsg_frame, monkeypatch):
    """Install the fixture universe and a warm macro cache into ``serve``."""
    import serve
    import features.fetchers as fetchers

    all_ticker_data = {t: raw_ohlcv[t].dropna() for t in FIXTURE_TICKERS}
    monkeypatch.setattr(
        serve, "load_universe_context",
        lambda: (raw_ohlcv, ihsg_frame, all_ticker_data),
    )

    # Sentiment is stubbed to 0.0 rather than left alone.  The real
    # features.fetchers.fetch_news_sentiment() imports src/news_sentiment.py,
    # which (a) writes data/sentiment/<ticker>.json — a write into the real
    # data/ tree, and one that a second run would read back, making the test
    # depend on mutable on-disk state — and (b) calls
    # warnings.filterwarnings("ignore") process-wide, which would hide exactly
    # the warnings this suite escalates.  Stubbing keeps serving's own code path
    # (flag-gated call, value handed to the builder) while removing both.
    monkeypatch.setattr(fetchers, "fetch_news_sentiment", lambda ticker: 0.0)

    # A warm daily macro cache: the state every request after the first one sees,
    # and what makes usdidr_rate / bi_rate_official real rather than all-NaN.
    # Injecting it keeps the test offline (the real refresh_daily_cache()
    # downloads).
    index = raw_ohlcv.index
    monkeypatch.setattr(serve, "_usdidr_cache", pd.DataFrame(
        {
            "usdidr_rate": np.linspace(15_000.0, 15_600.0, len(index)),
            "usdidr_return": np.full(len(index), 0.0004),
        },
        index=index,
    ))
    monkeypatch.setattr(serve, "_fred_cache", pd.DataFrame(
        {"fed_rate": np.full(len(index), 5.25)}, index=index
    ))
    monkeypatch.setattr(serve, "_bi_cache",
                        pd.Series(6.0, index=index, name="bi_rate_official"))
    return serve


@pytest.fixture
def served_frame(injected, raw_ohlcv, fundamentals_dict) -> pd.DataFrame:
    """One served inference row for ``BBCA.JK``, built entirely from fixtures."""
    return injected.build_inference_features(
        raw_ohlcv["BBCA.JK"].dropna(), "BBCA.JK", fundamentals_dict
    )


def test_no_placeholder_nan_columns(served_frame):
    """No feature is a fabricated placeholder any more.

    The brief's assertion is ``all_nan == []``.  ``out`` is a one-row frame, so
    that reads as "the row contains no NaN at all" — which is false for training
    too (see ``PRE_EXISTING_NAN_COLUMNS``), and ``out`` cannot be narrowed
    without making the served column set depend on the row date and on how warm
    the macro cache is, which Task 1.4's ``align_to_schema`` rejects outright.
    The pre-existing set is pinned instead, so any *other* empty column still
    fails here: the 42 placeholder columns this task deletes would.
    """
    out = served_frame
    all_nan = [c for c in out.columns if out[c].isna().all()]
    unexpected = sorted(set(all_nan) - PRE_EXISTING_NAN_COLUMNS)
    assert unexpected == [], f"features still faked at inference: {unexpected}"


def test_audit_named_market_features_carry_real_values(served_frame):
    out = served_frame

    missing = [c for c in AUDIT_NAMED_MARKET_FEATURES if c not in out.columns]
    assert missing == [], f"model-used market features not built at inference: {missing}"

    faked = [c for c in AUDIT_NAMED_MARKET_FEATURES if pd.isna(out[c].iloc[0])]
    assert faked == [], f"model-used market features still NaN at inference: {faked}"


def test_volume_profile_family_is_absent_not_faked(served_frame):
    """Serving cannot obtain 5-minute intraday data, so it must not pretend to."""
    out = served_frame
    faked = [c for c in out.columns if c.startswith("vp_")]
    assert faked == [], f"volume-profile placeholders still emitted: {faked}"


def test_inference_frame_has_no_target_and_no_ticker(served_frame):
    out = served_frame
    assert "target" not in out.columns
    assert "ticker" not in out.columns


def test_cross_stock_window_matches_full_history(injected, raw_ohlcv,
                                                 fundamentals_dict, monkeypatch):
    """The bounded cross-stock window is an optimisation, not an approximation.

    Serving gives ``compute_cross_stock_features`` only the tail of the history
    because that pass costs ~29 s on the real universe, while the row it serves
    depends on at most the last six rows.  This pins that claim: with the window
    widened past the whole fixture, the served row must come out byte-identical.
    """
    single = raw_ohlcv["BBCA.JK"].dropna()
    default_window = injected.CROSS_STOCK_WINDOW_ROWS
    assert default_window < len(raw_ohlcv), "fixture is too small to exercise a window"

    monkeypatch.setattr(injected, "CROSS_STOCK_WINDOW_ROWS", len(raw_ohlcv) + 1)
    full_history = injected.build_inference_features(
        single, "BBCA.JK", fundamentals_dict)

    monkeypatch.setattr(injected, "CROSS_STOCK_WINDOW_ROWS", default_window)
    windowed = injected.build_inference_features(
        single, "BBCA.JK", fundamentals_dict)

    pd.testing.assert_frame_equal(windowed, full_history)


def test_universe_staleness_is_reported(tmp_path, monkeypatch):
    import serve

    # A universe file written 10 days and an hour ago.
    stocks = tmp_path / "stocks.csv"
    stocks.write_text("Ticker,Field\n")
    age_seconds = 10 * 86_400 + 3_600
    stamp = time.time() - age_seconds
    os.utime(stocks, (stamp, stamp))

    # A freshly written IHSG file: if universe_age_days() read the index's mtime
    # instead of the universe file's, this test would report 0 days.
    ihsg = tmp_path / "ihsg.csv"
    ihsg.write_text("Date,Close\n")

    monkeypatch.setattr(serve, "UNIVERSE_RAW_PATH", str(stocks))
    monkeypatch.setattr(serve, "IHSG_PATH", str(ihsg))

    assert serve.universe_age_days() >= 10


def test_universe_age_is_unknown_not_fresh_when_file_absent(tmp_path, monkeypatch):
    """An absent universe has no age; reporting 0 would claim it was just ingested."""
    import serve

    monkeypatch.setattr(serve, "UNIVERSE_RAW_PATH", str(tmp_path / "never-written.csv"))

    assert serve.universe_age_days() == -1
