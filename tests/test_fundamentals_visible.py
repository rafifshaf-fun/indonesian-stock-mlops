"""Task 2.4 — a fundamentals outage is visible, and dividend yield is not fabricated.

Two defects, both the same family as F8 — **a safe default that is
indistinguishable from a real value** — and both made *certain* by Task 2.3's
retrain, which gave all 45 models a 60-name feature contract that
unconditionally requires five fundamentals and ``usdidr_return``:

1. **A Yahoo ``.info`` outage was invisible.**  ``fetch_fundamentals`` caught
   every exception and returned ``{key: np.nan}`` for the eight fields, and an
   ``.info`` that answers with an empty dict produced the *same* eight NaNs
   with no exception at all.  ``pipeline.inject_macro_features`` then
   materialises the five schema-required fundamentals as constant-NaN columns,
   and ``align_to_schema`` checks **presence only** — so the request answered
   HTTP 200, a real probability, and ``degraded: []``.  Nothing in the response
   said five of the sixty inputs were missing.
2. **``dividend_yield`` fabricated a zero.**  ``info.get("dividendYield",
   np.nan) or 0.0`` read yfinance's "not published" as a real ``0.0`` in a
   column all 45 models consume.  XGBoost reads ``0.0`` and ``NaN``
   differently, so this did not merely lose information: it fed the model a
   claim that the ticker pays nothing.

The seam is Task 1.9's shape, not a new mechanism:
``fetch_fundamentals_status(ticker) -> (values, available, reason)`` says
whether the eight values are a *measurement*, one distinct reason string per
failure mode, so the serving path reads a status instead of catching an
exception the real code never raises.  ``degraded`` names ``"fundamentals"``
and the reason is logged at ``warning``.

Every test here is offline and drives the **real** fetcher: only ``yf.Ticker``
(the network) is stood in for, never ``fetch_fundamentals`` or the seam itself.
A stub that replaces the fetcher could return whatever shape the test liked,
which is precisely how the F8 tests were vacuous before Task 1.9 rewrote them.
``fetch_fundamentals``' four existing stubs elsewhere in the suite are moved to
the same seam for the same reason: with the status seam in the serving path, a
stub on the value-only function would be dead code and the tests would silently
stop covering the fundamentals fetch at all.

``serve`` imports mlflow → opentelemetry, whose runtime-context loader still
calls the removed ``importlib.metadata`` dict interface on Python 3.11, and
FastAPI raises its own deprecation for ``@app.on_event("startup")``; the
TestClient import raises a third.  All three are upstream of this repository, so
the filters below are scoped to their exact text and to this module only,
exactly as ``tests/test_degradation_visible.py`` does.
"""
import json
import logging
import types
import warnings

import numpy as np
import pandas as pd
import pytest
import requests

# ``fastapi.testclient`` → ``starlette.testclient`` emits a collection-time
# ``StarletteDeprecationWarning`` when ``httpx2`` is absent.  A module
# ``pytestmark`` cannot catch it: the filter would have to be installed before
# the statement below, which is the statement that raises.  The import is
# wrapped narrowly and the captured messages are pinned, so the wrapper cannot
# hide a different starlette warning.
with warnings.catch_warnings(record=True) as _import_warnings:
    warnings.simplefilter("always")
    from fastapi.testclient import TestClient

_IMPORT_WARNINGS = [str(w.message) for w in _import_warnings]

#: The exact message the wrapped import is allowed to capture.
EXPECTED_IMPORT_WARNING = (
    "Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` "
    "instead."
)

# ``src/news_sentiment.py`` calls ``warnings.filterwarnings("ignore")`` at import
# time, which would otherwise disable the suite's ``filterwarnings = ["error"]``
# for every test collected after it; ``catch_warnings`` discards that global
# ignore when the block exits.  Imported here rather than inside a test so the
# one import-time "vaderSentiment is not installed" notice belongs to collection
# instead of to whichever test happens to touch the seam first.
with warnings.catch_warnings():
    import news_sentiment

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:SelectableGroups dict interface is deprecated:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:(?s).*on_event is deprecated.*:DeprecationWarning"
    ),
]

#: The eight keys ``fetch_fundamentals`` has always returned.
FUNDAMENTAL_KEYS = (
    "pe_ratio", "pb_ratio", "dividend_yield", "market_cap",
    "debt_to_equity", "return_on_equity", "revenue_growth", "profit_margins",
)

#: A healthy ``.info``.  ``dividendYield`` is a real, non-zero yield, and the
#: values are distinct so a key/field mix-up cannot pass by coincidence.
HEALTHY_INFO = {
    "trailingPE": 18.4,
    "priceToBook": 3.1,
    "dividendYield": 0.027,
    "marketCap": 1.15e12,
    "debtToEquity": 0.62,
    "returnOnEquity": 0.21,
    "revenueGrowth": 0.083,
    "profitMargins": 0.34,
}

HEALTHY_VALUES = {
    "pe_ratio": 18.4,
    "pb_ratio": 3.1,
    "dividend_yield": 0.027,
    "market_cap": 1.15e12,
    "debt_to_equity": 0.62,
    "return_on_equity": 0.21,
    "revenue_growth": 0.083,
    "profit_margins": 0.34,
}

#: The reason each failure mode must be named with, as a substring.  Literals
#: rather than imports from the module under test, so this table is a
#: *behavioural* pin: renaming a reason stops the overlay's log line matching it
#: and fails here rather than passing by construction.
RATE_LIMITED_REASON = "rate limit"
REJECTED_REASON = "HTTP 401"
UNPARSEABLE_REASON = "not JSON"
DEAD_ENDPOINT_REASON = "ConnectionError"
UNKNOWN_REASON = "RuntimeError"
EMPTY_RESPONSE_REASON = "empty response"

#: ``serve.build_inference_features`` reads this module-level path, and the
#: ``universe_context`` fixture below replaces the loader that reads it.
SCHEMA = ["pe_ratio"]


# ═══════════════════════════════════════════════════════════════════════════════
# Driving the real fetcher — the network stands in, nothing else does
# ═══════════════════════════════════════════════════════════════════════════════

def _info_ticker(monkeypatch, info):
    """Point ``yf.Ticker(...).info`` at a literal dict, through the real module.

    ``features.fetchers`` and ``serve`` import the same ``yfinance`` module
    object, so patching its ``Ticker`` attribute here is visible to both paths —
    and ``fetch_fundamentals``/``fetch_fundamentals_status`` run for real, which
    is the point.  Nothing in this module replaces the fetcher itself.
    """
    import yfinance as yf

    class _Ticker:
        def __init__(self, ticker):
            self.ticker = ticker

        @property
        def info(self):
            if isinstance(info, Exception):
                raise info
            return dict(info)

    monkeypatch.setattr(yf, "Ticker", _Ticker)


def _raise_info(monkeypatch, exc):
    """``.info`` raising ``exc``: the shape every Yahoo failure arrives in."""
    _info_ticker(monkeypatch, exc)


def _http_error(status_code):
    """A ``requests.HTTPError`` carrying a status, as yfinance's client raises.

    yfinance lets ``requests``' ``HTTPError`` out of ``_fetch`` when Yahoo
    answers a non-200 (``yfinance/scrapers/quote.py:597``), and the status is the
    only thing that separates a 429 from a 401 — so the reason string has to read
    it.
    """
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(f"HTTP {status_code}", response=response)


def _mode_rate_limited(monkeypatch):
    _raise_info(monkeypatch, _http_error(429))


def _mode_rejected(monkeypatch):
    _raise_info(monkeypatch, _http_error(401))


def _mode_dead_endpoint(monkeypatch):
    _raise_info(monkeypatch, requests.ConnectionError("connection refused"))


def _mode_unparseable(monkeypatch):
    _raise_info(monkeypatch, json.JSONDecodeError("Expecting value", "<html>", 0))


def _mode_unknown(monkeypatch):
    _raise_info(monkeypatch, RuntimeError("something new broke"))


def _mode_empty_response(monkeypatch):
    """The silent mode: Yahoo answers 200 with an empty body and raises nothing."""
    _info_ticker(monkeypatch, {})


def _modes():
    """``FAILURE_MODES`` as ``(prepare, expected_reason_substring)`` pairs.

    ``pytest.param``'s ``.values`` is the argument tuple, so the table can be
    read both by ``parametrize`` and by the plain loops below from one place.
    """
    return [(param.values[0], param.values[1]) for param in FAILURE_MODES]


#: Every production failure mode of the fundamentals fetch.  Only the first five
#: raise; the last one is the mode that makes the old ``except``-based detection
#: structurally insufficient — a caller that caught an exception would see a
#: successful fetch and eight NaNs.
FAILURE_MODES = [
    pytest.param(_mode_rate_limited, RATE_LIMITED_REASON, id="rate-limited"),
    pytest.param(_mode_rejected, REJECTED_REASON, id="rejected-key"),
    pytest.param(_mode_dead_endpoint, DEAD_ENDPOINT_REASON, id="dead-endpoint"),
    pytest.param(_mode_unparseable, UNPARSEABLE_REASON, id="unparseable-body"),
    pytest.param(_mode_unknown, UNKNOWN_REASON, id="unknown-exception"),
    pytest.param(_mode_empty_response, EMPTY_RESPONSE_REASON, id="empty-response"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 2E — the seam exists, reports each failure mode, and keeps a distinct reason
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("prepare, expected", _modes())
def test_the_seam_reports_every_real_failure_mode(monkeypatch, prepare, expected):
    """``fetch_fundamentals_status`` returns ``(values, available, reason)``.

    ``available is False`` for every failure mode — *including* the mode that
    raises nothing — and the eight values are NaNs rather than the fabricated
    zeros of the old ``dividend_yield`` fallback.  The reason names the cause.
    Parametrized rather than looped so one broken mode cannot hide the rest.
    """
    import features.fetchers as fetchers

    prepare(monkeypatch)
    values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")

    assert available is False, values
    assert list(values) == list(FUNDAMENTAL_KEYS), values
    assert all(pd.isna(v) for v in values.values()), values
    assert reason is not None and expected in reason, reason


def test_every_failure_mode_has_its_own_reason(monkeypatch):
    """Distinct modes are distinct strings, not one generic "unavailable".

    This is the shape Task 1.9 pinned for sentiment and the reason a 429 must
    not be reported as a dead key: an operator reading ``degraded`` has to be
    able to tell a rate limit from an outage from a broken response body.
    """
    import features.fetchers as fetchers

    reasons = {}
    for prepare, _expected in _modes():
        prepare(monkeypatch)
        _values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")
        assert available is False
        reasons[reason] = reasons.get(reason, 0) + 1

    assert len(reasons) == len(FAILURE_MODES), reasons
    assert all(count == 1 for count in reasons.values()), reasons


# ═══════════════════════════════════════════════════════════════════════════════
# 2A — a real outage reaches ``degraded`` through the serving path
# ═══════════════════════════════════════════════════════════════════════════════

class _StubModel:
    """The two-method surface every predicting path uses."""

    def predict(self, frame):
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[0.4, 0.6]])


def _price_frame(rows: int = 60) -> pd.DataFrame:
    """A minimal OHLCV frame long enough for the builder's 50-row floor."""
    index = pd.bdate_range("2024-01-01", periods=rows)
    price = np.arange(float(rows)) + 100.0
    return pd.DataFrame(
        {
            "Open": price,
            "High": price + 0.5,
            "Low": price - 0.5,
            "Close": price,
            "Volume": np.full(rows, 1_000_000.0),
        },
        index=index,
    )


@pytest.fixture
def universe_context(monkeypatch):
    """The universe loader, pinned to the synthetic frame the download returns.

    ``serve`` caches the real ``data/raw/`` snapshot on the module and a bare
    assignment would leak into every later test in the session, so the loader
    itself is patched.  There is one ticker, so the cross-stock family is not
    applied — that is not what these tests are about.
    """
    import serve

    frame = _price_frame()
    monkeypatch.setattr(serve, "load_universe_context",
                        lambda: (pd.DataFrame(), pd.DataFrame(), {}))
    return frame


def _client_for(tmp_path, monkeypatch, schema, frame=None):
    """A ``TestClient`` over ``serve.app``, offline, with ``schema`` as the contract.

    ``schema`` is a parameter because the contract decides what is *observable*:
    a column the schema does not name never reaches the model input, so a defect
    in it cannot be measured (and, worse, an assertion about it passes for the
    wrong reason).  Tests that measure a specific column name it here.
    """
    import serve

    prices = _price_frame() if frame is None else frame
    monkeypatch.setattr(serve, "PRED_CACHE_DB", str(tmp_path / "cache.db"))
    monkeypatch.setattr(serve, "load_best_model",
                        lambda ticker: (_StubModel(), list(schema)))
    monkeypatch.setattr(serve, "_cache_date", "1970-01-01")
    monkeypatch.setattr(serve.yf, "download", lambda *a, **k: prices.copy())
    # Two module-level caches key the *values* by (ticker, today) and are not
    # per-request: ``model_cache`` holds the model and its contract, and
    # ``_fundamentals_cache`` holds the last fundamentals fetch of the day.  A
    # leftover entry from an earlier test would answer the fetch this test is
    # driving and the ``.info`` stub below would never be reached — which is
    # exactly how a stale 0.0 (or a stale NaN) can be measured as if it were
    # this request's fetch.
    serve.model_cache.clear()
    serve._fundamentals_cache.clear()
    return TestClient(serve.app)


def _capture_model_input(monkeypatch, serve) -> dict:
    """Record the one-row frame the model is about to be fed.

    ``serve._prediction_frame`` is the single construction every predicting path
    goes through (Task 1.7), so wrapping it captures exactly the model input —
    the artefact this task's defect is about — without replacing any feature
    logic.  The real function still runs and its result is still returned.
    """
    seen: dict = {}
    real = serve._prediction_frame

    def capture(feature_row, schema):
        frame = real(feature_row, schema)
        seen.clear()
        seen.update({str(name): frame[name].iloc[0] for name in frame.columns})
        return frame

    monkeypatch.setattr(serve, "_prediction_frame", capture)
    return seen


@pytest.fixture
def client(tmp_path, monkeypatch, universe_context):
    """A ``TestClient`` over ``serve.app``, offline, with ``SCHEMA`` as the contract."""
    return _client_for(tmp_path, monkeypatch, SCHEMA, universe_context)


def test_a_fundamentals_outage_is_degraded_and_named(client, monkeypatch, caplog):
    """The defect, end to end: an ``.info`` outage used to be ``degraded: []``.

    ``BBCA.JK`` is not cached, so the request takes the slow path that fetches
    fundamentals.  The response must still be a real prediction — fundamentals
    are five inputs, not a blocker — but it must carry ``"fundamentals"`` in
    ``degraded`` and log the reason at ``warning``, so a client and an operator
    can both tell this apart from a healthy request.
    """
    _info_ticker(monkeypatch, {})

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert "fundamentals" in body["degraded"], body
    assert body["prediction"] in (0, 1), body
    assert "sentiment" in body["degraded"] or body["degraded"] == ["fundamentals"], body

    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "Fundamentals unavailable" in line and EMPTY_RESPONSE_REASON in line
        for line in lines
    ), lines


def test_the_outage_names_the_reason_in_the_response_path(client, monkeypatch):
    """A raising ``.info`` is degraded too, and the reason is the one raised."""
    _raise_info(monkeypatch, _http_error(429))

    response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    assert "fundamentals" in response.json()["degraded"], response.json()


def test_the_degradation_survives_the_daily_fundamentals_cache(client, monkeypatch):
    """The status is cached *with* the values, so the fast path cannot lose it.

    A second request for the same ticker the same day is answered from
    ``serve._fundamentals_cache`` rather than from Yahoo — and the eight cached
    numbers cannot say whether they were a measurement (five are ``NaN`` for an
    outage and for a field Yahoo does not publish alike).  So the availability has
    to be cached beside them.  The second request below also takes the *prediction*
    cache's fast path, which is the path that never builds features at all, so
    between the two caches this is the case where the response has the least
    evidence to work from.

    The ``.info`` stub is switched to a healthy body before the second request on
    purpose: if the response only looked healthy because it refetched, this test
    would pass while the cache silently dropped the status.
    """
    _info_ticker(monkeypatch, {})

    first = client.post("/predict", json={"ticker": "BBCA.JK"})
    assert first.status_code == 200, first.text
    assert "fundamentals" in first.json()["degraded"], first.json()

    _info_ticker(monkeypatch, HEALTHY_INFO)
    second = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert second.status_code == 200, second.text
    assert "fundamentals" in second.json()["degraded"], second.json()


# ═══════════════════════════════════════════════════════════════════════════════
# 2B — the fabricated zeros are gone, at the frame
# ═══════════════════════════════════════════════════════════════════════════════

def _fundamentals_frame(values: dict) -> pd.DataFrame:
    """``values`` through the real ``inject_macro_features``, as one column each.

    ``inject_macro_features`` is where the eight fields become columns
    (``df[col] = val``, a scalar each), so this is the last pure step before the
    values meet the model's contract.
    """
    import features.pipeline as p

    return p.inject_macro_features(
        pd.DataFrame({"Close": [1.0]}, index=pd.bdate_range("2024-01-01", periods=1)),
        values, None, None, None,
    )


def test_an_explicit_none_dividend_yield_reaches_the_frame_as_nan(
        monkeypatch, tmp_path):
    """``dividendYield: None`` — yfinance's "unknown" — is ``NaN``, not ``0.0``.

    This is the fabrication measured: ``None`` is falsy, so ``... or 0.0``
    returned ``0.0`` for a field Yahoo had just said it does not have, and the
    column all 45 models consume then carried a claim that the ticker pays
    nothing.  The assertion is on the value the **model input** carries: the
    request runs through the real serving path with ``dividend_yield`` in the
    contract, and what is read back is the aligned one-row frame
    ``align_to_schema`` produced.  A contract that omitted ``dividend_yield``
    would silently drop the column instead of measuring it, which is why the
    contract here names it.
    """
    import serve

    _info_ticker(monkeypatch, {**HEALTHY_INFO, "dividendYield": None})
    client = _client_for(tmp_path, monkeypatch, ["pe_ratio", "dividend_yield"])
    seen = _capture_model_input(monkeypatch, serve)

    response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    # Anti-vacuity: the contract names the column, so it reached the model input
    # rather than being dropped by the alignment.
    assert "dividend_yield" in seen, seen
    assert pd.isna(seen["dividend_yield"]), seen


def test_a_none_dividend_yield_is_not_reported_as_a_measurement(monkeypatch):
    """The fetcher reports the ``None`` spelling as absent, not as a zero.

    The seam's own half of the fix: the value is ``NaN`` *and* the fetch is still
    a successful measurement of the other seven fields, so ``available`` stays
    ``True``.  Conflating "Yahoo does not publish this field" with "the fetch
    failed" would send an operator after an outage that did not happen.
    """
    import features.fetchers as fetchers

    _info_ticker(monkeypatch, {**HEALTHY_INFO, "dividendYield": None})

    values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")

    assert available is True, reason
    assert reason is None
    assert pd.isna(values["dividend_yield"]), values
    assert values["dividend_yield"] != 0.0, values

    frame = _fundamentals_frame(values)
    assert pd.isna(frame["dividend_yield"].iloc[-1]), frame["dividend_yield"]


def test_a_missing_dividend_yield_key_reaches_the_frame_as_nan(monkeypatch):
    """The absent-key spelling of the same unknown is ``NaN`` too.

    yfinance drops unknown fields from ``info`` entirely rather than publishing a
    null (``yfinance/scrapers/quote.py:650``), so this is the shape production
    actually produces for a ticker whose yield is unknown — and the default in
    ``info.get`` was already ``np.nan``, which is why ``or 0.0`` was the only
    thing that could fabricate a zero, and why its removal has to leave this case
    exactly as it was.
    """
    import features.fetchers as fetchers

    info = {k: v for k, v in HEALTHY_INFO.items() if k != "dividendYield"}
    _info_ticker(monkeypatch, info)

    values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")
    assert available is True, reason
    assert pd.isna(values["dividend_yield"]), values

    frame = _fundamentals_frame(values)
    assert pd.isna(frame["dividend_yield"].iloc[-1]), frame["dividend_yield"]


def test_a_published_zero_dividend_yield_is_left_as_published(monkeypatch):
    """A ``0.0`` that reaches us from ``info`` is passed through unchanged.

    R1's fix is the removal of the truthiness fallback, not a reclassification:
    ``or 0.0`` could not tell a published ``0.0`` from the default it had just
    supplied, and this half of the fix is what keeps that ``0.0`` a *value*
    rather than converting every zero into a missing one.  The committed
    artefact is the evidence that this branch is not the reachable one today —
    it carries **no** exact ``0.0`` in ``dividend_yield`` across 63,996 rows, and
    all seven non-payers are ``NaN``, which only the ``get`` default produces
    (see the report's §2B).  The behaviour is pinned anyway, because a fix that
    silently reclassifies a real measurement is the same defect mirrored.
    """
    import features.fetchers as fetchers

    _info_ticker(monkeypatch, {**HEALTHY_INFO, "dividendYield": 0.0})

    values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")
    assert available is True, reason
    assert values["dividend_yield"] == 0.0, values

    frame = _fundamentals_frame(values)
    assert frame["dividend_yield"].iloc[-1] == 0.0, frame["dividend_yield"]


def test_a_failed_fetch_fabricates_no_zero_either(monkeypatch):
    """The failure path's eight NaNs are NaNs for ``dividend_yield`` too.

    The old failure branch already returned ``np.nan`` for all eight keys, so
    this pins that the seam did not regress it while removing the ``or 0.0``
    from the success branch.
    """
    import features.fetchers as fetchers

    _info_ticker(monkeypatch, {})

    values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")
    assert available is False and reason
    assert pd.isna(values["dividend_yield"]), values


# ═══════════════════════════════════════════════════════════════════════════════
# 2C — the success path is provably unchanged, and healthy is not degraded
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_healthy_info_yields_exactly_the_same_eight_values(monkeypatch):
    """R4: same inputs, same eight values, ``available is True``, no reason.

    The fleet consumes these values, so a change on the healthy path would
    silently alter every prediction.  The comparison is against the literal dict
    above rather than against the implementation, so an accidental key rename or
    field swap fails here.
    """
    import features.fetchers as fetchers

    _info_ticker(monkeypatch, HEALTHY_INFO)

    values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")

    assert available is True
    assert reason is None
    assert list(values) == list(FUNDAMENTAL_KEYS), list(values)
    assert values == HEALTHY_VALUES, values


def test_the_value_only_fetcher_is_unchanged_for_existing_callers(monkeypatch):
    """``fetch_fundamentals`` keeps its signature and its behaviour.

    Training (``build_feature_set``), ``predict.py`` and ``cli.py`` all call it
    and need a plain dict, so it stays — as the value-only view of the seam, the
    way ``fetch_news_sentiment`` is the value-only view of its status companion.
    """
    import features.fetchers as fetchers

    _info_ticker(monkeypatch, HEALTHY_INFO)
    assert fetchers.fetch_fundamentals("BBCA.JK") == HEALTHY_VALUES

    _info_ticker(monkeypatch, _http_error(429))
    failed = fetchers.fetch_fundamentals("BBCA.JK")
    assert list(failed) == list(FUNDAMENTAL_KEYS), failed
    assert all(pd.isna(v) for v in failed.values()), failed


@pytest.fixture
def healthy_news(monkeypatch, tmp_path):
    """The real news module in the state a healthy production fetch produces.

    The sentiment seam is not what this task changes, but a *healthy-serving*
    assertion is only meaningful if the other component reporting itself healthy
    is genuinely healthy — otherwise ``degraded == []`` would be unreachable and
    the pin would be weakened to tolerate a standing failure.
    """
    class _Analyzer:
        def polarity_scores(self, text):
            return {"compound": 0.4}

    class _Response:
        status_code = 200

        def json(self):
            return {"articles": [{"title": "Bank BCA profit rises"}]}

    monkeypatch.setattr(news_sentiment, "CACHE_DIR", str(tmp_path / "sentiment"))
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "test-key")
    monkeypatch.setattr(news_sentiment, "VADER_AVAILABLE", True)
    monkeypatch.setattr(
        news_sentiment, "SentimentIntensityAnalyzer", _Analyzer, raising=False
    )
    monkeypatch.setattr(news_sentiment, "_sentiment_analyzer", None)
    monkeypatch.setattr(
        news_sentiment, "requests",
        types.SimpleNamespace(get=lambda *a, **k: _Response()),
    )
    return news_sentiment


def test_a_healthy_request_is_not_degraded_and_carries_real_fundamentals(
        client, monkeypatch, caplog, healthy_news):
    """R4 at the response: a healthy ``.info`` gives ``degraded == []``.

    The value also reaches the model input: ``pe_ratio`` is the one column in the
    stub contract, and it must carry the fetched 18.4 rather than a NaN or a
    fabricated default.
    """
    import serve

    _info_ticker(monkeypatch, HEALTHY_INFO)
    seen = _capture_model_input(monkeypatch, serve)

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["degraded"] == [], body
    assert seen["pe_ratio"] == pytest.approx(HEALTHY_VALUES["pe_ratio"])

    noisy = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert [m for m in noisy if "Fundamentals unavailable" in m] == [], noisy


def test_the_testclient_import_wrapper_hides_nothing_else():
    """The one wrapped import captured the one known message and nothing more."""
    assert set(_IMPORT_WARNINGS) <= {EXPECTED_IMPORT_WARNING}, _IMPORT_WARNINGS
    assert len(_IMPORT_WARNINGS) <= 1, _IMPORT_WARNINGS


# ═══════════════════════════════════════════════════════════════════════════════
# 2D — the third instance, measured: ``usdidr_return``
# ═══════════════════════════════════════════════════════════════════════════════
# ``pipeline.inject_macro_features`` writes ``usdidr_rate = NaN`` and
# ``usdidr_return = 0.0`` when the USD/IDR fetch failed — and ``usdidr_return`` is
# in all 45 schemas, so it is the same question the ``dividend_yield`` fix
# answers.  The test below *measures* the current behaviour instead of assuming
# it, so the report's determination rests on a run rather than on a reading, and
# so a future change to this branch fails here rather than passing unremarked.

def test_usdidr_return_fallback_fabricates_a_zero():
    """Measured: an absent USD/IDR frame becomes a fabricated ``0.0`` return.

    The neighbouring line already distinguishes the honest spelling: on the
    *successful* fetch path, a hole in the series is filled with ``0``
    (``pipeline.py:101``), while the *failed* fetch path writes a constant
    ``0.0``.  The first loses a row; the second claims the rupiah did not move
    on a day nobody measured it.

    This pins the finding, not its acceptance: it is the evidence the task's R2
    determination rests on, and ``usdidr_return``'s fix is a separate, tested
    change (see the report's §R2).
    """
    import features.pipeline as p

    frame = pd.DataFrame(
        {"Close": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2)
    )
    out = p.inject_macro_features(frame, {}, None, None, None)

    assert pd.isna(out["usdidr_rate"].iloc[-1]), out["usdidr_rate"]
    assert out["usdidr_return"].iloc[-1] == 0.0, out["usdidr_return"]
