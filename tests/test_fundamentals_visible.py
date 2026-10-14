"""Task 2.4 — a fundamentals outage is visible, and dividend yield is not fabricated.

Two defects, both the same family as F8 — **a safe default that is
indistinguishable from a real value** — and both made *certain* by Task 2.3's
retrain, which gave all 45 models a 60-name feature contract that
unconditionally requires five fundamentals and ``usdidr_return``:

1. **A Yahoo ``.info`` outage was invisible.**  ``fetch_fundamentals`` caught
   every exception and returned ``{key: np.nan}`` for the eight fields, and a
   ``.info`` that answered with *no fundamentals at all* — the real shape of
   which is ``{'trailingPegRatio': None}``, appended by
   ``Quote._fetch_complementary`` after an empty response — produced the *same*
   eight NaNs with no exception at all.  ``pipeline.inject_macro_features`` then
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

The macro half of the same family (§2D and §2F) is the third instance:
``inject_macro_features`` writes ``usdidr_return = 0.0`` when the USD/IDR frame
is absent, and the cache behind that frame is **cold until a background task has
run** — so the state a fresh process serves in, and the state the startup
pre-warm caches the day's busiest rows in, is the one the fabrication is certain
in.  §2F pins that cold start, the pre-warmed row served from the cache-hit
path, and the two directions of the availability predicate.

Every test here is offline and drives the **real** fetcher: only the transport is
stood in for, never ``fetch_fundamentals`` or the seam itself.  A stub that
replaces the fetcher could return whatever shape the test liked, which is
precisely how the F8 tests were vacuous before Task 1.9 rewrote them — and it is
also how the *first* version of the empty-response test here was vacuous: it
stubbed ``.info`` to ``{}``, a body the real property does not produce, so the
defect survived a test that claimed to cover it.  The no-data mode is therefore
driven through the real ``Quote``/``Ticker`` machinery
(``_no_fundamentals_ticker``) and its body is asserted, not assumed.
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
import asyncio
import json
import logging
import types
import warnings

import numpy as np
import pandas as pd
import pytest
import requests
import yfinance as yf

#: The real ``yf.Ticker``, captured before any test installs a stand-in.  See
#: ``_no_fundamentals_ticker`` for why the stand-ins must not subclass the
#: *current* attribute.
_REAL_TICKER = yf.Ticker

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
#: Named for the *fields*, not for an empty body: the body yfinance produces is
#: not empty (see ``_mode_empty_response`` below).
EMPTY_RESPONSE_REASON = "no fundamental data"

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

    A literal body is the right stand-in for every mode whose shape the *library*
    does not decide — a rate limit, a rejected key, a dead endpoint, an
    unparseable body.  It is the wrong one for the empty-200 mode, whose body
    ``Quote.info`` builds itself: that mode is driven by
    ``_no_fundamentals_ticker`` instead.
    """
    class _Ticker:
        def __init__(self, ticker):
            self.ticker = ticker

        @property
        def info(self):
            if isinstance(info, Exception):
                raise info
            return dict(info)

    monkeypatch.setattr(yf, "Ticker", _Ticker)


class _EmptyYahoo:
    """An in-memory Yahoo whose every fundamentals endpoint answers with nothing.

    Duck-typed against the two ``YfData`` methods ``Quote`` actually uses —
    ``get_raw_json`` and ``cache_get`` — rather than a mock of ``Quote``, so that
    the real ``Quote.info`` property runs: ``_fetch_info``,
    ``_fetch_additional_info`` and ``_fetch_complementary``, in that order,
    including the step that appends the complementary key to the body.  It is the
    *shape the production library produces* that this file's tests have to
    exercise, and only the library can produce it.

    Subclassing ``YfData`` would be wrong for a second reason beyond this one:
    ``YfData`` is a singleton, so a subclass instance would be handed back to
    other code that asks for one.
    """

    def __init__(self, *args, **kwargs):
        # ``YfData.__init__`` builds a requests session; nothing here talks to a
        # network, so the session is not needed.
        pass

    @staticmethod
    def _json(payload):
        import types

        return types.SimpleNamespace(text=json.dumps(payload))

    def get_raw_json(self, url, params=None, timeout=30):
        """``quoteSummary`` and v7 ``quote`` both answer 200 with ``result: []``."""
        if "quoteSummary" in url:
            return {"quoteSummary": {"result": []}}
        if "/v7/finance/quote" in url:
            return {"quoteResponse": {"result": []}}
        raise AssertionError(f"unexpected get_raw_json url: {url}")

    def cache_get(self, url, params=None, timeout=30):
        """The fundamentals-timeseries body, *lacking* ``trailingPegRatio``."""
        if "fundamentals-timeseries" in url:
            return self._json(
                {
                    "timeseries": {
                        "result": [{"meta": {"type": ["trailingPegRatio"]}}],
                        "error": None,
                    }
                }
            )
        raise AssertionError(f"unexpected cache_get url: {url}")


def _no_fundamentals_ticker(monkeypatch):
    """``yf.Ticker`` whose ``.info`` is what the real library returns for no data.

    The measured body is ``{'trailingPegRatio': None}``: ``_fetch_info`` leaves
    ``_info`` as ``{}`` (both result lists are empty), and ``info`` then runs
    ``_fetch_complementary`` (``yfinance/scrapers/quote.py:502-508``), which
    writes ``self.info[k] = None`` for its one key when the timeseries carries no
    value for it (``quote.py:755-760``).  The body is therefore **truthy**, which
    is why the predicate that stood here — ``if not info`` — read this response as
    healthy with all eight fields ``NaN``, five of them schema-required.

    A real ``yf.Ticker`` whose ``get_info`` returns
    ``Quote(<in-memory YfData>).info``: ``.info`` is the real property calling the
    real ``_fetch_info``/``_fetch_complementary``, so the body under test is one
    the *library* built.  Only the transport is in memory.

    The stand-in's base is :data:`_REAL_TICKER`, captured when this module was
    imported, rather than whatever ``yf.Ticker`` happens to be at call time.
    Several helpers in this module install their own ``yf.Ticker``, and pytest's
    ``monkeypatch`` restores them in reverse order — so a class that subclassed the
    *current* attribute would layer stub on stub, and the resulting chain is not
    the library anymore.
    """
    from yfinance.scrapers.quote import Quote

    class _Ticker(_REAL_TICKER):
        def get_info(self, proxy=None):
            return Quote(_EmptyYahoo(), self.ticker).info

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
    """The silent mode: a 200 that raises nothing and publishes no fundamentals.

    Driven through the **real** ``Quote``/``Ticker`` machinery, because this mode
    is the one whose body the library builds: what arrives is
    ``{'trailingPegRatio': None}``, not ``{}``.  A hand-built ``{}`` would let a
    predicate that depends on dict emptiness pass — which is exactly the defect
    the review found here.
    """
    _no_fundamentals_ticker(monkeypatch)


def _mode_hand_built_empty_body(monkeypatch):
    """The conservative case: a body that is literally ``{}``.

    The library does not produce this shape once ``_fetch_complementary``
    succeeds, so it is not the mode under test — but a body with no keys at all
    must still be read as unavailable, and a predicate keyed on the eight fetched
    fields has to answer that without being asked to.
    """
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
#: successful fetch and eight NaNs.  The last *two* are the same mode, driven at
#: the two shapes it can arrive in: the real one (a truthy body carrying only
#: ``trailingPegRatio``) and the conservative one (no keys at all).
FAILURE_MODES = [
    pytest.param(_mode_rate_limited, RATE_LIMITED_REASON, id="rate-limited"),
    pytest.param(_mode_rejected, REJECTED_REASON, id="rejected-key"),
    pytest.param(_mode_dead_endpoint, DEAD_ENDPOINT_REASON, id="dead-endpoint"),
    pytest.param(_mode_unparseable, UNPARSEABLE_REASON, id="unparseable-body"),
    pytest.param(_mode_unknown, UNKNOWN_REASON, id="unknown-exception"),
    pytest.param(_mode_empty_response, EMPTY_RESPONSE_REASON,
                 id="no-fundamental-fields"),
    pytest.param(_mode_hand_built_empty_body, EMPTY_RESPONSE_REASON,
                 id="empty-body"),
]

#: The two failure modes above that are the *same* reason, so the
#: "every mode has its own reason" test counts distinct modes rather than distinct
#: strings.
SAME_REASON_MODES = 2


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
    for prepare, expected in _modes():
        prepare(monkeypatch)
        _values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")
        assert available is False
        # The reason is pinned per mode, not merely counted: the two shapes of the
        # no-data mode share one reason *because they are one mode*, and every
        # other mode's reason has to carry its own text.
        assert expected in reason, (expected, reason)
        # ``expected``, not ``reason``: the two empty-body spellings collapse to
        # one entry here, which is the point — they are the same failure.
        reasons[expected] = reasons.get(expected, 0) + 1

    assert len(reasons) == len(FAILURE_MODES) - SAME_REASON_MODES + 1, reasons
    assert reasons[EMPTY_RESPONSE_REASON] == SAME_REASON_MODES, reasons
    assert all(
        count == 1 for key, count in reasons.items() if key != EMPTY_RESPONSE_REASON
    ), reasons


def test_the_real_empty_info_body_is_truthy_and_carries_no_fundamentals(monkeypatch):
    """The shape the library produces, measured — not a dict this test built.

    The review's finding, as an assertion: an in-memory Yahoo whose
    ``quoteSummary`` and v7-quote responses both carry ``result: []``, and whose
    timeseries body lacks ``trailingPegRatio``, produces the body
    ``{'trailingPegRatio': None}``.  It is truthy, so a predicate keyed on dict
    emptiness calls a fundamentals-free 200 healthy — and the eight fields are
    NaNs with five of them schema-required.  The seam must read the *fields*.

    Also pins that the reason names the fields rather than an "empty response":
    the response is not empty, which is why the old wording sent an operator
    looking for a body that never arrives.
    """
    import features.fetchers as fetchers

    _no_fundamentals_ticker(monkeypatch)
    info = yf.Ticker("BBCA.JK").info

    # The premise of the whole test: the body yfinance really builds here is not
    # empty, so ``not info`` cannot be the detector.
    assert info == {"trailingPegRatio": None}, info
    assert bool(info) is True, info

    values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")

    assert available is False, values
    assert reason == EMPTY_RESPONSE_REASON or EMPTY_RESPONSE_REASON in reason, reason
    assert list(values) == list(FUNDAMENTAL_KEYS), values
    assert all(pd.isna(v) for v in values.values()), values


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
    #
    # ``monkeypatch.setattr`` rather than ``dict.clear()``: clearing mutates
    # shared state that no fixture restores, so every *later* test in the session
    # would inherit the emptied dict — the session-leak anti-pattern this suite
    # documents on ``serve.model_cache`` in
    # ``tests/test_audit_regressions.py::_serving_client``.  Replacing the object
    # gives this test a clean cache and restores the original, contents and all,
    # at teardown.
    monkeypatch.setattr(serve, "model_cache", {})
    monkeypatch.setattr(serve, "_fundamentals_cache", {})
    # ``_macro_outcomes`` is written by ``refresh_daily_cache``, which is a
    # *background* task that really runs in these tests — ``_cache_date`` is
    # pinned to 1970 above, so it cannot return early.  That write is a fact about
    # this test's fetch, not about the next test's, and it is currently benign
    # only because the stubbed download yields a usable frame and hence a healthy
    # outcome.  A stub that made ``fetch_usdidr`` return an empty frame would
    # otherwise attach ``MACRO_NO_FRAME`` to every later test in the session —
    # including the ones that assert ``degraded == []``.  Replacing the dict gives
    # this test its own (empty, i.e. "no fetch has run yet") record and restores
    # the original, contents and all, at teardown.
    monkeypatch.setattr(serve, "_macro_outcomes", {})
    return TestClient(serve.app)


@pytest.fixture
def macro_cache(monkeypatch):
    """A warm, healthy daily USD/IDR cache — and the day's outcome that goes with it.

    ``inject_macro_features`` writes ``usdidr_return = 0.0`` when the frame is
    absent, and the serving paths report that in ``degraded`` as ``"macro"`` — so
    a test that asserts ``degraded == []`` has to install the state a *healthy*
    macro fetch leaves behind, exactly as the ``healthy_news`` fixture does for
    sentiment.  Without it such an assertion would be measuring this test
    process's cold macro cache instead of the code under test.

    ``_macro_outcomes`` is patched rather than left alone because
    ``_cache_date`` is pinned to 1970 by ``_client_for``: the outcome is the
    fetch's own record, and the test states it directly rather than depending on
    which day the suite happens to run.
    """
    import serve
    from datetime import date as _date

    index = pd.bdate_range("2024-01-01", periods=60)
    monkeypatch.setattr(serve, "_usdidr_cache", pd.DataFrame(
        {
            "usdidr_rate": np.linspace(15_000.0, 15_600.0, len(index)),
            "usdidr_return": np.full(len(index), 0.0004),
        },
        index=index,
    ))
    monkeypatch.setattr(serve, "_macro_outcomes", {str(_date.today()): None})
    return serve._usdidr_cache


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


def test_a_contract_with_no_fundamentals_gets_no_constant_nan_claim(
        tmp_path, monkeypatch, caplog):
    """The warning is about *this* contract, so a contract without them says less.

    ``_mark_fundamentals`` used to assert unconditionally that five
    schema-required inputs were constant NaN.  That is true of the current fleet
    and false of any model trained without fundamentals, and a warning that is
    false for a request is a warning an operator learns to distrust.  The
    ``degraded`` entry stays either way — a failed fetch is a failed fetch — so
    this test pins both halves at once: the softer wording *and* the entry that
    must not have been softened away with it.
    """
    import serve

    client = _client_for(tmp_path, monkeypatch, ["ret_1d"], _price_frame())
    _info_ticker(monkeypatch, {})
    assert not (set(["ret_1d"]) & set(serve.FUNDAMENTAL_KEYS)), "anti-vacuity"

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    assert "fundamentals" in response.json()["degraded"], response.json()

    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    fundamentals = [m for m in lines if "Fundamentals unavailable" in m]
    assert len(fundamentals) == 1, lines
    assert "contract names no fundamental column" in fundamentals[0], fundamentals
    assert "five schema-required inputs" not in fundamentals[0], fundamentals


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
        client, monkeypatch, caplog, healthy_news, macro_cache):
    """R4 at the response: a healthy ``.info`` gives ``degraded == []``.

    The value also reaches the model input: ``pe_ratio`` is the one column in the
    stub contract, and it must carry the fetched 18.4 rather than a NaN or a
    fabricated default.

    ``macro_cache`` is part of "healthy" now that a macro outage is named too: a
    response is only ``degraded == []`` when every component it reports on is
    genuinely reporting itself healthy, and asserting the empty list against a
    cold macro cache would be asserting that the macro check is silent rather
    than that the request is clean.
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


def test_a_healthy_request_logs_no_macro_warning(
        client, monkeypatch, caplog, healthy_news, macro_cache):
    """The negative control: a healthy macro fetch names nothing and warns nothing."""
    _info_ticker(monkeypatch, HEALTHY_INFO)

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    assert response.json()["degraded"] == [], response.json()
    noisy = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert [m for m in noisy if "macro" in m] == [], noisy


def test_the_testclient_import_wrapper_hides_nothing_else():
    """The one wrapped import captured the one known message and nothing more."""
    assert set(_IMPORT_WARNINGS) <= {EXPECTED_IMPORT_WARNING}, _IMPORT_WARNINGS
    assert len(_IMPORT_WARNINGS) <= 1, _IMPORT_WARNINGS


# ═══════════════════════════════════════════════════════════════════════════════
# 2D — the third instance: ``usdidr_return`` is *named*, and deliberately kept
# ═══════════════════════════════════════════════════════════════════════════════
# ``pipeline.inject_macro_features`` writes ``usdidr_rate = NaN`` and
# ``usdidr_return = 0.0`` when the USD/IDR fetch failed — and ``usdidr_return`` is
# in all 45 schemas, so it is the same question the ``dividend_yield`` fix
# answers.  Task 2.4's ruling splits it in two:
#
# * the **value** stays ``0.0``.  The committed training artefact carries real
#   ``usdidr_return`` values, so writing ``NaN`` at serving time would introduce a
#   *fresh* train/serve skew in a column the fleet consumes; changing it needs a
#   feature rebuild plus retrain (Phase 4, with the deferred ICT fix).
# * the **failure is made visible**: the serving paths put ``"macro"`` in
#   ``degraded`` and log the reason at ``warning``, so the fabricated ``0.0`` can
#   no longer be read as a measurement of a quiet rupiah.
#
# Both halves are pinned below, and the rationale lives in a comment at the
# branch itself (``src/features/pipeline.py``, the ``else`` on the USD/IDR join)
# so the residual is explicit at the code rather than only in this report.


def _macro_outage(monkeypatch, reason=None):
    """The state a *failed* USD/IDR fetch leaves: no usable frame, its outcome recorded."""
    import serve
    from datetime import date as _date

    monkeypatch.setattr(serve, "_usdidr_cache",
                        pd.DataFrame(columns=["usdidr_rate", "usdidr_return"]))
    monkeypatch.setattr(
        serve, "_macro_outcomes",
        {str(_date.today()): reason or serve.MACRO_NO_FRAME},
    )
    return serve


def test_a_macro_outage_is_named_in_degraded_and_the_zero_is_kept(
        tmp_path, monkeypatch, caplog, healthy_news):
    """R2's ruling, both halves at the response: named, and still ``0.0``.

    ``"usdidr_return"`` is the contract here, so the frame the model is fed
    carries the fallback and the assertion on it is about the *model input*
    rather than about an intermediate.  The response must carry ``"macro"`` in
    ``degraded`` and log the reason at ``warning`` — the same shape the
    fundamentals half uses — and the value must still be ``0.0``, because that is
    the ruling: visibility now, the value change with the Phase 4 rebuild that
    can also retrain.
    """
    import serve

    # A client of this test's own, with ``usdidr_return`` as the contract: a
    # column a schema does not name never reaches the model input, so the
    # anti-regression assertion below would otherwise measure nothing.
    client = _client_for(tmp_path, monkeypatch, ["usdidr_return"], _price_frame())

    _info_ticker(monkeypatch, HEALTHY_INFO)
    _macro_outage(monkeypatch)
    seen = _capture_model_input(monkeypatch, serve)

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert "macro" in body["degraded"], body
    # The anti-regression half of the ruling: the value is *not* substituted.
    assert "usdidr_return" in seen, seen
    assert seen["usdidr_return"] == 0.0, seen

    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("USD/IDR" in line and "0.0" in line for line in lines), lines


def test_usdidr_return_fallback_fabricates_a_zero():
    """Measured: an absent USD/IDR frame becomes a fabricated ``0.0`` return.

    The neighbouring line already distinguishes the honest spelling: on the
    *successful* fetch path, a hole in the series is filled with ``0``
    (``pipeline.py:101``), while the *failed* fetch path writes a constant
    ``0.0``.  The first loses a row; the second claims the rupiah did not move
    on a day nobody measured it.

    **This pins the residual, not its acceptance.**  Task 2.4's controller ruling
    kept the value and required only that the failure be visible (see
    ``test_a_macro_outage_is_named_in_degraded_and_the_zero_is_kept`` above and
    the comment at the branch); replacing the ``0.0`` with ``NaN`` needs a feature
    rebuild plus retrain plus the same Phase 4 experiment as the deferred ICT fix.
    This test fails the day that decision is taken, which is what makes the
    residual explicit rather than implied.
    """
    import features.pipeline as p

    frame = pd.DataFrame(
        {"Close": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2)
    )
    out = p.inject_macro_features(frame, {}, None, None, None)

    assert pd.isna(out["usdidr_rate"].iloc[-1]), out["usdidr_rate"]
    assert out["usdidr_return"].iloc[-1] == 0.0, out["usdidr_return"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2F — the state that occurs first: the day's fetch has not run yet
# ═══════════════════════════════════════════════════════════════════════════════
# ``serve._usdidr_cache`` starts as ``None``, and its only writer is
# ``refresh_daily_cache``, whose only callers are background tasks that run
# *after* the response.  A cold process therefore builds its first frame from the
# fallback (``usdidr_return = 0.0``, ``usdidr_rate = NaN``), and the startup
# pre-warm caches exactly such rows — keyed by ``date = today``, for the five
# busiest tickers.  Reading "no outcome recorded for today" as "no news" left all
# of that silent, and it is the one state in which the fabrication is *certain*
# rather than merely possible.
#
# Nothing is added to the cache's stored format to answer the cached-row half.
# The row carries its own witness: ``inject_macro_features`` writes a real
# ``usdidr_rate`` on the success branch (``ffill().bfill()``) and a constant
# ``NaN`` on the fallback branch, and ``usdidr_rate`` is in all 45 persisted
# contracts beside ``usdidr_return`` (measured).  That is what lets the *hit*
# path name the fabrication even after the day's fetch has succeeded — a
# live-state check could not, and a pre-warmed fallback row would look clean for
# the rest of the day.

#: The cold cache's reason, pinned as a literal substring so a rename fails here.
COLD_CACHE_REASON = "no USD/IDR fetch has produced a frame in this process yet"
#: The cached row's reason, likewise.
CACHED_FALLBACK_REASON = (
    "this cached row was built while the USD/IDR fetch had produced no frame"
)


def _warm_usdidr_frame() -> pd.DataFrame:
    """A USD/IDR frame with a *real* rate: what a successful fetch leaves behind.

    On the same business-day index as ``_price_frame``, so the join inside
    ``inject_macro_features`` overlaps and ``usdidr_rate`` is a measurement rather
    than NaN — the difference the cached row's own witness reads.
    """
    index = pd.bdate_range("2024-01-01", periods=60)
    return pd.DataFrame(
        {
            "usdidr_rate": np.linspace(15_000.0, 15_600.0, len(index)),
            "usdidr_return": np.full(len(index), 0.0004),
        },
        index=index,
    )


def test_a_cold_macro_cache_is_named_and_the_zero_is_kept(
        tmp_path, monkeypatch, caplog, healthy_news):
    """A frame built before the day's fetch reports ``"macro"``, and keeps ``0.0``.

    The cold start is *the* case the visibility contract exists for — the frame
    takes ``pipeline.py``'s fallback deterministically — so silence here is not
    "no news", it is the fabrication going unnamed on the day's first prediction.
    The value half is unchanged, as ruled: the model input still carries the
    fabricated ``0.0``, and it is the *response* that says so.
    """
    import serve

    client = _client_for(tmp_path, monkeypatch, ["usdidr_rate", "usdidr_return"])
    _info_ticker(monkeypatch, HEALTHY_INFO)
    # The cold start, stated rather than inherited from whatever the session left
    # on the module: no fetch has produced a frame in this process, and no outcome
    # is recorded for today.
    monkeypatch.setattr(serve, "_usdidr_cache", None)
    monkeypatch.setattr(serve, "_macro_outcomes", {})
    seen = _capture_model_input(monkeypatch, serve)

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    body = response.json()
    # Sentiment and fundamentals are driven healthy, so macro is the only
    # degradation and the entry can be asserted exactly.
    assert body["degraded"] == ["macro"], body

    # The value half of the ruling: still the fabricated 0.0, at the model input.
    assert "usdidr_return" in seen, seen
    assert seen["usdidr_return"] == 0.0, seen
    assert pd.isna(seen["usdidr_rate"]), seen

    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(COLD_CACHE_REASON in line and "0.0" in line for line in lines), lines
    # Its own named reason: this is not the recorded-outage mode, and an operator
    # must not be told a fetch failed when no fetch has run.
    assert not any(serve.MACRO_NO_FRAME in line for line in lines), lines


def test_a_row_cached_by_the_pre_warm_names_the_macro_fallback(
        tmp_path, monkeypatch, caplog, healthy_news):
    """A pre-warmed fallback row is not clean — not even after the fetch succeeds.

    ``_warmup_prediction_cache`` runs at startup, before any request and hence
    before ``refresh_daily_cache`` has ever run, and caches one row per priority
    ticker under ``date = today``.  Those rows hold the fabricated ``0.0`` for the
    rest of the day, and the hit path is what serves them.  Two states are driven
    on the *same* row:

    * the cold one it was built in, which is also the one its first hit is served
      in;
    * the **healthy** one that follows it.  The row's macro input was fixed when
      the row was built, so a fetch that later succeeds cannot turn it into a
      measurement — which is why the hit path reads the row and not today's cache.
    """
    import serve
    from datetime import date as _date

    client = _client_for(tmp_path, monkeypatch, ["usdidr_rate", "usdidr_return"])
    _info_ticker(monkeypatch, HEALTHY_INFO)
    monkeypatch.setattr(serve, "_usdidr_cache", None)
    monkeypatch.setattr(serve, "_macro_outcomes", {})

    # The real pre-warm, on a cold cache and an empty cache database.
    asyncio.run(serve._warmup_prediction_cache())
    schema = ["usdidr_rate", "usdidr_return"]
    assert serve._cache_get("BBCA.JK", schema) is not None, (
        "the pre-warm cached nothing, so this test would be measuring a miss"
    )

    seen = _capture_model_input(monkeypatch, serve)
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        cold_hit = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert cold_hit.status_code == 200, cold_hit.text
    assert cold_hit.json()["degraded"] == ["macro"], cold_hit.json()
    # The row the hit served is the fabricated one, so the assertion above is
    # about the prediction's own input rather than about the process's state.
    assert "usdidr_return" in seen, seen
    assert seen["usdidr_return"] == 0.0, seen
    assert pd.isna(seen["usdidr_rate"]), seen

    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(CACHED_FALLBACK_REASON in line for line in lines), lines

    # The day's fetch now succeeds; the row is not rebuilt and still holds the
    # fallback.  A caller must still be able to tell.
    monkeypatch.setattr(serve, "_macro_outcomes", {str(_date.today()): None})
    monkeypatch.setattr(serve, "_usdidr_cache", _warm_usdidr_frame())

    later_hit = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert later_hit.status_code == 200, later_hit.text
    assert later_hit.json()["degraded"] == ["macro"], later_hit.json()


def test_a_row_built_from_a_real_series_is_not_degraded_on_a_hit(
        tmp_path, monkeypatch, healthy_news):
    """The witness's negative control: a row with a real rate stays clean.

    Both paths are driven on the same contract — the slow path caches a row built
    from a real ``usdidr_rate``, and the hit path reads that row — so a witness
    that fired on any row (or on any row with a ``NaN`` anywhere) fails here.
    ``_macro_outcomes`` states the day's healthy outcome explicitly rather than
    relying on the session's, which is what makes ``degraded == []`` a statement
    about this request.
    """
    import serve
    from datetime import date as _date

    client = _client_for(tmp_path, monkeypatch, ["usdidr_rate", "usdidr_return"])
    _info_ticker(monkeypatch, HEALTHY_INFO)
    monkeypatch.setattr(serve, "_usdidr_cache", _warm_usdidr_frame())
    monkeypatch.setattr(serve, "_macro_outcomes", {str(_date.today()): None})

    miss = client.post("/predict", json={"ticker": "BBCA.JK"})
    assert miss.status_code == 200, miss.text
    assert miss.json()["degraded"] == [], miss.json()

    hit = client.post("/predict", json={"ticker": "BBCA.JK"})
    assert hit.status_code == 200, hit.text
    assert hit.json()["degraded"] == [], hit.json()


# ═══════════════════════════════════════════════════════════════════════════════
# 2G — the availability predicate reads *values*, not key presence
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_body_of_present_nulls_is_not_a_measurement(monkeypatch):
    """Eight fields, all *present*, all ``None``: nothing was published.

    Defensive rather than reachable: yfinance drops ``None`` values from the
    flattened body (``yfinance/scrapers/quote.py:650``), so a body that spells
    every field out as ``None`` is not one this version produces — which is
    exactly why it went unnoticed.  ``any(field in info ...)`` read it as a
    measurement and returned ``available=True`` with all eight values NaN, the
    original defect one key-shape away; reading the *values* closes it.  The
    partially-populated tests below pin the other direction, so this cannot be
    "fixed" by requiring all eight.
    """
    import features.fetchers as fetchers

    body = {field: None for field in HEALTHY_INFO}
    assert sorted(body) == sorted(HEALTHY_INFO), body  # anti-vacuity
    _info_ticker(monkeypatch, body)

    values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")

    assert available is False, values
    assert list(values) == list(FUNDAMENTAL_KEYS), values
    assert all(pd.isna(v) for v in values.values()), values
    assert reason is not None and EMPTY_RESPONSE_REASON in reason, reason


def test_a_lone_present_null_is_not_a_measurement(monkeypatch):
    """One field present as ``None`` and nothing else is the no-data mode.

    The single-field spelling of the body above: a ``None`` is not an answer, so
    it must not be the one key that makes a fundamentals-free body look healthy.
    """
    import features.fetchers as fetchers

    _info_ticker(monkeypatch, {"trailingPE": None})

    values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")

    assert available is False, values
    assert all(pd.isna(v) for v in values.values()), values
    assert reason is not None and EMPTY_RESPONSE_REASON in reason, reason


def test_a_partially_populated_body_is_still_a_measurement(monkeypatch):
    """Two of the eight fields published is a healthy fetch, not an outage.

    The blast-radius pin for the hardening: a body carrying a *value* for any
    field is an answer, so an implementation that required all eight would fail
    here — a partially-populated healthy response must not become a false
    negative.
    """
    import features.fetchers as fetchers

    _info_ticker(monkeypatch, {"trailingPE": 18.4, "marketCap": 1.15e12})

    values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")

    assert available is True, reason
    assert reason is None
    assert values["pe_ratio"] == 18.4, values
    assert values["market_cap"] == 1.15e12, values
    assert all(
        pd.isna(values[key]) for key in FUNDAMENTAL_KEYS
        if key not in ("pe_ratio", "market_cap")
    ), values


def test_a_published_zero_is_a_value_not_an_absence(monkeypatch):
    """``0.0`` published by Yahoo is a measurement, so the predicate is not truthy.

    ``is not None`` rather than a truthiness test: ``dividendYield: 0.0`` is the
    one value the old ``or 0.0`` fallback could not distinguish from its own
    default, and a predicate that asked ``if not info.get(field)`` would
    re-introduce that family one layer up.
    """
    import features.fetchers as fetchers

    _info_ticker(monkeypatch, {"dividendYield": 0.0})

    values, available, reason = fetchers.fetch_fundamentals_status("BBCA.JK")

    assert available is True, reason
    assert values["dividend_yield"] == 0.0, values
