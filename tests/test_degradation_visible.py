"""Task 1.9 — dead calibration removed, degradation visible (F8).

Three failures share one shape: **a component fails and the caller is told the
run succeeded.**

* **The calibration block never ran.**  It called ``CalibratedCVClassifier``, a
  name that exists nowhere, so the ``NameError`` was swallowed by a bare
  ``except Exception``, the run logged ``calibrated=False``, and ``README.md``
  advertised isotonic calibration as a feature.  The block is deleted and the
  README corrected; the absence is asserted structurally (see ``_ast_symbols``)
  rather than by scanning the source text, so a comment may name the symbol it
  explains — a raw text scan is the landmine Task 1.5 left in place.
* **A dead news API was indistinguishable from neutral news.**  The sentiment
  overlay's ``except Exception: pass`` left ``sentiment_score == 0.0``, which is
  also the honest reading of "the news was neutral".  ``PredictResponse`` now
  carries ``degraded``, and a failed fetch appends ``"sentiment"`` *and* logs at
  ``warning``.
* **A batch failure was a signal.**  ``/predict/batch`` answered a failed ticker
  with ``prediction: 0`` (SELL), ``probability_up: 0.0`` and the error text in
  the ``signal`` field, under HTTP 200 — a caller reading ``prediction`` traded
  on it.  It now raises ``HTTPException(502)``.

Every test here is offline: ``serve`` never downloads, no fetcher is reached
unpatched, and the prediction cache is redirected to ``tmp_path``.

``serve`` imports mlflow → opentelemetry, whose runtime-context loader still
calls the removed ``importlib.metadata`` dict interface on Python 3.11, and
FastAPI raises its own deprecation for ``@app.on_event("startup")``.  Both are
upstream of this repository, so the two filters below are scoped to their exact
text and to this module only, exactly as ``tests/test_prediction_cache.py`` does.
"""
import ast
import logging
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

# ``fastapi.testclient`` → ``starlette.testclient`` emits a collection-time
# ``StarletteDeprecationWarning`` when ``httpx2`` is absent ("Using `httpx` with
# `starlette.testclient` is deprecated; install `httpx2` instead").  A module
# ``pytestmark`` cannot catch it: the filter is installed when this module is
# *imported*, which is the very statement below.  So the import is wrapped —
# narrowly, and only around that one statement, which is the only place the
# warning is raised.  The message is captured and re-asserted so the wrapper
# cannot hide a *different* starlette warning, and every warning raised from a
# test body still escalates under ``filterwarnings = ["error"]``.
with warnings.catch_warnings(record=True) as _import_warnings:
    warnings.simplefilter("always")
    from fastapi.testclient import TestClient

_IMPORT_WARNINGS = [str(w.message) for w in _import_warnings]

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:SelectableGroups dict interface is deprecated:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:(?s).*on_event is deprecated.*:DeprecationWarning"
    ),
]

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN_SOURCE = _REPO_ROOT / "src" / "train.py"
BACKTEST_SOURCE = _REPO_ROOT / "src" / "backtest.py"

# The name the deleted block called.  It has never existed in scikit-learn and
# never will: the symbol the code wanted was ``CalibratedClassifierCV``.
NEVER_EXISTED = "CalibratedCVClassifier"

SCHEMA = ["rsi"]


# ═══════════════════════════════════════════════════════════════════════════════
# R1 — absence is asserted structurally, so prose is free
# ═══════════════════════════════════════════════════════════════════════════════

def _symbols_in_source(source: str) -> set[str]:
    """Every name a module's source binds or refers to, as text.

    Four responsibilities, because a name can be bound without ever appearing as
    a ``Name`` node:

    * a bare ``Name`` contributes its id;
    * an ``Attribute`` contributes the dotted chain it spells (``obj.attr``, and
      longer), so ``obj.CalibratedCVClassifier`` is visible;
    * ``ClassDef``/``FunctionDef``/``AsyncFunctionDef`` contribute their ``.name``
      — which is a plain ``str``, so a scanner that walks only ``Name`` and
      ``Attribute`` misses ``class CalibratedCVClassifier:`` entirely (this test's
      first draft did exactly that, and the both-directions test caught it);
    * ``ImportFrom``/``Import`` contribute each alias's ``.name`` and ``.asname``,
      so ``import x as y`` is visible under both spellings.

    Docstrings, comments and plain string literals contribute nothing, which is
    the whole point: deleting a block and explaining why it went must not be in
    tension.  The corollary is that ``getattr(mod, "CalibratedCVClassifier")`` is
    outside this check — the brief scopes it to names and attributes.
    """
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            parts, cursor = [], node
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name):
                parts.append(cursor.id)
            found.add(".".join(reversed(parts)))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            found.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                found.add(alias.name)
                if alias.asname:
                    found.add(alias.asname)
    return found


def _ast_symbols(path: pathlib.Path) -> set[str]:
    """``_symbols_in_source`` for a file, which is how the tests below use it."""
    return _symbols_in_source(path.read_text(encoding="utf-8"))


def _offenders(source: str) -> list[str]:
    """The assertion the whole of R1 rests on, as a function of source text.

    Kept in one place so the both-directions test below exercises *this* check
    and not a paraphrase of it.
    """
    return sorted(s for s in _symbols_in_source(source) if NEVER_EXISTED in s)


def test_the_dead_calibration_symbol_is_not_code():
    """R1: the misspelled symbol may be *named in prose*, but never executed.

    Fails if the symbol is genuinely reintroduced — as a bare name, as part of a
    dotted attribute, or as a class definition — and passes when a comment, a
    docstring or a string records that it never existed.
    """
    symbols = _ast_symbols(TRAIN_SOURCE)
    offenders = sorted(s for s in symbols if NEVER_EXISTED in s)
    assert not offenders, (
        f"{NEVER_EXISTED} is back as code in src/train.py: {offenders}"
    )


def test_the_absence_check_catches_reintroduction_and_allows_prose():
    """R1's both directions, asserted here rather than only demonstrated once.

    A structural check that never fails is worse than none: this pins that
    ``_offenders`` — the real check — fires for each way the symbol could come
    back, and stays quiet for the ways it must not.  The three failing forms are
    the three mutations a maintainer (or a merge) would plausibly reintroduce.
    """
    real_source = TRAIN_SOURCE.read_text(encoding="utf-8")

    # Direction 1 — prose is free.  The live module already carries a comment
    # naming the symbol, and a string mentioning it is likewise not code.
    assert NEVER_EXISTED in real_source
    assert _offenders(real_source) == []
    assert _offenders(f'MSG = "{NEVER_EXISTED} never existed"\n') == []

    # Direction 2 — each genuine reintroduction is caught.
    assert _offenders(f"final = {NEVER_EXISTED}(estimator=xgb)\n") != []
    assert _offenders(f"x = {NEVER_EXISTED}.fit\n") != []
    assert _offenders(f"class {NEVER_EXISTED}:\n    pass\n") != []
    assert _offenders(f"import sklearn.calibration as {NEVER_EXISTED}\n") != []
    # ...and the real symbol the block should have called is *not* flagged.
    assert _offenders("from sklearn.calibration import CalibratedClassifierCV\n") == []


def test_the_prose_naming_the_symbol_is_present_and_the_scan_ignores_it():
    """Both halves of R1 in one place: the tombstone exists and costs nothing.

    A raw text scan would fail on the comment this asserts is there, which is
    exactly the trap the structural assertion above avoids.
    """
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    assert NEVER_EXISTED in source, (
        "the tombstone recording why the calibration block was deleted is gone"
    )
    assert not [s for s in _ast_symbols(TRAIN_SOURCE) if NEVER_EXISTED in s]


def test_the_real_calibration_import_is_gone():
    """R2: ``CalibratedClassifierCV`` had no use once the block was removed."""
    tree = ast.parse(TRAIN_SOURCE.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "sklearn.calibration"
        for alias in node.names
    }
    assert "CalibratedClassifierCV" not in imported, imported
    assert "sklearn.calibration" not in TRAIN_SOURCE.read_text(encoding="utf-8")


def test_the_dead_scaler_import_is_gone_from_the_backtester():
    """R2's second import: ``StandardScaler`` was never referenced in backtest."""
    source = BACKTEST_SOURCE.read_text(encoding="utf-8")
    assert "StandardScaler" not in source, (
        "the unused StandardScaler import is still in src/backtest.py"
    )


def test_the_testclient_import_wrapper_hides_nothing_else():
    """The one wrapped import captured the one known message and nothing more.

    Pinned so that growing the wrapper into a general-purpose silencer would
    fail here rather than quietly reduce the suite's warning coverage.
    """
    assert all("starlette.testclient" in m for m in _IMPORT_WARNINGS), _IMPORT_WARNINGS
    assert len(_IMPORT_WARNINGS) <= 1, _IMPORT_WARNINGS


# ═══════════════════════════════════════════════════════════════════════════════
# The serving harness — offline, and one fake model for every path
# ═══════════════════════════════════════════════════════════════════════════════

class _StubModel:
    """The two-method surface every predicting path uses."""

    def predict(self, frame):
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[0.4, 0.6]])


def _stub_download(monkeypatch):
    """A 60-row price frame for the ticker being predicted, offline."""
    import serve

    frame = pd.DataFrame(
        {"Close": np.arange(60.0) + 100.0},
        index=pd.bdate_range("2024-01-01", periods=60),
    )
    monkeypatch.setattr(serve.yf, "download", lambda *a, **k: frame.copy())


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A ``TestClient`` over ``serve.app`` with the cache and model stubbed.

    ``_apply_sentiment_overlay`` is *not* stubbed here: the tests below patch
    ``features.fetchers.fetch_news_sentiment`` instead, so they drive the real
    overlay, the real ``degraded`` bookkeeping and the real ``warning``.

    The prediction cache is pre-populated for ``BBCA.JK``, so ``/predict`` takes
    its fast path.  That is deliberate: these tests are about what the response
    says, and the fast path is the one that exercises the overlay without a
    download.
    """
    import serve

    monkeypatch.setattr(serve, "PRED_CACHE_DB", str(tmp_path / "cache.db"))
    monkeypatch.setattr(serve, "load_best_model", lambda ticker: (_StubModel(), list(SCHEMA)))
    serve._cache_put("BBCA.JK", pd.Series({"rsi": 0.5}), SCHEMA, serve.CACHE_WINDOW_DEFAULT)
    serve.model_cache.clear()
    return TestClient(serve.app)


# ═══════════════════════════════════════════════════════════════════════════════
# `degraded` is real — both directions
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_sentiment_failure_is_a_200_naming_the_failed_component(
        client, monkeypatch, caplog):
    """A dead news API is a 200 with ``degraded == ["sentiment"]``, plus a warning.

    The prediction itself is intact — sentiment is an overlay, never a blocker —
    so the honest answer is "here is the signal, computed without sentiment".  A
    bare ``0.0`` said "the news was neutral", which is not what happened.

    The fetcher is replaced with one that *raises*, because that is the shape of
    the failure: ``features.fetchers.fetch_news_sentiment`` already swallows its
    own errors and returns ``0.0``, which is exactly why a dead NewsAPI key and
    a quiet news day were the same value in the response.
    """
    import features.fetchers as fetchers

    def no_news(ticker):
        raise RuntimeError("news api down")

    monkeypatch.setattr(fetchers, "fetch_news_sentiment", no_news)

    with caplog.at_level(logging.WARNING):
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["degraded"] == ["sentiment"], body
    assert body["sentiment_score"] == 0.0
    assert body["prediction"] in (0, 1)
    # Matched on the ASCII prefix only: this message contains a non-ASCII dash,
    # and ``caplog`` re-reads the module's source to render the record on a
    # cp1252 console, which mangles anything after that byte.  The prefix is
    # unambiguous and is the part an operator greps for.
    assert any(
        record.getMessage().startswith("Sentiment unavailable")
        and record.levelno >= logging.WARNING
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_a_healthy_sentiment_fetch_leaves_degraded_empty(client, monkeypatch):
    """The negative direction: no failure, no flag, no invented degradation."""
    import features.fetchers as fetchers

    monkeypatch.setattr(fetchers, "fetch_news_sentiment", lambda ticker: 0.25)

    response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    assert response.json()["degraded"] == [], response.json()


def test_a_healthy_sentiment_fetch_is_silent(client, monkeypatch, caplog):
    """A successful overlay logs no warning — ``degraded`` must stay meaningful."""
    import features.fetchers as fetchers

    monkeypatch.setattr(fetchers, "fetch_news_sentiment", lambda ticker: 0.0)

    with caplog.at_level(logging.WARNING):
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], \
        [r.getMessage() for r in caplog.records]


def test_a_downstream_feature_failure_is_never_a_signal(client, monkeypatch):
    """The brief's own test, minus the false lead.

    ``build_inference_features`` raising is a *feature* failure, not a sentiment
    one — so it is a 500 naming the cause, and it must not be reported as a
    sentiment degradation, which would send an operator to the wrong service.

    ``BBRI.JK`` rather than ``BBCA.JK``: the fixture seeds a cache row for the
    latter, and the fast path does not call the builder at all.
    """
    import serve

    def explode(*args, **kwargs):
        raise RuntimeError("news api down")

    monkeypatch.setattr(serve, "build_inference_features", explode)
    _stub_download(monkeypatch)
    monkeypatch.setattr(serve, "fetch_fundamentals", lambda ticker: {})

    response = client.post("/predict", json={"ticker": "BBRI.JK"})

    assert response.status_code == 500, response.text
    assert "news api down" in response.text
    assert "sentiment" not in response.text
    assert "prediction" not in response.text


# ═══════════════════════════════════════════════════════════════════════════════
# The batch endpoint — a failure is an error, not a SELL
# ═══════════════════════════════════════════════════════════════════════════════

def _stub_batch_io(monkeypatch):
    """Make the batch path's I/O local, synchronous and inert."""
    import serve

    async def inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    price_frame = pd.DataFrame(
        {"Close": np.arange(60.0) + 100.0},
        index=pd.bdate_range("2024-01-01", periods=60),
    )
    monkeypatch.setattr(serve.asyncio, "to_thread", inline)
    monkeypatch.setattr(serve.yf, "download", lambda *a, **k: price_frame.copy())
    monkeypatch.setattr(serve, "fetch_fundamentals", lambda ticker: {})


def test_a_failing_batch_ticker_is_502_not_a_sell(monkeypatch):
    """The shape the audit measured: ``prediction: 0``, confidence 0.0, HTTP 200.

    A caller cannot now mistake the failure for a SELL, because there is no
    prediction in the response at all.
    """
    import serve

    client = TestClient(serve.app, raise_server_exceptions=False)
    monkeypatch.setattr(serve, "load_best_model", lambda ticker: (_StubModel(), list(SCHEMA)))

    def explode(*args, **kwargs):
        raise ValueError("feature build failed for BROKEN.JK")

    _stub_batch_io(monkeypatch)
    monkeypatch.setattr(serve, "build_inference_features", explode)

    response = client.post("/predict/batch", json={"tickers": ["BROKEN.JK"]})

    assert response.status_code == 502, response.text
    body = response.json()
    assert "detail" in body, body
    assert "prediction" not in response.text
    assert "signals" not in response.text


def test_a_healthy_batch_ticker_still_returns_a_signal(tmp_path, monkeypatch):
    """The counterpart: the 502 is the failure path and nothing else changed."""
    import serve

    client = TestClient(serve.app)
    monkeypatch.setattr(serve, "PRED_CACHE_DB", str(tmp_path / "cache.db"))
    monkeypatch.setattr(serve, "load_best_model", lambda ticker: (_StubModel(), list(SCHEMA)))
    monkeypatch.setattr(serve, "build_inference_features",
                        lambda df, ticker, fundamentals: pd.DataFrame([{"rsi": 0.5}]))
    monkeypatch.setattr("features.fetchers.fetch_news_sentiment", lambda ticker: 0.25)
    _stub_batch_io(monkeypatch)

    response = client.post("/predict/batch", json={"tickers": ["BBCA.JK"]})

    assert response.status_code == 200, response.text
    predictions = response.json()["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["prediction"] == 1
    assert predictions[0]["degraded"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# R3 — an omitted feature family is named at warning; a deliberate None is not
# ═══════════════════════════════════════════════════════════════════════════════
# ``build_features`` distinguishes two absences that used to be the same value:
#
# * the argument was **omitted** (its default is a sentinel) — the caller forgot
#   the family, which is a degradation and is logged at ``warning``;
# * the argument is an explicit **None** — the caller declares the absence
#   deliberate (serving cannot obtain vp_*/idn_*/google_trend without per-request
#   network I/O), which is a design decision and is logged not at all.
#
# Both cases still omit the columns.  Warning on the second case is what made
# serving log two warnings on every healthy request, which trains operators to
# ignore warnings and so defeats the visibility this task exists to create.

def _omission_warnings(caplog, family: str) -> list[str]:
    """Records at WARNING or above naming ``family``, rendered as text."""
    return [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING and family in r.getMessage()
    ]


def test_an_omitted_feature_family_warns_once_and_still_omits(
        raw_ohlcv, ihsg_frame, fundamentals_dict, caplog):
    """One warning per forgotten family, and the family stays absent.

    The two arguments are *not passed at all* here, which is the only shape that
    now warns: a caller that leaves both families to their defaults has
    reproduced the fabricated-placeholder defect invisibly.
    """
    import features.pipeline as p

    with caplog.at_level(logging.WARNING):
        out = p.build_features(
            raw_ohlcv, "BBCA.JK",
            fundamentals=fundamentals_dict, usdidr=None, fred_macro=None,
            bi_rate=None, ihsg=ihsg_frame, news_sentiment=0.0,
            google_trend=-0.5,
            mode="full", with_target=True,
        )

    assert out is not None
    # Omitted, not fabricated: the volume profile family is simply absent.  The
    # ``idn_*`` names also arrive from ``inject_macro_features`` (the
    # ``fundamentals`` fixture keys are ``idn_der``/``idn_roe``/…), so the
    # omission here removes the family's *second* writer rather than the column
    # names; the warning is the observable change.
    assert not [c for c in out.columns if c.startswith("vp_")]

    vp = _omission_warnings(caplog, "volume_profile")
    idn = _omission_warnings(caplog, "idx_fundamentals")
    assert len(vp) == 1, vp
    assert len(idn) == 1, idn
    assert all(r.levelno == logging.WARNING for r in caplog.records
               if r.levelno >= logging.WARNING), caplog.records


def test_an_omitted_google_trend_warns_and_is_not_fabricated(
        raw_ohlcv, ihsg_frame, fundamentals_dict, caplog):
    """The third family, and the one whose default used to be a value.

    ``google_trend`` defaulted to ``0.0``, so "left at its default" was
    indistinguishable from "the caller passed 0.0" — a safe default
    indistinguishable from a real value, which is the defect.  With the sentinel
    the omission is detectable, it is named, and the column is not materialised.
    """
    import features.pipeline as p

    with caplog.at_level(logging.WARNING):
        out = p.build_features(
            raw_ohlcv, "BBCA.JK",
            fundamentals=fundamentals_dict, usdidr=None, fred_macro=None,
            bi_rate=None, ihsg=ihsg_frame, news_sentiment=0.0,
            vp_features={}, idx_fundamentals={},
            mode="full", with_target=True,
        )

    assert out is not None
    assert "google_trend" not in out.columns
    named = _omission_warnings(caplog, "google_trend")
    assert len(named) == 1, named


def test_a_deliberate_none_is_silent_and_still_omits(
        raw_ohlcv, ihsg_frame, fundamentals_dict, caplog):
    """The distinction itself: explicit ``None`` is a decision, not a fault.

    This is exactly what serving passes (``serve.build_inference_features``), so
    it is also the pin that the healthy serving path stays warning-free.
    """
    import features.pipeline as p

    with caplog.at_level(logging.WARNING):
        out = p.build_features(
            raw_ohlcv, "BBCA.JK",
            fundamentals=fundamentals_dict, usdidr=None, fred_macro=None,
            bi_rate=None, ihsg=ihsg_frame, news_sentiment=0.0,
            google_trend=None, vp_features=None, idx_fundamentals=None,
            mode="full", with_target=True,
        )

    assert out is not None
    assert not [c for c in out.columns if c.startswith("vp_")]
    assert "google_trend" not in out.columns
    for family in ("volume_profile", "idx_fundamentals", "google_trend"):
        assert _omission_warnings(caplog, family) == [], caplog.records


def test_a_supplied_family_is_not_reported_as_omitted(
        raw_ohlcv, ihsg_frame, fundamentals_dict, caplog):
    """The negative direction: a caller that supplies values gets no warning."""
    import features.pipeline as p

    with caplog.at_level(logging.WARNING):
        out = p.build_features(
            raw_ohlcv, "BBCA.JK",
            fundamentals=fundamentals_dict, usdidr=None, fred_macro=None,
            bi_rate=None, ihsg=ihsg_frame, news_sentiment=0.0,
            google_trend=-0.5, vp_features={"vp_poc": 1.0},
            idx_fundamentals={"idn_roe": 0.2},
            mode="full", with_target=True,
        )

    assert out is not None
    assert "vp_poc" in out.columns
    assert "idn_roe" in out.columns
    families = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not [m for m in families if "volume_profile" in m], families
    assert not [m for m in families if "idx_fundamentals" in m], families
    assert not [m for m in families if "google_trend" in m], families


def test_a_healthy_serving_request_logs_no_omission_warning(
        raw_ohlcv, ihsg_frame, fundamentals_dict, tmp_path, monkeypatch, caplog):
    """The whole point: the serving path is silent about its standing omissions.

    This drives the *real* ``serve.build_inference_features`` — the only serving
    caller of ``build_features`` — through ``POST /predict``, with the universe,
    macro caches and sentiment stubbed offline.  It is the request shape an
    operator sees on a healthy day, and it must produce neither an omission
    warning nor a ``degraded`` entry.
    """
    import serve
    import features.fetchers as fetchers
    from datetime import date as _date

    monkeypatch.setattr(serve, "PRED_CACHE_DB", str(tmp_path / "cache.db"))
    # ``SCHEMA``'s ``rsi`` is a *cached-row* column name that no computed frame
    # actually carries (``ta`` names it ``momentum_rsi``); this test runs the
    # real builder, so the contract it aligns to is one the builder really
    # produces.
    monkeypatch.setattr(serve, "load_best_model",
                        lambda ticker: (_StubModel(), ["ret_1d"]))
    serve.model_cache.clear()

    # The slow path is the one that calls the builder, so the download, the
    # fundamentals fetch and the daily macro refresh must all be local.  Pinning
    # ``_cache_date`` to today makes the background refresh return immediately
    # rather than reach for the network after the response.
    monkeypatch.setattr(serve, "_cache_date", str(_date.today()))
    monkeypatch.setattr(serve, "fetch_fundamentals", lambda ticker: {})
    monkeypatch.setattr(serve.yf, "download",
                        lambda *a, **k: raw_ohlcv["BBCA.JK"].dropna().copy())
    monkeypatch.setattr(fetchers, "fetch_news_sentiment", lambda ticker: 0.25)

    all_ticker_data = {t: raw_ohlcv[t].dropna() for t in ("BBCA.JK", "BBRI.JK", "BMRI.JK")}
    monkeypatch.setattr(serve, "load_universe_context",
                        lambda: (raw_ohlcv, ihsg_frame, all_ticker_data))
    index = raw_ohlcv.index
    monkeypatch.setattr(serve, "_usdidr_cache", pd.DataFrame(
        {"usdidr_rate": np.linspace(15_000.0, 15_600.0, len(index)),
         "usdidr_return": np.full(len(index), 0.0004)}, index=index))
    monkeypatch.setattr(serve, "_fred_cache",
                        pd.DataFrame({"fed_rate": np.full(len(index), 5.25)}, index=index))
    monkeypatch.setattr(serve, "_bi_cache",
                        pd.Series(6.0, index=index, name="bi_rate_official"))

    client = TestClient(serve.app)
    with caplog.at_level(logging.WARNING):
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    # Healthy path: nothing failed, so nothing is reported as degraded.
    assert response.json()["degraded"] == [], response.json()

    noisy = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert [m for m in noisy if "not supplied" in m] == [], noisy
    for family in ("volume_profile", "idx_fundamentals", "google_trend"):
        assert _omission_warnings(caplog, family) == [], noisy


# ═══════════════════════════════════════════════════════════════════════════════
# R4 — a volume-profile download failure is visible, and still returns {}
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_volume_profile_download_failure_logs_at_warning(monkeypatch, caplog):
    """R4: the failure was ``logger.debug``, so 19 features vanished silently."""
    import features.volume_profile as vp

    def explode(*args, **kwargs):
        raise RuntimeError("intraday download refused")

    monkeypatch.setattr("ingest.download_intraday_data", explode, raising=False)

    with caplog.at_level(logging.WARNING):
        result = vp.compute_volume_profile_features("BBCA.JK")

    assert result == {}, result
    assert any(
        "BBCA.JK" in record.getMessage() and record.levelno >= logging.WARNING
        for record in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_the_volume_profile_omission_is_named_by_the_training_wrapper(
        raw_ohlcv, fundamentals_dict, monkeypatch, caplog):
    """End to end: the empty family reaches ``build_features`` and is named.

    ``engineer_features_for_ticker`` catches the download failure itself, so the
    omission has to be named at *both* layers: the fetcher for the cause, the
    builder for the consequence.
    """
    import features.pipeline as p

    def explode(ticker):
        raise RuntimeError("intraday download refused")

    # ``engineer_features_for_ticker`` binds the name at import, so the pipeline's
    # own reference is what the wrapper calls.
    monkeypatch.setattr(p, "compute_volume_profile_features", explode)

    with caplog.at_level(logging.WARNING):
        out = p.engineer_features_for_ticker(
            raw_ohlcv, "BBCA.JK", fundamentals_dict,
            usdidr=None, fred_macro=None, bi_rate=None, ihsg=None, mode="full",
        )

    assert out is not None
    assert not [c for c in out.columns if c.startswith("vp_")]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("volume profile" in m.lower() for m in warnings), warnings
