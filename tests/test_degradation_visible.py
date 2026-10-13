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
  overlay caught an exception out of the fetcher, but the fetcher catches every
  exception itself and returns ``0.0``, so ``predict`` presented a dead NewsAPI
  endpoint, a missing key, a 429 and an absent ``vaderSentiment`` as a clean
  neutral reading.  ``PredictResponse`` now carries ``degraded``, and the overlay
  reads ``features.fetchers.fetch_news_sentiment_status`` — the seam that says
  whether the score is a *measurement* — so an unavailable source appends
  ``"sentiment"`` *and* logs at ``warning`` with the reason.  The tests below
  drive that fetcher for real (see ``news_sentiment_state``); none of them
  substitutes a fetcher that *raises*, because that is a shape the production
  path never produces.
* **A batch failure was a signal.**  ``/predict/batch`` answered a failed ticker
  with ``prediction: 0`` (SELL), ``probability_up: 0.0`` and the error text in
  the ``signal`` field, under HTTP 200 — a caller reading ``prediction`` traded
  on it.  It now raises ``HTTPException(502)``.

Every test here is offline: ``serve`` never downloads, the prediction cache is
redirected to ``tmp_path``, and the only collaborators reached are the ones the
tests stand in for deliberately — the model, the price download, and the two
third-party halves of the news path (``requests`` and the VADER analyser).  The
news *seam* itself is never stubbed: ``news_sentiment`` runs for real, because a
stub is what hid this defect in the first place.

``serve`` imports mlflow → opentelemetry, whose runtime-context loader still
calls the removed ``importlib.metadata`` dict interface on Python 3.11, and
FastAPI raises its own deprecation for ``@app.on_event("startup")``.  Both are
upstream of this repository, so the two filters below are scoped to their exact
text and to this module only, exactly as ``tests/test_prediction_cache.py`` does.
"""
import ast
import logging
import pathlib
import sys
import types
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

#: The exact message the wrapped import is allowed to capture.  Pinned in full
#: rather than by substring: a second, different starlette deprecation must fail
#: the pin below instead of being silently swallowed along with this one.
EXPECTED_IMPORT_WARNING = (
    "Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` "
    "instead."
)

# ``src/news_sentiment.py`` calls ``warnings.filterwarnings("ignore")`` at import
# time, which would otherwise disable the suite's ``filterwarnings = ["error"]``
# for every test collected after it.  ``catch_warnings`` saves and restores the
# filter list, so the module's global ignore is discarded when this block exits.
# The import is at module scope on purpose: it emits one ``warning`` naming the
# absent ``vaderSentiment``, and that record belongs to collection rather than to
# whichever test happens to be running when the seam is first touched.
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


def _offenders(source: str, symbol: str = NEVER_EXISTED) -> list[str]:
    """The assertion the whole of R1 rests on, as a function of source text.

    Kept in one place so the both-directions test below exercises *this* check
    and not a paraphrase of it.  ``symbol`` is a parameter because the same
    structural check covers R2's dead imports (``StandardScaler``,
    ``sklearn.calibration``).
    """
    return sorted(s for s in _symbols_in_source(source) if symbol in s)


#: The module the deleted calibration block imported.  Its absence is a separate
#: check from the symbol's: ``from sklearn.calibration import X`` puts the module
#: in an ``ImportFrom.module``, which is a plain ``str`` the symbol walk cannot
#: see, while ``import sklearn.calibration`` and ``sklearn.calibration.X`` are
#: aliases and attribute chains that it can.
CALIBRATION_MODULE = "sklearn.calibration"


def _calibration_module_offenders(source: str) -> list[str]:
    """Names in ``source`` that reach for the ``sklearn.calibration`` module.

    Deliberately structural: a tombstone comment that names the module while
    explaining what was deleted contributes nothing here, which is the whole
    reason this check replaced a raw ``not in source`` scan.
    """
    found = [
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith(CALIBRATION_MODULE)
    ]
    found += [s for s in _symbols_in_source(source) if s.startswith(CALIBRATION_MODULE)]
    return sorted(set(found))


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
    """R2: ``CalibratedClassifierCV`` had no use once the block was removed.

    Asserted structurally, both ways the module can be reached.  This test used
    to end in ``assert "sklearn.calibration" not in source`` — a raw text scan,
    which is the landmine the AST check exists to remove: the tombstone comment
    recording *why* the block was deleted names the module, and a text scan fails
    on that prose.
    """
    tree = ast.parse(TRAIN_SOURCE.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "sklearn.calibration"
        for alias in node.names
    }
    assert "CalibratedClassifierCV" not in imported, imported
    assert _calibration_module_offenders(TRAIN_SOURCE.read_text(encoding="utf-8")) == []


def test_the_dead_scaler_import_is_gone_from_the_backtester():
    """R2's second import: ``StandardScaler`` was never referenced in backtest.

    A structural check for the same reason as the calibration one above: a
    comment recording that the import was removed must not fail its own test.
    """
    offenders = _offenders(BACKTEST_SOURCE.read_text(encoding="utf-8"), "StandardScaler")
    assert not offenders, (
        f"the unused StandardScaler import is still in src/backtest.py: {offenders}"
    )


def test_the_ast_absence_checks_still_catch_a_real_reintroduction():
    """Both replaced checks, shown to fail if the symbol genuinely comes back.

    Replacing a text scan with a structural check is only an improvement if the
    structural check still catches *code*.  Each case below is a shape a
    maintainer or a merge would plausibly restore, followed by the prose shapes
    that must stay free.
    """
    # ``StandardScaler`` — every way a name can be bound or referenced.
    assert _offenders("import StandardScaler\n", "StandardScaler") != []
    assert _offenders("from sklearn.preprocessing import StandardScaler\n",
                      "StandardScaler") != []
    assert _offenders("scaler = StandardScaler()\n", "StandardScaler") != []
    assert _offenders("def StandardScaler():\n    pass\n", "StandardScaler") != []
    assert _offenders("# StandardScaler was removed as unused\n",
                      "StandardScaler") == []
    assert _offenders('DOC = "StandardScaler"\n', "StandardScaler") == []

    # ``sklearn.calibration`` — as an import, as a dotted module, as an attribute.
    assert _calibration_module_offenders(
        "from sklearn.calibration import CalibratedClassifierCV\n") != []
    assert _calibration_module_offenders("import sklearn.calibration\n") != []
    assert _calibration_module_offenders("sklearn.calibration.CalibratedClassifierCV()\n") != []
    assert _calibration_module_offenders("# nothing from sklearn.calibration here\n") == []
    assert _calibration_module_offenders('DOC = "sklearn.calibration"\n') == []


def test_the_testclient_import_wrapper_hides_nothing_else():
    """The one wrapped import captured the one known message and nothing more.

    Pinned to the *exact* message, not a substring: a second, unrelated starlette
    deprecation would otherwise satisfy a substring check and be swallowed
    silently.  Absence is tolerated — if a future starlette stops emitting this
    deprecation the wrapper simply has nothing to catch — but any other message,
    or more than one of them, fails here.
    """
    assert set(_IMPORT_WARNINGS) <= {EXPECTED_IMPORT_WARNING}, _IMPORT_WARNINGS
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


# ── Driving the real news seam ────────────────────────────────────────────────
# The point of this task is that a *production* sentiment failure must be
# visible.  ``features.fetchers.fetch_news_sentiment`` returns the same ``0.0``
# for every one of them and never raises, so a test that stubs it with something
# that raises proves nothing about the seam production actually uses — that was
# the flaw in the first version of these tests.  Everything below therefore runs
# the real ``news_sentiment`` module.  Only the two third-party collaborators it
# talks to are stood in for, and each stand-in carries exactly the surface the
# module uses.

class _FakeAnalyzer:
    """``vaderSentiment``'s analyzer, as ``analyze_sentiment`` uses it.

    One method, ``polarity_scores(text) -> {"compound": float}``, and a fixed
    compound so the expected overlay arithmetic is exact.  ``VADER_AVAILABLE`` is
    forced true alongside it — the state a genuine install produces — so the
    module's real lazy-load path in ``get_sentiment_analyzer`` runs.
    """

    COMPOUND = 0.4

    def polarity_scores(self, text):
        return {"compound": self.COMPOUND}


class _NeutralAnalyzer(_FakeAnalyzer):
    """The same analyzer, reading every headline as genuinely neutral."""

    COMPOUND = 0.0


class _FakeResponse:
    """``requests.Response``, as ``fetch_news`` uses it: status and ``json()``."""

    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _articles(*titles):
    """A NewsAPI success body carrying one article per title."""
    return {"articles": [{"title": title} for title in titles]}


def _fake_newsapi(monkeypatch, *responses):
    """Point ``news_sentiment``'s HTTP client at canned responses.

    ``fetch_news`` uses exactly one attribute of ``requests`` — ``get`` — so the
    stand-in is that attribute and nothing else.  Responses are consumed in call
    order and the last repeats, because the function tries up to three search
    strategies; one response therefore covers every strategy, which is what the
    "endpoint answered with no articles" case needs.
    """
    queue = list(responses)
    calls = []

    def get(url, params=None, timeout=None):
        calls.append(params)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(news_sentiment, "requests", types.SimpleNamespace(get=get))
    return calls


def _vader_present(monkeypatch, analyzer=_FakeAnalyzer):
    """Put the module in the state a real ``vaderSentiment`` install produces."""
    monkeypatch.setattr(news_sentiment, "VADER_AVAILABLE", True)
    monkeypatch.setattr(
        news_sentiment, "SentimentIntensityAnalyzer", analyzer, raising=False
    )
    monkeypatch.setattr(news_sentiment, "_sentiment_analyzer", None)


@pytest.fixture
def news_sentiment_state(tmp_path, monkeypatch):
    """The real module, with its on-disk cache and its optional dependency isolated.

    ``CACHE_DIR`` is redirected because the real path is a repository cache the
    module writes on every call (``data/sentiment/<ticker>.json``), and because an
    entry already sitting there would answer before the seam under test was
    reached — the repo's own cache is full of ``n_articles: 0`` entries that a
    pre-fix run persisted as neutral scores.
    """
    monkeypatch.setattr(news_sentiment, "CACHE_DIR", str(tmp_path / "sentiment"))
    monkeypatch.setattr(news_sentiment, "_sentiment_analyzer", None)
    return news_sentiment


# The production failure modes, each as "the state the module is left in".  None
# of them raises anywhere: that is exactly why the overlay's ``except`` was dead
# code, and why the reason has to travel as data.
def _mode_no_key(monkeypatch):
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "")
    _vader_present(monkeypatch)


def _mode_dead_endpoint(monkeypatch):
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "test-key")
    _vader_present(monkeypatch)

    def refuse(url, params=None, timeout=None):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(news_sentiment, "requests", types.SimpleNamespace(get=refuse))


def _mode_rejected_key(monkeypatch):
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "test-key")
    _vader_present(monkeypatch)
    _fake_newsapi(monkeypatch, _FakeResponse(401, {"status": "error"}))


def _mode_rate_limited(monkeypatch):
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "test-key")
    _vader_present(monkeypatch)
    _fake_newsapi(monkeypatch, _FakeResponse(429))


def _mode_no_vader(monkeypatch):
    """The plain-venv case: the key and the endpoint are fine, VADER is absent."""
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "test-key")
    monkeypatch.setattr(news_sentiment, "VADER_AVAILABLE", False)
    monkeypatch.setattr(news_sentiment, "_sentiment_analyzer", None)
    _fake_newsapi(monkeypatch, _FakeResponse(200, _articles("Bank BCA profit rises")))


def _mode_no_articles(monkeypatch):
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "test-key")
    _vader_present(monkeypatch)
    _fake_newsapi(monkeypatch, _FakeResponse(200, {"articles": []}))


#: The reason each failure mode must be named with.  Written as literals rather
#: than read from the module so that this table is a *behavioural* pin: if a
#: reason string is renamed, the overlay's log line stops matching these and the
#: mode test fails.  ``test_the_reason_constants_are_what_the_tests_expect`` ties
#: the literals back to the module's own constants.
NO_NEWSAPI_KEY_REASON = "NEWSAPI_KEY is not set"
NO_VADER_REASON = "vaderSentiment is not installed"
RATE_LIMITED_REASON = "NewsAPI rate limit reached (HTTP 429)"

FAILURE_MODES = [
    pytest.param(_mode_no_key, NO_NEWSAPI_KEY_REASON, id="no-newsapi-key"),
    pytest.param(_mode_dead_endpoint, "ConnectionError", id="dead-endpoint"),
    pytest.param(_mode_rejected_key, "NewsAPI returned HTTP 401", id="rejected-key"),
    pytest.param(_mode_rate_limited, RATE_LIMITED_REASON, id="rate-limited"),
    pytest.param(_mode_no_vader, NO_VADER_REASON, id="no-vader"),
    pytest.param(_mode_no_articles, "returned no articles", id="no-articles"),
]


def test_the_reason_constants_are_what_the_tests_expect():
    """The mode table above is tied to the module's own reason strings."""
    assert news_sentiment.NO_NEWSAPI_KEY == NO_NEWSAPI_KEY_REASON
    assert news_sentiment.NO_VADER == NO_VADER_REASON
    assert news_sentiment.NEWSAPI_RATE_LIMITED == RATE_LIMITED_REASON


@pytest.fixture
def healthy_news(news_sentiment_state, monkeypatch):
    """The real seam in the state a healthy production fetch produces."""
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "test-key")
    _vader_present(monkeypatch)
    _fake_newsapi(monkeypatch, _FakeResponse(200, _articles("Bank BCA profit rises")))
    return news_sentiment


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A ``TestClient`` over ``serve.app`` with the cache and model stubbed.

    ``_apply_sentiment_overlay`` is *not* stubbed here, and neither is the
    fetcher it reads: the tests below drive the real
    ``features.fetchers.fetch_news_sentiment_status`` against the real
    ``news_sentiment`` module, so they exercise the seam production uses — not a
    stub whose shape production never produces.

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

@pytest.mark.parametrize("prepare, expected_reason", FAILURE_MODES)
def test_every_real_sentiment_failure_mode_is_degraded_and_named(
        client, news_sentiment_state, monkeypatch, caplog, prepare, expected_reason):
    """Each production failure mode is ``degraded == ["sentiment"]`` plus a warning.

    The prediction itself stays intact — sentiment is an overlay, never a blocker
    — so the honest answer is "here is the signal, computed without sentiment".
    What must never happen is the old one: a bare ``0.0`` saying "the news was
    neutral", which is not what happened.

    The four modes differ only in the state they leave the *real* module in (no
    key, an unreachable or rejected endpoint, a 429, no VADER).  Nothing here
    replaces the seam with something that raises, because the seam does not
    raise: all of these used to return ``0.0`` from it, which is how the
    overlay's ``except`` came to be dead code.
    """
    prepare(monkeypatch)

    with caplog.at_level(logging.WARNING):
        # Cleared after the level is raised but before the request: the module's
        # own import-time "vaderSentiment not installed" notice is emitted at
        # collection, and this assertion is about what the request logged.
        caplog.clear()
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["degraded"] == ["sentiment"], body
    assert body["prediction"] in (0, 1)
    assert body["sentiment_score"] == 0.0, body

    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "Sentiment unavailable" in line and expected_reason in line for line in lines
    ), lines


@pytest.mark.parametrize("prepare, expected_reason", FAILURE_MODES)
def test_the_value_only_fetcher_still_returns_a_float(
        news_sentiment_state, monkeypatch, prepare, expected_reason):
    """Existing callers keep working: the failure is still a float, not an exception.

    ``build_features``'s ``news_sentiment`` model input needs a float (a neutral
    ``0.0`` for "unavailable" is acceptable and consistent with training), and
    ``cli.py sentiment`` must not crash — so the value-only fetcher keeps
    returning ``0.0`` for every failure mode.  The status companion is what makes
    that ``0.0`` visible, and the second half of this test pins that the
    unavailability survives the module's cache: the reason read back is the
    reason that was written, not a fresh guess.
    """
    import features.fetchers as fetchers

    prepare(monkeypatch)

    score = fetchers.fetch_news_sentiment("BBCA.JK")
    assert isinstance(score, float) and score == 0.0

    _, available, reason = fetchers.fetch_news_sentiment_status("BBCA.JK")
    assert available is False
    assert expected_reason in reason, reason


def test_a_plain_venv_degradation_is_visible(
        client, news_sentiment_state, monkeypatch, caplog):
    """The state ``AGENTS.md`` documents: no ``NEWSAPI_KEY`` and no ``vaderSentiment``.

    ``pip install -r requirements.txt`` installs neither, so this is the default
    developer environment, and the documented guarantee is that the degradation
    there is visible as ``degraded: ["sentiment"]`` *and* a ``warning`` — not only
    in the score.  That sentence was false while the failure was silent; it is
    asserted here so it cannot quietly become false again.

    The reason named is the missing key rather than the missing analyser: the key
    is checked first, because a request that never leaves the machine cannot be
    blamed on VADER.  With a key set and no VADER, the ``no-vader`` mode above
    names VADER instead — the two are separable, which is the point.
    """
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "")
    monkeypatch.setattr(news_sentiment, "VADER_AVAILABLE", False)
    monkeypatch.setattr(news_sentiment, "_sentiment_analyzer", None)

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["degraded"] == ["sentiment"], body
    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "Sentiment unavailable" in line and NO_NEWSAPI_KEY_REASON in line
        for line in lines
    ), lines


def test_a_real_sentiment_score_still_flows_through_unchanged(client, healthy_news):
    """The success path is untouched: the score reaches the response, and the
    overlay arithmetic is exactly what it was.

    ``fetch_news_sentiment_status`` returns ``available=True`` here, so the score
    is used as a score — the change is additive on the failure path only.
    """
    response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["degraded"] == [], body
    assert body["sentiment_score"] == _FakeAnalyzer.COMPOUND
    assert body["probability_adjusted"] == pytest.approx(
        round(0.6 + _FakeAnalyzer.COMPOUND * 0.10, 4)
    )
    assert body["signal_adjusted"] == "BUY"


def test_a_genuine_neutral_reading_is_not_degraded(
        client, news_sentiment_state, monkeypatch, caplog):
    """A successful analysis that happens to score 0.0 is not a degradation.

    This is the other half of the fix, and the reason the mechanism is a status
    rather than "treat 0.0 as failure": a quiet news day and a dead news API both
    produce ``sentiment_score == 0.0``, and now they are distinguishable — one
    carries ``degraded == []`` and logs nothing, the other carries
    ``degraded == ["sentiment"]`` and logs the reason.
    """
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "test-key")
    _vader_present(monkeypatch, _NeutralAnalyzer)
    _fake_newsapi(monkeypatch, _FakeResponse(200, _articles("Bank BCA holds steady")))

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        response = client.post("/predict", json={"ticker": "BBCA.JK"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["degraded"] == [], body
    assert body["sentiment_score"] == 0.0, body
    noisy = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert noisy == [], noisy


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


def test_a_healthy_batch_ticker_still_returns_a_signal(
        tmp_path, monkeypatch, healthy_news):
    """The counterpart: the 502 is the failure path and nothing else changed.

    ``healthy_news`` drives the real fetcher to a success, so the ``degraded``
    assertion below is about the batch path staying clean rather than about a
    stub that was told to look clean.
    """
    import serve

    client = TestClient(serve.app)
    monkeypatch.setattr(serve, "PRED_CACHE_DB", str(tmp_path / "cache.db"))
    monkeypatch.setattr(serve, "load_best_model", lambda ticker: (_StubModel(), list(SCHEMA)))
    monkeypatch.setattr(serve, "build_inference_features",
                        lambda df, ticker, fundamentals: pd.DataFrame([{"rsi": 0.5}]))
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


def test_a_supplied_but_empty_family_is_silent(
        raw_ohlcv, ihsg_frame, fundamentals_dict, caplog):
    """A supplied-but-*empty* dict is supplied: it is not an omission.

    ``vp_features={}`` and ``idx_fundamentals={}`` are exactly what the training
    wrapper passes when its fetch failed — ``compute_volume_profile_features``
    returns ``{}`` on every failure path — so this is a real production shape, and
    it was the unpinned one.  The columns stay absent, as they must (never
    fabricated), and nothing is reported as *omitted*: the omission warning names a
    caller who forgot a family, and this caller did not.  The failure that produced
    the empty dict is named where it happened, by
    ``test_a_volume_profile_download_failure_logs_at_warning``.
    """
    import features.pipeline as p

    with caplog.at_level(logging.WARNING):
        out = p.build_features(
            raw_ohlcv, "BBCA.JK",
            fundamentals=fundamentals_dict, usdidr=None, fred_macro=None,
            bi_rate=None, ihsg=ihsg_frame, news_sentiment=0.0,
            google_trend=-0.5, vp_features={}, idx_fundamentals={},
            mode="full", with_target=True,
        )

    assert out is not None
    assert not [c for c in out.columns if c.startswith("vp_")]
    assert _omission_warnings(caplog, "volume_profile") == [], caplog.records
    assert _omission_warnings(caplog, "idx_fundamentals") == [], caplog.records


def test_a_healthy_serving_request_logs_no_omission_warning(
        raw_ohlcv, ihsg_frame, fundamentals_dict, tmp_path, monkeypatch, caplog,
        healthy_news):
    """The whole point: the serving path is silent about its standing omissions.

    This drives the *real* ``serve.build_inference_features`` — the only serving
    caller of ``build_features`` — through ``POST /predict``, with the universe
    and the macro caches stubbed offline.  It is the request shape an operator
    sees on a healthy day, and it must produce neither an omission warning nor a
    ``degraded`` entry.

    Sentiment is a *real* fetch here too (``healthy_news``), not a stub: a
    request only counts as healthy if the component the previous round of this
    task was about is genuinely reporting itself healthy.
    """
    import serve
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
        caplog.clear()
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
# ``compute_volume_profile_features`` gained four ``warning`` lines in this task
# and only the first of them was covered.  A visibility constraint whose warnings
# are untested is a constraint that can be deleted without anyone noticing, so
# each branch gets a driver here — and the success path gets a negative control,
# which is what keeps the other four meaningful.

def _vp_warnings(caplog, cause: str) -> list[str]:
    """Records at WARNING or above that mention ``cause``."""
    return [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING and cause in r.getMessage()
    ]


def _stub_intraday(monkeypatch, frame):
    """Make ``volume_profile``'s download return ``frame`` (or raise, if given)."""
    monkeypatch.setattr("ingest.download_intraday_data", frame, raising=False)


def _intraday_frame(rows: int, start: str = "2024-01-02 09:00") -> pd.DataFrame:
    """A plausible 5-minute intraday frame with unique minute-resolution rows."""
    index = pd.date_range(start, periods=rows, freq="5min", name="Datetime")
    price = np.linspace(100.0, 101.0, rows)
    return pd.DataFrame(
        {
            "Open": price,
            "High": price + 0.2,
            "Low": price - 0.2,
            "Close": price,
            "Volume": np.linspace(1_000.0, 2_000.0, rows),
        },
        index=index,
    )


def test_a_volume_profile_download_failure_logs_at_warning(monkeypatch, caplog):
    """R4: the failure was ``logger.debug``, so 19 features vanished silently."""
    import features.volume_profile as vp

    def explode(*args, **kwargs):
        raise RuntimeError("intraday download refused")

    _stub_intraday(monkeypatch, explode)

    with caplog.at_level(logging.WARNING):
        result = vp.compute_volume_profile_features("BBCA.JK")

    assert result == {}, result
    assert any(
        "BBCA.JK" in record.getMessage() and record.levelno >= logging.WARNING
        for record in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_a_volume_profile_import_failure_logs_at_warning(monkeypatch, caplog):
    """R4's first branch: ``ingest`` is not importable, so 19 columns go missing.

    ``sys.modules[...] = None`` is how CPython is told an import is refused: the
    ``from ingest import download_intraday_data`` inside the function then raises
    a real ``ImportError``, which is the shape the handler exists for.
    """
    import features.volume_profile as vp

    monkeypatch.setitem(sys.modules, "ingest", None)

    with caplog.at_level(logging.WARNING):
        result = vp.compute_volume_profile_features("BBCA.JK")

    assert result == {}, result
    named = _vp_warnings(caplog, "not importable")
    assert named, [r.getMessage() for r in caplog.records]


def test_a_volume_profile_empty_download_logs_at_warning(monkeypatch, caplog):
    """R4's second branch: the download succeeded and returned nothing usable.

    Both halves are driven — an empty frame and a frame below the ten-row floor —
    because they take the same branch for the same reason and a regression in
    either direction should be visible.
    """
    import features.volume_profile as vp

    for label, frame in (("empty", pd.DataFrame()),
                         ("too short", _intraday_frame(4))):
        caplog.clear()
        _stub_intraday(monkeypatch, lambda *a, **k: frame)
        with caplog.at_level(logging.WARNING):
            result = vp.compute_volume_profile_features("BBCA.JK")
        assert result == {}, (label, result)
        named = _vp_warnings(caplog, "empty or too short")
        assert named, (label, [r.getMessage() for r in caplog.records])


def test_a_volume_profile_short_latest_session_logs_at_warning(monkeypatch, caplog):
    """R4's third branch: enough rows overall, too few in the latest session.

    One session carries 12 rows and the next only 3, which is what a partial
    trading day looks like — the frame passes the earlier length gate and the
    features would otherwise be computed from three points.
    """
    import features.volume_profile as vp

    frame = pd.concat([
        _intraday_frame(12, "2024-01-02 09:00"),
        _intraday_frame(3, "2024-01-03 09:00"),
    ])
    _stub_intraday(monkeypatch, lambda *a, **k: frame)

    with caplog.at_level(logging.WARNING):
        result = vp.compute_volume_profile_features("BBCA.JK")

    assert result == {}, result
    named = _vp_warnings(caplog, "latest session")
    assert named, [r.getMessage() for r in caplog.records]


def test_a_successful_volume_profile_download_warns_about_nothing(monkeypatch, caplog):
    """The negative control: the four warnings above are failure-specific.

    Without this, "the failure is visible" would be satisfied by a function that
    warns unconditionally — which is the noise that teaches operators to ignore
    the channel.
    """
    import features.volume_profile as vp

    _stub_intraday(monkeypatch, lambda *a, **k: _intraday_frame(60))

    with caplog.at_level(logging.WARNING):
        result = vp.compute_volume_profile_features("BBCA.JK")

    assert result, result
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


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
