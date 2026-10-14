"""Task 2.1 — the integration gate: the real API, over HTTP, on a real model.

Everything else in the rescue tests one layer with its neighbours replaced.  This
file is the other end of the range: ``TestClient(serve.app)`` — the real
FastAPI app, the real ``model_store`` resolution, the real 60-column persisted
contract, the real ``build_inference_features`` the training path shares, the
real retrained XGBoost model, the real ``PredictResponse`` — driven through
``POST /predict`` and ``POST /predict/unknown`` exactly as a caller would.

Offline, and by construction rather than by luck
------------------------------------------------

The brief's version of ``test_predict_contract`` posts to ``/predict`` with
nothing patched.  Measured in this session with the process's egress pointed at a
closed port, that is not a test of the app: the request reaches Yahoo Finance
through ``yf.download``, comes back empty, and the app answers
``400 {"detail": "No data on Yahoo Finance for BBCA.JK"}`` after ~20 s of
retries.  A gate that only passes when the machine has internet is not a gate, so
this file fakes **the network boundary alone** and runs everything behind it for
real:

* ``yf.download`` serves the ticker's own rows out of the committed
  ``data/raw/stocks.csv`` (the universe the repository ships), in the exact
  layout yfinance 1.4.1 returns for a single symbol — a ``(Price, Ticker)``
  two-level column ``MultiIndex``, which is why ``serve``'s
  ``droplevel(1)`` is exercised rather than bypassed.  The window is the last
  250 *calendar* days of the committed rows, anchored to the file's own last row
  rather than to ``date.today()`` so the served row is byte-identical on any day
  the suite runs.
* A symbol the committed file cannot answer — ``IDR=X`` for the daily macro
  fetch, or ``yf.Ticker(...).info`` for fundamentals — is **refused and
  recorded**, never fabricated.  ``requests.get``/``post`` are refused the same
  way, which is what the BI-rate scrape and (with a key set) NewsAPI reach.
* ``features.fetchers.fetch_fred_macro`` is a fifth seam, and the one whose
  absence used to make this file's "offline" claim untrue whenever
  ``FRED_API_KEY`` was exported: ``refresh_daily_cache`` really runs here (the
  fixture sets ``_cache_date = None``) and reaches FRED through ``fredapi``,
  which leaves by ``urllib``.  Neither the ``requests`` patches nor a patch of
  ``urllib.request.urlopen`` can close that route — ``fredapi`` binds
  ``urlopen`` into its own module namespace at import
  (``fredapi/fred.py:16``) — so the seam is installed where the call resolves.
  It records the attempt and answers with an *empty frame*, which is the value
  the real function returns for every failure mode it swallows internally, so the
  daily refresh still completes; refusing it changes no served number, because no
  FRED series is in the 60-name contract and the one macro-derived column in it,
  ``usdidr_stress``, comes from the USD/IDR frame rather than from FRED.
* ``TestClient(serve.app)`` is constructed directly, never used as a context
  manager: Starlette runs startup hooks only from ``__enter__``, and
  ``startup_load_index`` launches ``_warmup_model_cache`` and
  ``_warmup_prediction_cache``, which between them download five tickers from
  Yahoo.  Skipping them is *why* this file can stay offline, and
  ``test_no_startup_warmup_touched_the_network`` pins that they did not run —
  not by reading a flag, but by asserting no download was recorded for the four
  warmup tickers this file never asks about.  That test is a tripwire on this
  fixture rather than a behavioural proof: nothing reachable from this file can
  start the hooks (see its docstring).  Nothing is lost by skipping
  startup: the request path resolves a ticker through ``model_store``, which
  reads ``model_index.json`` off disk on each uncached request and never consults
  the ``_model_index`` startup fills in.

The two cache paths, pinned at the API level
--------------------------------------------

F7 measured the same ticker on the same day answering ``0.187`` on one path and
``0.303`` on the other, because the hit path padded a missing column with ``0.0``
where the miss path used ``NaN``.  Three tests close that at the HTTP layer: a
cold request is a **miss** and a second one is a **hit** (proved by counting the
real ``_cache_put`` calls, not by log text); the bodies must be equal; and a row
written through the real ``_cache_put`` for the model's own schema and window,
into a *fresh* database, must be served the same body again — with no download
at all, which is the one thing a hit is allowed to promise.

Assertions are about the contract's *meaning* (R3), not only its shape:
``prediction`` and ``signal`` must agree, ``degraded`` must name only known
components and each name it carries must match the condition it names (offline
every condition is true, so the half these assertions can falsify is a *missing*
name, not a spurious one — see the contract test), and ``probability_up`` must be
the model's own pre-adjustment probability — compared against the model loaded
straight from the store, and pinned unchanged when the sentiment overlay is fed a
non-zero score.

``serve`` imports mlflow → opentelemetry, whose runtime-context loader still
calls the removed ``importlib.metadata`` dict interface on Python 3.11, and
FastAPI raises its own deprecation for ``@app.on_event("startup")``.  Both are
upstream of this repository, so the two filters below are scoped to their exact
message and to this module only, exactly as ``tests/test_serve_parity.py`` and
``tests/test_prediction_cache.py`` already do; every other warning still
escalates under ``filterwarnings = ["error"]``.
"""
from __future__ import annotations

import json
import pathlib
import socket
import warnings

import numpy as np
import pandas as pd
import pytest

# ``fastapi.testclient`` → ``starlette.testclient`` emits a collection-time
# ``StarletteDeprecationWarning`` when ``httpx2`` is absent.  A module-level
# ``pytestmark`` cannot catch it: the filter would have to be installed before
# the statement that raises, so the import is wrapped narrowly here and the
# captured message is pinned below, exactly as ``test_audit_regressions.py`` and
# ``test_degradation_visible.py`` do.
with warnings.catch_warnings(record=True) as _import_warnings:
    warnings.simplefilter("always")
    from fastapi.testclient import TestClient

_IMPORT_WARNINGS = [str(w.message) for w in _import_warnings]

#: The one message the wrapped import is allowed to capture.  ``starlette`` warns
#: at its own module import, so a session that imported it earlier records
#: nothing here — the pin is "nothing *else* was captured", in both cases.
EXPECTED_IMPORT_WARNING = (
    "Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` "
    "instead."
)

# ``src/news_sentiment.py`` calls ``warnings.filterwarnings("ignore")`` at import
# time, which would switch off this suite's ``filterwarnings = ["error"]`` for
# every test collected after it.  ``catch_warnings`` saves and restores the
# filter list, so the module's global ignore is discarded when the block exits.
# Imported at module scope on purpose: the one ``vaderSentiment is not installed``
# notice then belongs to collection rather than to whichever test first touches
# the news seam.
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

# ═══════════════════════════════════════════════════════════════════════════════
# The repository's own inputs
# ═══════════════════════════════════════════════════════════════════════════════

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
STOCKS_CSV = _REPO_ROOT / "data" / "raw" / "stocks.csv"

#: The ticker every test drives: mapped in ``model_index.json``, present in the
#: committed universe, with a persisted 60-column contract on disk.
TICKER = "BBCA.JK"

#: A well-formed symbol the index maps nothing to.  The audit's F2 measured this
#: exact shape being answered from the alphabetically first model folder.
UNMAPPED_TICKER = "ZZZZ.JK"

#: The window the request path asks for, and the one the boundary fake answers.
REQUEST_PERIOD = "250d"

#: The four tickers ``_warmup_prediction_cache`` downloads (BBCA.JK is the fifth,
#: and this file requests it deliberately).  No download may be recorded for any
#: of them: that is how "the startup hooks did not run" is proved below.
WARMUP_ONLY_TICKERS = ("BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK")

#: Every component ``PredictResponse.degraded`` may name.  A fourth name, or a
#: free-text reason smuggled into the list, is a contract break.
KNOWN_DEGRADED = frozenset({"sentiment", "fundamentals", "macro"})

#: The eight fields the brief's contract test names.
CONTRACT_FIELDS = (
    "ticker", "prediction", "probability_up", "signal",
    "sentiment_score", "probability_adjusted", "signal_adjusted", "degraded",
)

#: The sentiment score the overlay is fed in the one test that needs a *non-zero*
#: one.  ``vaderSentiment`` is not installed in this virtualenv and no
#: ``NEWSAPI_KEY`` is set, so the news seam is deterministically unavailable here
#: and the overlay's arithmetic could not otherwise be observed at all.  See
#: ``test_probability_up_is_pre_adjustment_not_sentiment_adjusted``.
SEAM_SENTIMENT = 0.8

#: Rounding headroom when a rounded field is compared against a value derived from
#: another rounded field: ``probability_up`` is rounded to 4 dp and
#: ``probability_adjusted`` is computed from the *unrounded* probability, so the
#: two can differ by up to one unit in the last place at 4 dp on each side.
ROUNDING_TOLERANCE = 2e-4

#: The five OHLCV fields, in yfinance's own order.
OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")


def _committed_universe() -> pd.DataFrame:
    """The committed multi-ticker OHLCV snapshot, in yfinance's two-level layout.

    Read with ``header=[0, 1]`` for the same reason ``features.pipeline.load_data``
    does: the file was written by ``yf.download`` for many symbols, so its columns
    are a ``(Ticker, Price)`` ``MultiIndex``.
    """
    return pd.read_csv(STOCKS_CSV, header=[0, 1], index_col=0, parse_dates=True)


def _period_days(period: str) -> int:
    """``"250d"`` → ``250``, and a loud failure for any window this file cannot fake.

    An unhandled period must not silently serve a different window: the served
    features are a function of it, so a wrong window would be a wrong prediction
    under a test that still passed.
    """
    if not period.endswith("d") or not period[:-1].isdigit():
        raise AssertionError(
            f"the boundary fake has no window for period={period!r}; only 'Nd' is "
            "served, from the committed rows"
        )
    return int(period[:-1])


def _yfinance_single_ticker(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """``frame`` as ``yf.download(symbol)`` returns it under yfinance 1.4.1.

    ``download`` concatenates per-symbol frames with ``names=['Ticker', 'Price']``
    and then, for ``group_by='column'``, swaps the levels
    (``yfinance/multi.py:210-218``).  A single symbol therefore still comes back
    with two column levels — ``(Price, Ticker)``, not flat — which is why
    ``serve``'s ``df.columns.droplevel(1)`` exists.  Returning a flat frame here
    would leave that branch unexecuted and quietly test a shape production does
    not produce.
    """
    ordered = frame[list(OHLCV_FIELDS)].copy()
    ordered.columns = pd.MultiIndex.from_product(
        [ordered.columns, [symbol]], names=["Price", "Ticker"]
    )
    return ordered


def _is_loopback(address) -> bool:
    """Whether a ``connect`` target is this host.

    The guard below is deliberately loopback-scoped, so what it promises is
    "no *non-loopback* connect", not "no egress": a route configured to reach
    the internet through a **proxy on loopback** connects to ``127.0.0.1`` and is
    therefore neither refused nor recorded here.  Nothing in this file sets a
    proxy, and the boundary fakes stop every known route before a socket is
    opened, but the guard cannot be quoted against that case.

    The allowance itself is not optional.  A blanket refusal breaks the event
    loop the test client runs on: ``asyncio``'s Proactor self-pipe is a
    ``socket.socketpair``, which on Windows is emulated by connecting to a
    listener on ``127.0.0.1``.  Blocking that would fail the harness, not the
    network.
    """
    if not isinstance(address, tuple) or not address:
        return False
    host = str(address[0])
    return host.startswith("127.") or host in ("::1", "localhost")


class _Boundary:
    """The only network boundary this file allows — and its record.

    Every method here stands where a real client would — the five seams through
    which this process can leave: ``yf.download``, ``yf.Ticker``,
    ``requests.get``, ``requests.post`` and ``fetch_fred_macro``.  A call the
    committed ``data/raw/stocks.csv`` can answer is served from it; every other
    call is recorded and refused, so a code path that wants live data gets nothing
    it could mistake for a measurement.  For the four request-shaped seams that
    means raising; ``fetch_fred_macro`` instead returns the empty frame its own
    outage path returns (see :meth:`fred`), because its caller treats an
    exception differently from an empty frame and no model column is built from
    either.
    """

    def __init__(self, universe: pd.DataFrame):
        self.universe = universe
        self.served: list[tuple[str, str]] = []
        self.refused: list[tuple[str, str]] = []
        self.blocked_sockets: list[str] = []
        #: Inputs and outputs of ``serve.build_inference_features``, newest last.
        self.builds: list[dict] = []
        #: How many times the real ``serve._cache_put`` ran.
        self.puts = 0

    # ── yfinance ──────────────────────────────────────────────────────────────

    def download(self, symbol, *args, **kwargs):
        rows = self.universe[symbol] if symbol in self.universe.columns.levels[0] \
            else None
        if rows is None:
            self.refused.append(("yf.download", symbol))
            raise ConnectionError(
                f"{symbol} is not in the committed universe at {STOCKS_CSV}; this "
                "file refuses to fabricate a frame for it"
            )

        period = kwargs.get("period")
        start_date, end_date = kwargs.get("start"), kwargs.get("end")
        if period is not None:
            end_date = rows.index.max()
            start_date = end_date - pd.Timedelta(days=_period_days(period))
        if start_date is not None:
            rows = rows.loc[pd.Timestamp(start_date):]
        if end_date is not None:
            rows = rows.loc[:pd.Timestamp(end_date)]

        self.served.append(("yf.download", symbol))
        return _yfinance_single_ticker(rows, symbol)

    def ticker(self, symbol, *args, **kwargs):
        self.refused.append(("yf.Ticker", symbol))
        raise ConnectionError(
            f"yf.Ticker({symbol}).info is not served by the committed universe at "
            f"{STOCKS_CSV}; the fundamentals fetch is therefore an outage here, and "
            "the response has to say so"
        )

    # ── requests ──────────────────────────────────────────────────────────────

    def get(self, url, *args, **kwargs):
        self.refused.append(("requests.get", str(url)))
        raise ConnectionError(f"no HTTP egress in this file: GET {url}")

    def post(self, url, *args, **kwargs):
        self.refused.append(("requests.post", str(url)))
        raise ConnectionError(f"no HTTP egress in this file: POST {url}")

    # ── FRED ──────────────────────────────────────────────────────────────────

    def fred(self, start, end):
        """``fetch_fred_macro`` — the route the ``requests`` patches cannot see.

        ``refresh_daily_cache`` calls this on every ``/predict`` (it is scheduled
        as a FastAPI background task) and it really runs here, so with
        ``FRED_API_KEY`` exported a real ``fredapi`` client was constructed and
        eight series were fetched over ``urllib`` — through the socket guard, which
        recorded the connections and failed the teardown only *after* they had
        been made.

        Patched at the function rather than at ``urllib.request.urlopen`` for a
        reason worth keeping: ``fredapi/fred.py:16`` does
        ``urlopen = url_request.urlopen`` at import time, so a later patch of the
        ``urllib.request`` attribute rebinds nothing the client calls.  This seam
        is the one ``refresh_daily_cache`` actually resolves.

        Refused, and for the record: no request is made, no value is fabricated,
        and the attempt is named in ``refused``.  The *return* is the empty frame
        the real function returns for every one of its own failure modes — a
        missing key, a rejected key, a dead endpoint — because it swallows each of
        them internally and never raises.  Raising here instead would be the one
        unfaithful thing this fake could do: ``refresh_daily_cache`` catches it
        before stamping ``_cache_date``, so the day's macro refresh would be
        retried on every request and the hit path would fetch ``IDR=X`` again
        (measured: that broke the "a hit costs no network" pin below).  Nothing is
        served from the empty frame in any case: no FRED series name is among the
        model's 60 columns, and the one macro-derived column in it,
        ``usdidr_stress``, is computed from the ``usdidr_rate``/``usdidr_return``
        pair that ``fetch_usdidr``'s own frame supplies
        (``features/market.py:71-78``).
        """
        self.refused.append(("fetch_fred_macro", f"{start}..{end}"))
        return pd.DataFrame()

    # ── evidence ──────────────────────────────────────────────────────────────

    def downloaded(self) -> list[str]:
        """Every symbol ``yf.download`` was asked for, in order."""
        return [symbol for _kind, symbol in self.served + self.refused
                if _kind == "yf.download"]


class _Api:
    """The app under test: a client, the boundary, and the live module."""

    def __init__(self, serve, client, boundary: _Boundary, tmp_path, monkeypatch):
        self.serve = serve
        self.client = client
        self.boundary = boundary
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch

    def predict(self, ticker: str = TICKER):
        return self.client.post("/predict", json={"ticker": ticker})

    def body(self, ticker: str = TICKER) -> dict:
        response = self.predict(ticker)
        assert response.status_code == 200, response.text
        return response.json()

    def model_and_schema(self):
        """The ticker's model and contract, straight out of the model store.

        Deliberately **not** through ``serve.load_best_model``: the store is the
        authority the app itself resolves through, and going to it directly keeps
        the comparison below independent of the response path.
        """
        from model_store import load_model_for
        return load_model_for(TICKER)

    def fresh_cache(self, name: str) -> None:
        """Point the prediction cache at another, empty database."""
        self.monkeypatch.setattr(
            self.serve, "PRED_CACHE_DB", str(self.tmp_path / name)
        )


@pytest.fixture
def api(tmp_path, monkeypatch):
    """The real app behind a ``TestClient``, with every egress faked and recorded.

    Two layers keep the file offline, and neither alone is enough.  The first is
    the boundary above: the five seams through which this application can leave
    the process are replaced — ``yf.download``, ``yf.Ticker``, ``requests.get``,
    ``requests.post`` and ``fetch_fred_macro`` — so a call either comes back with
    committed rows or is refused (the four request-shaped seams raise; FRED
    returns the empty frame its own outage path returns).  That is the layer that
    actually binds, and it has
    to be: yfinance 1.4.1 talks HTTPS through ``curl_cffi``, whose connection is
    made inside libcurl and is invisible to any Python-level socket patch, so
    patching ``yf.download``/``yf.Ticker`` is the only reliable way to stop it.
    ``fetch_fred_macro`` is patched for the same kind of reason, and at the
    function rather than at ``urllib``: ``fredapi`` captures ``urlopen`` into its
    own namespace when it is first imported (``fredapi/fred.py:16``), so a later
    patch of ``urllib.request.urlopen`` rebinds nothing it calls.  The second
    layer is a socket guard that refuses any **non-loopback** ``connect`` and
    records the attempt: it catches the ``requests``/``urllib3`` routes — the
    BI-rate scrape and NewsAPI — and any other bare Python socket, and it turns
    "there is no other egress" from an assumption into an assertion at the end of
    every test in this file.  Two routes it cannot see are named rather than
    implied: ``curl_cffi``'s libcurl connect (above, which is why the fakes and
    not the guard are what bind) and an egress routed through a proxy on
    loopback, which connects to this host and is therefore invisible to it
    (``_is_loopback``).

    Module state is replaced rather than cleared (``monkeypatch`` restores the
    original, contents and all): ``model_cache`` and ``_fundamentals_cache`` key
    their contents by ``(ticker, today)`` and would otherwise answer this test
    with an earlier test's fetch, and ``_macro_outcomes`` is written by
    ``refresh_daily_cache``, which really runs here — it is a background task on
    every ``/predict`` — so a leftover ``MACRO_NO_FRAME`` from a previous module
    would attach itself to a later test that asserts something else.  The FRED
    seam returns the empty frame the real function returns on an outage rather
    than raising, so ``refresh_daily_cache`` completes and stamps ``_cache_date``
    exactly as it does with no key set.  The outcome recorded is therefore the
    same ``MACRO_NO_FRAME`` — and the same body served — under both configurations
    this file is run in, ``FRED_API_KEY`` unset and set to a dummy.  ``_cache_date``
    is forced to ``None`` anyway, so the request under test always pays a real
    refresh rather than inheriting one from the previous test.
    """
    import requests
    import serve
    import yfinance as yf
    import features
    import features.fetchers as fetchers
    import features.pipeline as pipeline

    boundary = _Boundary(_committed_universe())

    monkeypatch.setattr(yf, "download", boundary.download)
    monkeypatch.setattr(yf, "Ticker", boundary.ticker)
    monkeypatch.setattr(requests, "get", boundary.get)
    monkeypatch.setattr(requests, "post", boundary.post)

    # FRED last, and at *every* module that holds the name rather than at its
    # definition site alone.  ``features/__init__.py``, ``features.pipeline`` and
    # ``serve`` each do ``from features.fetchers import fetch_fred_macro``, so
    # replacing ``fetchers.fetch_fred_macro`` rebinds nothing any of their callers
    # resolves — and ``serve``'s binding is the one ``refresh_daily_cache``
    # actually calls.  Leaving it out is what
    # ``FRED_API_KEY=<dummy> pytest tests/test_api_end_to_end.py`` exposed: 6
    # failed, 2 passed, 6 errors, with eight real connections to FRED's host
    # recorded by the socket guard before it could fail the teardown.
    for module in (fetchers, features, pipeline, serve):
        monkeypatch.setattr(module, "fetch_fred_macro", boundary.fred)

    real_connect = socket.socket.connect

    def guarded_connect(self, address, *args, **kwargs):
        if _is_loopback(address):
            return real_connect(self, address, *args, **kwargs)
        boundary.blocked_sockets.append(str(address))
        raise AssertionError(
            f"a real network connection was attempted to {address!r}; every egress "
            "in this file is faked at its own boundary"
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    # The prediction cache, and the caches whose contents are keyed by "today".
    monkeypatch.setattr(serve, "PRED_CACHE_DB",
                        str(tmp_path / "prediction_cache.db"))
    monkeypatch.setattr(serve, "model_cache", {})
    monkeypatch.setattr(serve, "_fundamentals_cache", {})
    monkeypatch.setattr(serve, "_macro_outcomes", {})
    # The daily macro state a cold process is in.  The refresh really runs and
    # really fails (IDR=X is refused), which is the offline state production sees
    # when Yahoo is unreachable — named in ``degraded``, never silently zero.
    monkeypatch.setattr(serve, "_cache_date", None)
    monkeypatch.setattr(serve, "_usdidr_cache", None)
    monkeypatch.setattr(serve, "_fred_cache", None)
    monkeypatch.setattr(serve, "_bi_cache", None)
    # The universe is read from the real ``data/raw`` tree — that is the point —
    # but only for the duration of one test: the snapshot is a module-level cache,
    # so leaving it populated would hand a later module this test's frame.
    monkeypatch.setattr(serve, "_universe_context", None)
    # ``get_ticker_sentiment`` writes ``data/sentiment/<ticker>.json`` on every
    # call, including the unavailable one.  Redirected so the suite does not
    # mutate the repository, and so run 2 does not read run 1's answer.
    monkeypatch.setattr(news_sentiment, "CACHE_DIR",
                        str(tmp_path / "sentiment"))

    # Spies that delegate: the real functions still run, so the counts below are
    # evidence about production rather than about a stand-in.
    real_build = serve.build_inference_features
    real_put = serve._cache_put

    def capturing_build(df, ticker, fundamentals):
        featured = real_build(df, ticker, fundamentals)
        boundary.builds.append({
            "ohlcv": df, "ticker": ticker, "fundamentals": fundamentals,
            "features": featured,
        })
        return featured

    def counting_put(ticker, feature_row, schema, window=None):
        boundary.puts += 1
        return real_put(ticker, feature_row, schema,
                        serve.CACHE_WINDOW_DEFAULT if window is None else window)

    monkeypatch.setattr(serve, "build_inference_features", capturing_build)
    monkeypatch.setattr(serve, "_cache_put", counting_put)

    # Constructed directly, never entered: see the module docstring.
    client = TestClient(serve.app)
    yield _Api(serve, client, boundary, tmp_path, monkeypatch)

    assert boundary.blocked_sockets == [], (
        "this test attempted a real connection: " f"{boundary.blocked_sockets}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# The contract, on a real model, through the real app
# ═══════════════════════════════════════════════════════════════════════════════

def test_predict_contract_on_a_real_model(api):
    """The brief's contract test, plus what the fields have to *mean* (R3).

    The request is a genuine cache miss: the database is empty, so the app
    downloads (through the boundary), builds the 60-column feature row with the
    builder training uses, aligns it to the model's persisted contract, predicts,
    caches the row and answers.  Five things are asserted beyond key presence:

    * ``prediction`` and ``signal`` agree — 1 is BUY and 0 is SELL, and a body
      whose two halves disagree is exactly the "signal-shaped payload" that must
      not be readable as a prediction;
    * ``degraded`` names only known components, without duplicates, and each name
      is present exactly when the condition it names holds — the news seam's own
      ``available`` flag for ``"sentiment"``, the fundamentals status seam's for
      ``"fundamentals"``, the built row's ``usdidr_rate`` for ``"macro"``.  What
      that shows offline is one-directional, and the docstring says so rather than
      implying more: every condition is true in this process, so each ``==``
      falsifies a *missing* name (the silent fallback F8 was about) and cannot
      falsify a spurious one — a ``serve`` that appended these names
      unconditionally would pass this test.  The other direction has real teeth
      one test over
      (``test_probability_up_is_pre_adjustment_not_sentiment_adjusted`` feeds the
      sentiment seam a measurement and requires the name to be gone);
    * ``probability_up`` is the model's own output: the model is loaded from the
      store here and applied to the feature row this request built, and the two
      probabilities must be the same number;
    * the frame the request built from is the committed universe, row for row.
      Stated precisely, that pins the *fixture's* window — the last 250 calendar
      days of ``data/raw/stocks.csv``, ending on its own last row — and the
      row-preserving transforms the request path applies to it (``serve``'s
      ``droplevel(1)`` on yfinance's column ``MultiIndex``, and
      ``ta.utils.dropna``), and it deliberately pins no more than that: both sides
      of the comparison are built from the same committed rows, so it would pass
      even if the features *served* were flat.  The values the model really read
      are the ones the ``probability_up`` assertion recomputes from that frame;
    * the FRED route was **reached by this request and refused by the boundary**
      (``refresh_daily_cache`` runs as a background task on it), so the file's
      offline property rests on the seam rather than on ``FRED_API_KEY`` being
      absent from the environment.
    """
    api.fresh_cache("miss.db")
    body = api.body()

    print("\nBBCA.JK - full response body (cache miss):")
    print(json.dumps(body, indent=2))

    # ── shape ────────────────────────────────────────────────────────────────
    for field in CONTRACT_FIELDS:
        assert field in body, field
    assert set(body) == set(api.serve.PredictResponse.model_fields), sorted(body)
    assert body["ticker"] == TICKER
    assert body["prediction"] in (0, 1), body
    assert body["signal"] in ("BUY", "SELL"), body
    assert 0.0 <= body["probability_up"] <= 1.0, body
    assert 0.0 <= body["probability_adjusted"] <= 1.0, body

    # ── meaning ──────────────────────────────────────────────────────────────
    assert body["signal"] == ("BUY" if body["prediction"] == 1 else "SELL"), body

    degraded = body["degraded"]
    assert isinstance(degraded, list), degraded
    assert all(isinstance(name, str) for name in degraded), degraded
    assert len(degraded) == len(set(degraded)), degraded
    assert set(degraded) <= KNOWN_DEGRADED, degraded
    # The sentiment label and the value must agree: ``0.0`` is the overlay's
    # fallback, so a named failure may never be reported at a non-zero score.
    # The converse is deliberately *not* asserted on the score — a genuinely
    # neutral news day also reads ``0.0`` — so the equivalence is taken against
    # the seam's own status, which is what the label is derived from (F8).
    if "sentiment" in degraded:
        assert body["sentiment_score"] == 0.0, body
    import features.fetchers as fetchers

    _score, sentiment_available, _reason = fetchers.fetch_news_sentiment_status(TICKER)
    assert ("sentiment" in degraded) == (not sentiment_available), body
    # ...and the fundamentals label tracks the status seam's own answer rather
    # than being inferred from the eight numbers (which cannot say).
    _values, fundamentals_available, _reason = api.serve.get_fundamentals(TICKER)
    assert ("fundamentals" in degraded) == (not fundamentals_available), body

    # ── the model's own number ───────────────────────────────────────────────
    model, schema = api.model_and_schema()
    assert len(schema) == 60, len(schema)
    assert len(set(schema)) == 60, "the persisted contract repeats a column"

    assert api.boundary.builds, "the request built no features at all"
    built = api.boundary.builds[-1]
    from features.schema import align_to_schema

    model_input = align_to_schema(built["features"], schema)
    expected_probability = round(float(model.predict_proba(model_input)[0][1]), 4)
    expected_prediction = int(model.predict(model_input)[0])

    assert body["probability_up"] == expected_probability, (
        f"the served probability is not this model's own: {body['probability_up']} "
        f"against {expected_probability}"
    )
    assert body["prediction"] == expected_prediction, body

    # The macro label names the fabricated input or it is absent — measured on the
    # row the request actually built, not on live state that may since have moved.
    row = built["features"].iloc[0]
    if "usdidr_rate" in row.index:
        assert ("macro" in degraded) == bool(pd.isna(row["usdidr_rate"])), body

    # ── the committed data is what was served ────────────────────────────────
    frame = built["ohlcv"]
    committed = _committed_universe()[TICKER]
    assert frame.index.max() == committed.index.max(), (
        "the served window does not end on the committed universe's last row"
    )
    assert frame.index[0] >= committed.index.max() - pd.Timedelta(
        days=_period_days(REQUEST_PERIOD)), frame.index[0]
    # Values and index, not dtypes: ``ta.utils.dropna`` (the same call the request
    # path makes) rewrites ``Volume`` as ``float64``, and asserting the dtype of a
    # column this app has already been through would pin its library's behaviour
    # rather than the served data.
    pd.testing.assert_frame_equal(
        frame, committed.loc[frame.index, list(OHLCV_FIELDS)], check_dtype=False
    )
    assert ("yf.download", TICKER) in api.boundary.served, api.boundary.served

    # ── the FRED route was reached, and the boundary is what stopped it ──────
    #
    # ``refresh_daily_cache`` is a background task on this same request and it
    # calls ``fetch_fred_macro`` unconditionally — the ``FRED_API_KEY`` check
    # lives *inside* the real function — so this record is produced whether or not
    # a key is exported.  That is what makes it a pin on the boundary instead of
    # on the environment: remove the FRED seam and the call goes to ``fredapi`` →
    # ``urllib`` instead, leaving this list empty and failing this assertion, with
    # the socket guard's teardown failing only *after* the connections are already
    # made — both were measured with the seam removed and a dummy key exported.
    print(f"refused and recorded for this request: {api.boundary.refused}")
    assert any(entry[0] == "fetch_fred_macro" for entry in api.boundary.refused), (
        "the request never reached the FRED seam, so the boundary does not cover "
        f"the route it takes: {api.boundary.refused}"
    )


def test_cache_hit_serves_exactly_the_miss_body(api):
    """F7 at the API level: the fast path cannot answer differently.

    The first request is a miss and writes the row it predicted on; the second is
    a hit and predicts on the row it read back.  The two bodies must be equal
    dict for dict.  "The second one was a hit" is not taken on trust: the real
    ``_cache_put`` is counted, and a second miss would have to write again.
    """
    api.fresh_cache("hit.db")

    miss = api.body()
    assert api.boundary.puts == 1, "the cold request did not cache its row"

    model, schema = api.model_and_schema()
    cached = api.serve._cache_get(TICKER, schema, api.serve.CACHE_WINDOW_DEFAULT)
    assert cached is not None, "a miss left nothing for the hit path to read"
    assert sorted(cached.index) == sorted(schema), "the cached row is another schema"
    assert list(cached.index) == list(schema), "the cached row is out of contract order"

    hit = api.body()
    assert api.boundary.puts == 1, (
        "the second request recomputed and re-cached: it was a miss, not a hit"
    )

    print("\nBBCA.JK - cache MISS body:")
    print(json.dumps(miss, indent=2))
    print("BBCA.JK - cache HIT body (same ticker, same day):")
    print(json.dumps(hit, indent=2))
    print(f"identical: {hit == miss}")

    assert hit["ticker"] == miss["ticker"]
    assert hit == miss, {
        field: (miss.get(field), hit.get(field))
        for field in set(miss) | set(hit) if miss.get(field) != hit.get(field)
    }


def test_a_row_seeded_through_cache_put_is_served_without_a_download(api):
    """The hit path, seeded explicitly and on its own — and it never downloads.

    R1: the row is written through the *real* ``_cache_put`` under the model's
    own schema and the serving window, into a database that has never held
    anything else, and the request that follows must return the same body the
    miss produced.  The download count is asserted unchanged across the seeded
    request: whatever a hit costs, it costs no network, which is the property
    the two paths were diverging behind (F7).
    """
    api.fresh_cache("capture.db")
    miss = api.body()

    model, schema = api.model_and_schema()
    row = api.serve._cache_get(TICKER, schema, api.serve.CACHE_WINDOW_DEFAULT)
    assert row is not None

    api.fresh_cache("seeded.db")          # a database with no row in it at all
    api.serve._cache_put(TICKER, row, schema, api.serve.CACHE_WINDOW_DEFAULT)
    assert api.boundary.puts == 2, "the seed did not go through the real _cache_put"

    downloads_before = list(api.boundary.downloaded())
    hit = api.body()
    assert api.boundary.downloaded() == downloads_before, (
        "the hit path went to the network: "
        f"{set(api.boundary.downloaded()) - set(downloads_before)}"
    )

    print("\nBBCA.JK - seeded cache HIT body:")
    print(json.dumps(hit, indent=2))
    assert hit == miss, {
        field: (miss.get(field), hit.get(field))
        for field in set(miss) | set(hit) if miss.get(field) != hit.get(field)
    }


def test_probability_up_is_pre_adjustment_not_sentiment_adjusted(api):
    """R3: the served probability is the model's, before the overlay touches it.

    Two identical hit requests, one difference between them: the second is run
    with the news seam reporting a *measurement* (``0.8``) instead of the outage
    this virtualenv always has.  That makes the assertion falsifiable in the way
    a key-presence check is not — a body that put the adjusted probability in
    ``probability_up`` would move here, and the test would see it.

    ``probability_up`` must not move: it is the pre-adjustment probability, and
    it must still equal the model's own output.  ``probability_adjusted`` must
    move by exactly the overlay's weight, and by that much only.
    """
    import features.fetchers as fetchers

    api.fresh_cache("overlay.db")
    baseline = api.body()
    assert baseline["sentiment_score"] == 0.0, baseline
    assert baseline["probability_adjusted"] == baseline["probability_up"], baseline

    model, schema = api.model_and_schema()
    row = api.serve._cache_get(TICKER, schema, api.serve.CACHE_WINDOW_DEFAULT)
    assert row is not None
    api.fresh_cache("overlay-seeded.db")
    api.serve._cache_put(TICKER, row, schema, api.serve.CACHE_WINDOW_DEFAULT)

    api.monkeypatch.setattr(
        fetchers, "fetch_news_sentiment_status",
        lambda ticker: (SEAM_SENTIMENT, True, None),
    )
    adjusted = api.body()

    print("\nBBCA.JK - same row, sentiment seam reporting "
          f"{SEAM_SENTIMENT}:")
    print(json.dumps(adjusted, indent=2))

    assert adjusted["sentiment_score"] == SEAM_SENTIMENT, adjusted
    assert "sentiment" not in adjusted["degraded"], adjusted

    # The prediction itself is the model's, unmoved by the overlay...
    assert adjusted["probability_up"] == baseline["probability_up"], (
        "probability_up moved with the sentiment overlay — it is not the "
        "pre-adjustment probability"
    )
    assert adjusted["prediction"] == baseline["prediction"], adjusted
    assert adjusted["signal"] == baseline["signal"], adjusted

    # ...and it is still the model's own number.
    from features.schema import align_to_schema

    # ``align_to_schema`` takes the one-row frame every predicting path builds,
    # so the row read back from the cache is widened to one exactly as
    # ``serve._prediction_frame`` widens it.
    one_row = align_to_schema(pd.DataFrame([row]), schema)
    expected = round(float(model.predict_proba(one_row)[0][1]), 4)
    assert adjusted["probability_up"] == expected, adjusted

    # The adjusted pair carries the overlay, and only the overlay.
    weight = api.serve.SENTIMENT_WEIGHT
    expected_adjusted = min(
        1.0, max(0.0, adjusted["probability_up"] + weight * SEAM_SENTIMENT)
    )
    assert adjusted["probability_adjusted"] == pytest.approx(
        expected_adjusted, abs=ROUNDING_TOLERANCE
    ), adjusted
    assert adjusted["probability_adjusted"] != adjusted["probability_up"], adjusted
    assert adjusted["signal_adjusted"] == (
        "BUY" if adjusted["probability_adjusted"] >= 0.50 else "SELL"
    ), adjusted


# ═══════════════════════════════════════════════════════════════════════════════
# An unmapped ticker is an error, not a guess (F2, R2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_unknown_ticker_is_an_error_not_a_guess(api):
    """Exactly 404, with no signal-shaped payload anywhere in the body.

    The brief tolerated ``in (404, 422)``.  It is 404 and only 404: the handler
    catches ``ModelNotFound`` and raises ``HTTPException(404)``, and a 422 would
    mean the request had failed validation — a different event that the same
    assertion would hide.

    The audit's F2 was worse than a wrong status: a made-up symbol was answered
    with the alphabetically first model folder's prediction under HTTP 200, so
    the *keys* matter as much as the code.  ``prediction``/``signal``/
    ``probability_up`` must be absent, and — because the defect was a
    substitution rather than a miscalculation — no download may be attempted for
    a ticker nothing maps.
    """
    api.fresh_cache("unknown.db")
    response = api.predict(UNMAPPED_TICKER)
    body = response.json()

    print(f"\n{UNMAPPED_TICKER} - status {response.status_code}, body:")
    print(json.dumps(body, indent=2))

    assert response.status_code == 404, response.text
    for field in ("prediction", "signal", "probability_up", "probability_adjusted",
                  "signal_adjusted", "sentiment_score", "degraded"):
        assert field not in body, f"{field} in a failed response: {body}"
    assert UNMAPPED_TICKER in json.dumps(body), (
        "the error does not name the ticker it refused"
    )
    assert UNMAPPED_TICKER not in api.boundary.downloaded(), (
        "an unmapped ticker spent a download before being refused"
    )

    # The same client still answers a mapped ticker: the 404 is about this
    # symbol, not a broken app or a poisoned cache.
    assert api.body(TICKER)["ticker"] == TICKER


# ═══════════════════════════════════════════════════════════════════════════════
# The offline proof
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_startup_warmup_touched_the_network(api):
    """``TestClient(serve.app)`` skips the startup hooks — a fixture tripwire.

    Starlette runs ``on_startup`` handlers only from ``__enter__``.  Entering the
    context manager here would run ``startup_load_index``, which schedules
    ``_warmup_prediction_cache``: five tickers downloaded from Yahoo, one of them
    the ticker this file requests, at a moment the test does not control.

    What this test asserts is a counterfactual, and it says so rather than
    claiming a behavioural proof: the fixture constructs the client directly, so
    no code path reachable from this file can run a startup hook, and the
    assertion cannot fail as the file stands.  It is kept as a tripwire on the
    fixture — the day someone enters the client as a context manager, or the
    warmups are scheduled from somewhere else, a name nothing requested appears in
    the record and this is the test that names it.  The half it observes today is
    that **no download was recorded for the four warmup tickers this file never
    asks about**, while ``BBCA.JK`` was downloaded through the boundary.
    """
    api.fresh_cache("warmup.db")
    body = api.body()

    downloaded = api.boundary.downloaded()
    print(f"\ndownloads recorded for the whole request: {downloaded}")

    assert ("yf.download", TICKER) in api.boundary.served, downloaded
    warmup_leaks = [t for t in WARMUP_ONLY_TICKERS if t in downloaded]
    assert warmup_leaks == [], (
        f"the startup prediction warmup ran: {warmup_leaks} were downloaded "
        "without being requested"
    )
    assert body["ticker"] == TICKER


def test_the_socket_guard_refuses_off_host_and_allows_loopback(api):
    """The second net under the boundary fakes, tested in both directions.

    A guard that never fires proves nothing, so the refusal is demonstrated on
    the real ``socket`` module — and so is its scope: it refuses **non-loopback**
    connects, and loopback must still work, because the test client's event loop
    builds a self-pipe out of a ``socketpair``, which on Windows is a connection
    to ``127.0.0.1``.  A guard that blocked that would fail the harness rather
    than the network.  Those are the only two cases here: an egress routed through
    a proxy on loopback connects to this host, so the guard neither refuses nor
    records it (``_is_loopback``), and this test must not be read as covering
    that.
    """
    probe = socket.socket()
    try:
        with pytest.raises(AssertionError, match="real network connection"):
            probe.connect(("203.0.113.7", 80))
    finally:
        probe.close()
    assert api.boundary.blocked_sockets == ["('203.0.113.7', 80)"], \
        api.boundary.blocked_sockets
    # The one attempt above was made *by this test*, on purpose.  Its record is
    # dropped so the fixture's end-of-test assertion keeps its meaning: nothing
    # else in the test tried to leave the host.
    api.boundary.blocked_sockets.clear()

    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client = socket.socket()
        try:
            client.connect(listener.getsockname())     # loopback: not refused
        finally:
            client.close()
    finally:
        listener.close()


def test_the_import_wrapper_hides_nothing_else():
    """The one wrapped import captured the one known starlette message, or none."""
    assert set(_IMPORT_WARNINGS) <= {EXPECTED_IMPORT_WARNING}, _IMPORT_WARNINGS
    assert len(_IMPORT_WARNINGS) <= 1, _IMPORT_WARNINGS
