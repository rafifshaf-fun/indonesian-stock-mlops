import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import hashlib
import json

import mlflow
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from ta.utils import dropna
from datetime import date, timedelta
import asyncio
import traceback

from config import (
    TICKERS, MLFLOW_EXPERIMENT, FEATURE_FLAGS, get_logger,
)
from features import (
    fetch_fundamentals_status, fetch_usdidr, fetch_fred_macro, fetch_bi_rate,
    load_data, load_ihsg_data, compute_cross_stock_features,
)
# The one builder both paths share (Task 1.2).  Imported from its own module
# because features/__init__.py does not re-export it.
from features.pipeline import build_features
# The model's feature contract (Task 1.4), applied wherever a computed frame
# meets a model's columns, and the one model selector (Task 1.6).  Both are
# imported rather than reimplemented here: this module no longer knows how the
# model store is laid out or how a model's columns are discovered.
from features.schema import FeatureSchemaError, align_to_schema
from model_store import ModelNotFound, load_model_for, load_model_index

logger = get_logger(__name__)

# ── Dynamic path resolution ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The model store's path lives in model_store.model_dir() alone; this module
# deliberately builds no part of it (Task 1.6, R3).

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

app = FastAPI(title="Indonesian Stock Prediction API", version="2.0.0")
Instrumentator().instrument(app).expose(app)

# ── Startup: load model index and warm up cache in background ─────────────
_model_index = {}

@app.on_event("startup")
async def startup_load_index():
    """Load the model index at startup. Models are loaded lazily on first use."""
    global _model_index

    # model_store reads the index file, warns loudly about every way it can be
    # unusable, and returns {} rather than raising, so startup survives a
    # missing index.  An empty mapping is not benign, though: it means every
    # ticker now 404s, which is the whole point of deleting the old fallback.
    _model_index = load_model_index()
    if _model_index:
        logger.info("Loaded model index with %d tickers", len(_model_index))
    else:
        logger.warning(
            "Model index maps no tickers — every /predict will fail with 404 "
            "until scripts/build_model_index.py has been run"
        )

    # Warm up cache in background (don't block startup)
    asyncio.create_task(_warmup_model_cache())
    # Also pre-warm prediction cache for fastest-first-hit
    asyncio.create_task(_warmup_prediction_cache())


async def _warmup_model_cache():
    """Gradually warm up model cache in background after startup."""
    logger.info("Background model warmup started (%d tickers)", len(_model_index))
    loaded = 0
    for ticker in list(_model_index)[:10]:  # Top 10 only
        if ticker in model_cache:
            continue
        try:
            model_cache[ticker] = load_model_for(ticker)
            loaded += 1
            logger.debug("Warmup: loaded %s", ticker)
        except Exception as e:
            # A model the index names but which cannot be loaded is a
            # degradation: it is reported, not swallowed at debug level.
            logger.warning("Warmup failed for %s: %s", ticker, e)
    logger.info("Background warmup complete: %d models loaded", loaded)


async def _warmup_prediction_cache():
    """Pre-warm prediction cache for a few key tickers after startup.

    Warms only the top 5 most liquid tickers to avoid Yahoo Finance rate limits.
    Remaining tickers are cached on first user request.
    """
    priority = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]
    logger.info("Prediction cache pre-warm started (top %d tickers)...", len(priority))
    warmed = 0
    for ticker in priority:
        try:
            # The model's own contract, from the one model selector, *before* the
            # download: a ticker the index does not map has nothing to cache, so
            # warming it would spend a Yahoo Finance call to build a row that is
            # then dropped.  ``load_best_model`` is that selector plus the
            # per-process model cache, and it raises the same ``ModelNotFound`` /
            # ``FeatureSchemaError`` the request path raises.
            model, trained_cols = load_best_model(ticker)

            # The key carries the model's schema, so the check has to be made
            # with it: a row cached for another schema is not a warm entry.
            if _cache_get(ticker, trained_cols, CACHE_WINDOW_DEFAULT) is not None:
                warmed += 1
                continue

            df = await asyncio.to_thread(
                yf.download, ticker, period="250d", auto_adjust=True, progress=False
            )
            if df.empty or len(df) < 50:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = dropna(df)

            fundamentals, _f_available, _f_reason = await asyncio.to_thread(
                get_fundamentals, ticker
            )
            latest = build_inference_features(df, ticker, fundamentals)

            # The model's own contract decides the column set and its order —
            # the same call the /predict slow path makes, so a pre-warmed row is
            # the row a request would compute and store.
            feature_row = _warm_feature_row(latest, trained_cols)
            _cache_put(ticker, feature_row, trained_cols, CACHE_WINDOW_DEFAULT)
            warmed += 1
            logger.debug("Pre-warmed: %s", ticker)
        except Exception as e:
            # Best-effort warmup, but a skip is a degradation and is reported.
            logger.warning("Pre-warm skipped for %s: %s", ticker, e)
            continue
    logger.info("Prediction cache pre-warm complete: %d/%d tickers cached",
                warmed, len(priority))

try:
    PREDICTION_COUNTER = Counter("stock_predictions_total", "Total predictions made", ["ticker", "signal"])
    CONFIDENCE_HISTOGRAM = Histogram("prediction_confidence", "Model confidence score", ["ticker"])
except ValueError:
    from prometheus_client import REGISTRY
    PREDICTION_COUNTER = REGISTRY._names_to_collectors.get("stock_predictions_total")
    CONFIDENCE_HISTOGRAM = REGISTRY._names_to_collectors.get("prediction_confidence")

# ticker -> (model, feature_columns), exactly what model_store.load_model_for
# returns: the columns are cached with the model so a cached model can never be
# aligned against a different model's contract.
model_cache = {}

# ── Prediction Cache (SQLite) — sub-second predictions after first hit ──────
#
# One cache, one key, one reconciliation (F6, F7).  What this block replaced:
#
# * the cache stored a bare ``numpy`` vector under a key of ``(ticker, date)``,
#   and the request path rebuilt a feature frame by **zipping that vector
#   positionally** onto the current model's column order, defaulting any
#   shortfall to ``0.0``.  A vector written by a different model — a retrain, or
#   another ticker's model — was silently permuted across features.
# * the cache-hit path filled a column the model wanted but the vector lacked
#   with ``0.0``, while the cache-miss path filled it with ``NaN``, so the same
#   ticker on the same day had two different answers (measured 0.187 against
#   0.303, straddling the 0.50 threshold).
#
# The fix is structural rather than defensive: the key now carries a fingerprint
# of the model's schema, so values written for a different column set are a
# **miss** and a positional zip has nothing to zip; the row itself is stored
# keyed by column *name*; and the single frame builder below is the only thing
# that turns a row into a model input, so the fast and slow paths cannot diverge.
import sqlite3

PRED_CACHE_DB = os.path.join(BASE_DIR, "data", "prediction_cache.db")

#: The cache key's column set, **in the order the table is written**.  ``date``
#: is not part of the key: it is overwritten on every put and checked for "today"
#: on every get, so one key holds only the current day's row.
CACHE_COLUMNS = ("ticker", "window", "schema_hash", "date", "features")

#: The declared primary key, in key order.  It is what makes ``INSERT OR
#: REPLACE`` an update rather than a second row for the same key, and it is
#: checked separately from the columns because a table can carry all five names
#: and no key at all.
CACHE_PRIMARY_KEY = ("ticker", "window", "schema_hash")

#: Fingerprint namespace and version.  Bumped rather than migrated when the
#: stored encoding changes: an entry written by an older encoding then misses
#: instead of being read as if it matched, which is the whole point of the key.
_CACHE_FINGERPRINT_VERSION = "v1"

#: The one documented trading-window default (R1).  The cache key includes the
#: window, but serving has no notion of a trading window yet — Task 1.10 builds
#: the IDX calendar and Phase 4 schedules the three daily signals — so today
#: every caller passes this same value and behaviour is unchanged.  When Task
#: 1.10 has real windows it passes them through the parameter that already
#: exists here, and a 09:00 row can never be served as a 13:30 one: a different
#: window is a different key, hence a miss, hence a recompute.
CACHE_WINDOW_DEFAULT = "daily"


def _cache_schema_hash(schema) -> str:
    """Fingerprint a model's ordered feature schema for the cache key.

    Order is part of the fingerprint, not just the name set: XGBoost is fitted
    on a positionally-scaled matrix, so the same names in a different order are a
    different model input (see ``features.schema``).  Names are stringified first
    so a caller passing a ``numpy`` array of names — which ``load_model_for``
    does — fingerprints identically to a caller passing a ``list``.

    Returns:
        A stable hex digest.  A schema whose names cannot be marshalled is not
        silently skipped: it is a programming error, not a cache advisory, and
        :class:`TypeError` says so.
    """
    names = [str(name) for name in schema]
    payload = json.dumps(names, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(
        f"{_CACHE_FINGERPRINT_VERSION}|{payload}".encode("utf-8")
    ).hexdigest()


def _cache_table_info(path) -> list[tuple]:
    """``PRAGMA table_info(cache)`` rows, or ``[]`` when the table does not exist.

    Each row is ``(cid, name, type, notnull, default, pk)``.  ``pk`` is the
    1-based position of the column inside the table's PRIMARY KEY, or ``0`` when
    the column is not part of one.  The migration needs the whole row rather than
    the names: a table with the right *names* in another order, or with no key at
    all, still takes the positional ``INSERT OR REPLACE`` below.
    """
    conn = sqlite3.connect(str(path))
    try:
        return [tuple(row) for row in conn.execute("PRAGMA table_info(cache)")]
    finally:
        conn.close()


def _cache_columns(path) -> list[str]:
    """``cache``'s column names at ``path``, in table order; ``[]`` when absent.

    Ordered, not a set: see :func:`_cache_table_info`.
    """
    return [row[1] for row in _cache_table_info(path)]


def _cache_primary_key(info) -> list[str]:
    """The primary key described by ``_cache_table_info`` rows, in key order."""
    return [name for _, name in sorted((row[5], row[1]) for row in info if row[5])]


def _get_cache_conn():
    os.makedirs(os.path.dirname(PRED_CACHE_DB), exist_ok=True)
    conn = sqlite3.connect(PRED_CACHE_DB)
    # ``CREATE TABLE IF NOT EXISTS`` will not change the columns of a table that
    # already exists, so a database created by the pre-Task-1.8 version keeps its
    # old shape ``(ticker, date, features BLOB, ohlcv BLOB)`` and *every* query
    # against the new columns fails.  The shape is inspected and a mismatched
    # table is dropped and recreated (R5).  Dropping is safe by construction —
    # the contents are a regenerable cache, and the cost is one slow request per
    # ticker — and it must not throw on a fresh database, where there is no
    # table to inspect at all.
    #
    # Both the ordered column list *and* the primary key are compared.  Names
    # alone would accept a table whose columns are permuted, and the write below
    # binds positionally, so ``window`` would be written into ``ticker``; names
    # alone would also accept a keyless table, where ``INSERT OR REPLACE`` never
    # replaces and one key accumulates rows.
    info = _cache_table_info(PRED_CACHE_DB)
    existing = [row[1] for row in info]
    existing_key = _cache_primary_key(info)
    if info and (existing != list(CACHE_COLUMNS)
                 or existing_key != list(CACHE_PRIMARY_KEY)):
        logger.warning(
            "Prediction cache %s has the old shape %s with primary key %s — "
            "dropping and recreating as %s with primary key %s; its contents are "
            "a regenerable cache, so the cost is one recompute per ticker",
            PRED_CACHE_DB, existing, existing_key,
            list(CACHE_COLUMNS), list(CACHE_PRIMARY_KEY),
        )
        conn.execute("DROP TABLE cache")
    conn.execute("CREATE TABLE IF NOT EXISTS cache "
                 "(ticker TEXT, window TEXT, schema_hash TEXT, date TEXT, "
                 "features TEXT, PRIMARY KEY (ticker, window, schema_hash))")
    conn.commit()
    return conn


def _cache_put(ticker: str, feature_row: pd.Series, schema: list[str],
               window: str = CACHE_WINDOW_DEFAULT) -> None:
    """Store one ticker's latest feature row under the current model's key.

    Args:
        ticker: Exact ticker key, e.g. ``"BBCA.JK"``.
        feature_row: The feature values, **keyed by column name**.  Storing the
            names is what removes the positional zip: a value can no longer be
            read back under a column it was not computed for.  ``NaN`` is a
            legitimate value here (the served row carries it for the columns
            ``tests/test_serve_parity.py`` pins), and round-trips as ``NaN``.
        schema: The model's ordered feature contract — the same list passed to
            ``align_to_schema``.  It is fingerprinted into the key, so a row
            written for one column set is invisible to another.
        window: The trading window this row belongs to (R1); see
            :data:`CACHE_WINDOW_DEFAULT`.

    R4: the values are JSON, **never** ``pickle``.  This database is intended to
    become shared, and a shared store of pickled objects is a
    code-execution vector for anyone who can write to it.  JSON is data — a
    reader can be handed one without becoming an executor.
    """
    record = {str(name): _as_float(value, str(name))
              for name, value in feature_row.items()}
    payload = json.dumps(record, allow_nan=True)
    # ``with conn:`` commits; it does not close (``sqlite3.Connection.__exit__``
    # only ends the transaction), so the handle is released explicitly here as it
    # already was in ``_cache_get`` and ``_cache_columns``.
    conn = _get_cache_conn()
    try:
        with conn:
            conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?)",
                         (ticker, str(window), _cache_schema_hash(schema),
                          str(date.today()), payload))
    finally:
        conn.close()


def _cache_get(ticker: str, schema: list[str],
               window: str = CACHE_WINDOW_DEFAULT) -> pd.Series | None:
    """Return ``ticker``'s cached feature row for today, or ``None``.

    ``None`` covers every way the cache cannot answer: no entry, an entry from
    another day, another window, or a schema that fingerprints differently from
    the model about to consume it.  A schema mismatch is deliberately a **miss**
    rather than an adaptation — that is what makes the positional zip
    unreachable, because a hit can only ever carry values computed for exactly
    these columns (F6).

    Returns:
        ``pd.Series`` keyed by column name, or ``None``.  A row whose encoding
        decodes yet does not match the requested schema is a ``warning`` and a
        miss: an entry that cannot be used is a degradation worth seeing.  So is
        a stored value that is not a number (only an external writer can produce
        one — this store is intended to be shared): it is a ``warning`` naming
        the column and a miss, never the ``ValueError`` a bare
        ``pd.Series(..., dtype=float)`` raises, which would surface as a 500.
    """
    key = _cache_schema_hash(schema)
    conn = _get_cache_conn()
    try:
        row = conn.execute(
            "SELECT features FROM cache WHERE ticker = ? AND window = ? "
            "AND schema_hash = ? AND date = ?",
            (ticker, str(window), key, str(date.today())),
        ).fetchone()
    finally:
        conn.close()

    if row is None or not row[0]:
        return None

    try:
        stored = json.loads(row[0])
    except ValueError as exc:
        logger.warning("Prediction cache entry for %s is not valid JSON (%s) — "
                       "treated as a miss", ticker, exc)
        return None
    if not isinstance(stored, dict):
        logger.warning("Prediction cache entry for %s is not keyed by column "
                       "name (found %s) — treated as a miss",
                       ticker, type(stored).__name__)
        return None

    expected = [str(name) for name in schema]
    if sorted(stored) != sorted(expected):
        logger.warning(
            "Prediction cache entry for %s does not match the model's schema "
            "(%d stored columns vs %d expected) — treated as a miss",
            ticker, len(stored), len(expected),
        )
        return None

    values: dict[str, float] = {}
    for name in expected:
        try:
            values[name] = float(stored[name])
        except (TypeError, ValueError):
            logger.warning(
                "Prediction cache entry for %s holds a non-numeric value for "
                "column %r (%r) — treated as a miss", ticker, name, stored[name],
            )
            return None
    return pd.Series(values, dtype=float)


def _as_float(value, column: str):
    """``float`` for a feature value, or ``NaN`` when it is not numeric.

    ``NaN`` is the honest reading of "this column carried no number": it is what
    the builder already puts in an empty feature, and it is a value XGBoost
    handles.  A ``0.0`` here would be the fabricated default this task removes.

    ``column`` is required and is named in the ``warning``: reading a non-number
    as ``NaN`` is a degradation, and the global constraint is that every
    degradation is visible at ``warning`` or above rather than silent.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Feature column %r holds %r, which is not a number — "
                       "read as NaN", column, value)
        return float("nan")


def _prediction_frame(feature_row, schema: list[str]) -> pd.DataFrame:
    """The single model input: one row, in the schema's exact order.

    Every path that predicts — cache hit, cache miss, warmup, batch miss —
    builds its frame here, so "the fast path and the slow path agree" is a
    property of there being one construction rather than a coincidence that has
    to be re-checked.

    ``feature_row`` is either a name-keyed ``pd.Series`` (the cache-hit path, and
    the warmup path through :func:`_warm_feature_row`) or the one-row frame
    ``build_inference_features`` computes (the miss path and the batch path, both
    of which used to call ``align_to_schema`` directly).  ``align_to_schema`` is
    the only reconciliation either way and it raises for a column the frame
    cannot produce; nothing pads, and nothing defaults to ``0.0`` (F7).
    """
    frame = feature_row if isinstance(feature_row, pd.DataFrame) \
        else pd.DataFrame([feature_row])
    return align_to_schema(frame, schema)


def _warm_feature_row(latest: pd.DataFrame, schema: list[str]) -> pd.Series:
    """A computed 1-row frame as a name-keyed row ready for the cache.

    The counterpart of :func:`_prediction_frame` on the write side — it goes
    *through* it, so the write side cannot drift from the read side — and the
    reason ``feature_row`` and ``schema`` in :func:`_cache_put` are always
    consistent: the row that is stored is the row that was aligned, so the
    stored names are exactly the model's columns, in the model's order.
    """
    return _prediction_frame(latest, schema).iloc[0]

# ── Daily Caches ──────────────────────────────────────────────────────────────
_cache_date = None
_usdidr_cache = None
_fred_cache = None
_bi_cache = None
#: ticker -> ``{"data": dict, "date": str, "available": bool, "reason": str|None}``.
#: The availability travels with the values, not beside them: the eight numbers
#: alone cannot say whether they are a measurement, which is the whole defect this
#: cache would otherwise re-introduce one request later.
_fundamentals_cache = {}


def get_fundamentals(ticker: str) -> tuple[dict, bool, Optional[str]]:
    """``ticker``'s fundamentals for this process, with their availability.

    Reads :func:`features.fetchers.fetch_fundamentals_status` rather than the
    value-only fetcher, and caches the *status* per ticker for the day.  A value
    cache alone would lose it: the eight numbers are identical whether Yahoo
    answered healthily or not at all (five of them are ``NaN`` either way), so a
    second request for the same ticker would report ``degraded: []`` for a fetch
    that failed — the same safe-default-indistinguishable-from-a-value defect,
    moved one request later.

    Returns:
        ``(values, available, reason)`` — the seam's own shape, so a caller
        cannot accidentally take the values without the status.
    """
    today = str(date.today())
    cached = _fundamentals_cache.get(ticker)
    if cached is not None and cached.get("date") == today:
        return cached["data"], cached["available"], cached["reason"]

    values, available, reason = fetch_fundamentals_status(ticker)
    _fundamentals_cache[ticker] = {
        "data": values, "date": today, "available": available, "reason": reason,
    }
    return values, available, reason


def refresh_daily_cache():
    """Fetches macro data once per day to ensure fast API responses."""
    global _cache_date, _usdidr_cache, _fred_cache, _bi_cache
    today = str(date.today())

    if _cache_date == today:
        return

    logger.info("Refreshing Daily Macro Cache...")
    start_date = (date.today() - timedelta(days=10)).strftime("%Y-%m-%d")
    end_date = today

    try:
        _usdidr_cache = fetch_usdidr(start_date, end_date)
        _fred_cache = fetch_fred_macro(start_date, end_date)
        _bi_cache = fetch_bi_rate(start_date, end_date)
        _cache_date = today
        logger.info("Daily Macro Cache Updated")
    except Exception as e:
        logger.warning("Cache refresh failed: %s", e)

def load_best_model(ticker: str) -> tuple[Any, list[str]]:
    """Load the model the index maps to ``ticker``, with its feature contract.

    Selection is delegated to :mod:`model_store`, which reads
    ``model_index.json`` and raises when the ticker has no entry.  The two
    substitutions this function used to perform — reusing a ``_fallback`` model,
    and scanning every folder for the first loadable one — are gone: an unmapped
    ticker is a 404, not another ticker's model (F2), and the backtester resolves
    through the same index, so the two can no longer disagree (F9).

    The model and its columns are cached together, so the columns the API aligns
    its frame to are the columns of the model it just loaded.

    Raises:
        HTTPException: ``404`` when the index maps no model to ``ticker`` (or the
            folder it names holds none); ``500`` when the model cannot be loaded
            or its feature contract cannot be read.
    """
    if ticker in model_cache:
        return model_cache[ticker]

    try:
        entry = load_model_for(ticker)
    except ModelNotFound as e:
        # A client asking for a ticker the index does not map is a client error,
        # so the client gets a bare 404 — but the server keeps a trace naming the
        # ticker.  On a populated index that is the *only* server-side record of
        # the request: startup warns about a wholly-empty index and about nothing
        # else, so an unmapped ticker used to vanish without a line anywhere.
        # Deliberately not ``error``: no server fault occurred.
        logger.warning("No model mapped to %s: %s", ticker, e)
        raise HTTPException(status_code=404, detail=f"No model found for {ticker}") from e
    except FeatureSchemaError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    model_cache[ticker] = entry
    return entry

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED FEATURE BUILDING FOR INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

# The inference-time universe snapshot.  Module-level names rather than literals
# inside load_universe_context(), so a test can point them at a tmp_path without
# touching the real data/ tree.
UNIVERSE_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "stocks.csv")
IHSG_PATH = os.path.join(BASE_DIR, "data", "raw", "ihsg.csv")

# How much history the cross-stock (breadth/sector) pass is given at inference.
# It only ever contributes the latest row, whose every column needs at most six
# rows of history, so 30 leaves a five-fold margin while keeping the pass ~30x
# cheaper than walking the full 1500-row universe per request.  Configurable as
# a module attribute so the equivalence test can compare it against the full
# history.
CROSS_STOCK_WINDOW_ROWS = 30

_universe_context = None


def load_universe_context():
    """Load the daily universe snapshot that inference-time context needs.

    The market-context and sector features the builder computes belong to a
    ticker *relative to the universe* (IHSG, its sector peers, breadth), so a
    single request's OHLCV frame is not enough to build them — this is where
    serving gets the rest.

    Returns:
        ``(raw_ohlcv, ihsg, all_ticker_data)``: the multi-ticker OHLCV frame in
        yfinance ``group_by="ticker"`` layout, the IHSG index frame, and
        ``{ticker: dropna(raw_ohlcv[ticker])}`` for every configured ticker.

    Any source that cannot be read degrades to an empty frame/dict with a
    ``warning``, so a missing file yields absent/NaN context columns instead of
    an exception.  The result is cached for the life of the process, so a
    request pays the disk read at most once.
    """
    global _universe_context
    if _universe_context is not None:
        return _universe_context

    raw_ohlcv = pd.DataFrame()
    try:
        raw_ohlcv = load_data(UNIVERSE_RAW_PATH)
    except Exception as e:
        logger.warning("Universe raw data unavailable at %s: %s", UNIVERSE_RAW_PATH, e)

    ihsg = pd.DataFrame()
    try:
        ihsg = load_ihsg_data(IHSG_PATH)
    except Exception as e:
        logger.warning("IHSG data unavailable at %s: %s", IHSG_PATH, e)

    # Same per-ticker extraction build_feature_set does, so the cross-stock
    # features are computed from identical inputs on both paths.
    all_ticker_data = {}
    for ticker in TICKERS:
        try:
            all_ticker_data[ticker] = dropna(raw_ohlcv[ticker].copy())
        except KeyError:
            pass

    _universe_context = (raw_ohlcv, ihsg, all_ticker_data)
    return _universe_context


def universe_age_days() -> int:
    """Age in whole days of the raw universe file the features are built from.

    Reads the *universe* file's mtime (``UNIVERSE_RAW_PATH``) — not IHSG's, and
    not the process's.  The universe frame is what the market-context and sector
    features are computed from, so it is the file whose staleness makes those
    features stale.

    Returns:
        Whole days since that file was last written, or ``-1`` when the file is
        absent.  ``-1`` means *unknown*, never *fresh*: an absent file has no
        age, and reporting 0 would claim it had just been ingested.

    Note that this stats the file on disk, while the features actually served
    come from the snapshot cached at load_universe_context().  After a
    mid-process ingest the two disagree: this reports age 0 while serving still
    uses the older cached frame, so the number is evidence about the file, not
    about the features in the response.
    """
    try:
        mtime = os.path.getmtime(UNIVERSE_RAW_PATH)
    except OSError as e:
        logger.warning("Universe file %s unavailable — staleness unknown: %s",
                       UNIVERSE_RAW_PATH, e)
        return -1
    return int((time.time() - mtime) // 86400)


def build_inference_features(df: pd.DataFrame, ticker: str,
                             fundamentals: dict) -> pd.DataFrame:
    """Build the features for one inference request with the training builder.

    Delegates to the same ``build_features`` the training path uses, fed from
    the daily caches instead of the pipeline's per-run fetches, so a feature is
    either a real computed value or genuinely absent — never a placeholder.

    Args:
        df: OHLCV DataFrame for a single ticker (250 days), single-level columns
            as returned by ``yfinance.download(ticker)``
        ticker: Ticker symbol
        fundamentals: Dict of fundamental metrics from fetch_fundamentals()

    Returns:
        Latest row as a 1-row DataFrame, or an empty DataFrame when the ticker
        has fewer than the builder's 50 usable rows (the API layer rejects that
        case before calling).
    """
    # Three feature families are deliberately NOT supplied here, each for its own
    # measured reason:
    #
    # * vp_* (19 columns) is LIVE in training — no vp_* column is all-NaN in the
    #   committed data/processed/features.parquet — but it needs 5-minute
    #   intraday downloads, which serving cannot reproduce.  Excluded from the
    #   contract rather than faked.
    # * google_trend (1 column) is LIVE in training as well (build_feature_set
    #   fetches real values, and the committed parquet carries 98 distinct ones).
    #   Serving cannot reproduce the fetch, so passing any constant would
    #   fabricate a model-consumed column; None is passed, which makes the
    #   builder materialise no column at all.  Excluded from the contract too.
    # * idn_* (10 columns) is already DEAD in training: fetch_idx_fundamentals
    #   (src/features/fetchers.py) is a no-network stub that returns NaN for
    #   every field, and all ten columns are all-NaN in the committed parquet.
    #   There is nothing to reproduce.
    #
    # Because absence is a standing decision rather than a runtime failure, no
    # family is *omitted* from the call below: each is passed an explicit value,
    # or an explicit ``None`` when there is nothing to pass.  ``build_features``
    # distinguishes the two — omitting the keyword means "the caller forgot this
    # family" and logs one ``warning``, while an explicit ``None`` means
    # "deliberately absent" and logs nothing (see the ``_NOT_SUPPLIED`` sentinel
    # in ``features.pipeline``).  Passing ``None`` is what keeps a healthy
    # request silent: warning on every request would teach operators to ignore
    # the warning that this task exists to make visible (Task 1.9).  The model's
    # feature contract (feature_schema.json, Task 1.4) is derived so that it
    # never expects a column serving cannot supply.
    raw_ohlcv, ihsg, all_ticker_data = load_universe_context()

    # build_features selects its ticker with df[ticker], i.e. it wants the
    # multi-ticker yfinance group_by="ticker" layout, while a serving request
    # carries one ticker with flat columns.  Widen the frame here instead of
    # loosening the builder's contract: one top-level entry for this ticker,
    # its fields underneath.
    multi = pd.concat({ticker: df}, axis=1)

    # Sentiment is fetched the way serving has always fetched it: flag-gated,
    # and 0.0 on any failure.  It is handed to the builder as a value rather
    # than written into the frame so the builder keeps ownership of the column.
    # This call site is deliberately the *value-only* view
    # (``fetch_news_sentiment``): the builder needs a float, and a neutral 0.0
    # for "unavailable" is consistent with training.  Visibility belongs to the
    # overlay below, which reads ``fetch_news_sentiment_status`` and reports the
    # failure in ``degraded``; a silent fallback here is not a hidden one,
    # because the overlay covers the same failure for the same request.
    news_sentiment = 0.0
    if FEATURE_FLAGS.get("news_sentiment", True):
        try:
            from features.fetchers import fetch_news_sentiment
            news_sentiment = fetch_news_sentiment(ticker)
        except Exception:
            news_sentiment = 0.0

    featured = build_features(
        multi, ticker,
        fundamentals=fundamentals,
        usdidr=_usdidr_cache,
        fred_macro=_fred_cache,
        bi_rate=_bi_cache,
        ihsg=ihsg,
        all_ticker_data=all_ticker_data,
        news_sentiment=news_sentiment,
        # None, not a constant: a value here would materialise a fabricated
        # google_trend column at inference while training carries real fetched
        # values (see the family rationale above).  Explicit None (not an omitted
        # keyword): the builder logs the *omitted* case at warning, and this
        # absence is deliberate.
        google_trend=None,
        # Explicit None for the same reason: deliberate absence, not a family
        # this caller forgot.
        vp_features=None,
        idx_fundamentals=None,
        mode="full",
        with_target=False,
    )

    if featured is None:
        logger.warning("Fewer than 50 usable rows for %s — no features built", ticker)
        return pd.DataFrame()

    # Cross-stock features (breadth, sector return/rank/volume) are applied by
    # build_feature_set after every ticker is engineered, because they need the
    # whole universe.  Serving has the whole universe from
    # load_universe_context(), so it applies them at the same point for the same
    # reason.
    if FEATURE_FLAGS.get("market_context", True) and all_ticker_data:
        # compute_cross_stock_features walks every index label against every
        # ticker, so on the real universe (1553 rows x 45 tickers) the full
        # history costs ~29 s of CPU.  Serving returns only the latest row, and
        # every cross-stock column for that row is a function of at most the last
        # six rows (a 5-day return needs six), so the same function is applied to
        # a bounded tail window that still contains the row being served.  The
        # served row is identical either way — pinned by
        # tests/test_serve_parity.py::test_cross_stock_window_matches_full_history
        # and measured on the real universe in task-1.3-report.md.  Without the
        # window the first request per ticker stalls the event loop for ~30 s,
        # because this function is called synchronously from the async handler.
        cutoff = featured.index[-CROSS_STOCK_WINDOW_ROWS:][0]
        featured = compute_cross_stock_features(
            featured.loc[cutoff:].copy(),
            ticker,
            {t: d.loc[cutoff:] for t, d in all_ticker_data.items()},
            TICKERS,
        )

    numeric_df = featured.select_dtypes(include=[np.number])
    return numeric_df.iloc[[-1]].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# API MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    ticker: str

class BatchPredictRequest(BaseModel):
    tickers: List[str]

class PredictResponse(BaseModel):
    ticker: str
    prediction: int
    probability_up: float
    signal: str
    sentiment_score: float = 0.0
    probability_adjusted: float = 0.0
    signal_adjusted: str = ""
    drift_warning: bool = False
    drift_score: float = 0.0
    # Every component that failed while this response was being computed, by
    # name — ``["sentiment"]`` when the news fetch was unavailable.  The fields
    # above are a *safe default*, which the project's convention requires, but a
    # safe default that is indistinguishable from a real value is how a dead
    # news API came to look like genuinely neutral news (F8).  Defaults to empty
    # so a healthy response says so explicitly.  ``default_factory`` rather than
    # a bare ``[]``: this must not depend on Pydantic deep-copying a mutable
    # default.
    degraded: list[str] = Field(default_factory=list)

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

# ── Sentiment overlay strength ────────────────────────────────────────────────
SENTIMENT_WEIGHT = 0.10
PSI_THRESHOLD = 0.25  # Population Stability Index threshold for drift warning

# ── PSI (Population Stability Index) — distribution drift check ─────────────
def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between expected (training) and actual (live) distributions.

    PSI > 0.25 indicates significant drift — predictions may be unreliable.
    """
    all_vals = np.concatenate([expected, actual])
    bin_edges = np.percentile(all_vals, np.linspace(0, 100, bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_hist, _ = np.histogram(expected, bins=bin_edges)
    act_hist, _ = np.histogram(actual, bins=bin_edges)

    exp_hist = (exp_hist / exp_hist.sum()).clip(0.001)
    act_hist = (act_hist / act_hist.sum()).clip(0.001)

    psi = np.sum((act_hist - exp_hist) * np.log(act_hist / exp_hist))
    return float(psi)

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Indonesian Stock Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/tickers")
def list_tickers():
    return {"tickers": TICKERS, "count": len(TICKERS)}

@app.get("/health")
def health():
    return {"status": "ok"}

def _apply_sentiment_overlay(ticker: str, prob: float, background_tasks=None) -> dict:
    """Adjust model probability with news sentiment overlay.

    Fetches latest sentiment and tweaks probability by up to SENTIMENT_WEIGHT.
    Falls back to 0.0 sentiment when the news source is unavailable (no
    blocking) — but reports that fallback, because ``sentiment_score == 0.0`` is
    also the honest reading of "the headlines were neutral".  An unavailable
    source is therefore named in ``degraded`` *and* logged at ``warning`` with
    the reason, so an operator can tell a dead news API from a quiet news day
    (F8).

    The overlay reads ``fetch_news_sentiment_status``, not
    ``fetch_news_sentiment``: the latter returns a bare float, and every real
    failure mode of the news path — no ``NEWSAPI_KEY``, a dead/401 endpoint, a
    429, no ``vaderSentiment`` — returns 0.0 from it without raising, so an
    ``except`` around it could never fire.  Detecting a failure by catching an
    exception was the defect this function exists to remove.

    Returns dict with sentiment_score, probability_adjusted, signal_adjusted,
    degraded.
    """
    sentiment = 0.0
    degraded: list[str] = []

    try:
        from features.fetchers import fetch_news_sentiment_status
        score, available, reason = fetch_news_sentiment_status(ticker)
    except Exception as e:
        # Belt and braces.  The status companion swallows its own failures, so
        # this fires only if the import or the seam itself breaks — and even
        # then it is visible, never a bare ``pass`` (F8).
        score, available, reason = 0.0, False, f"{type(e).__name__}: {e}"

    if available:
        sentiment = max(-1.0, min(1.0, float(score)))
    else:
        # Never block a prediction on sentiment — but never hide the failure
        # either.  ``sentiment`` stays at the 0.0 initialised above, which is a
        # safe default, and the response now says that it is one.
        degraded.append("sentiment")
        # The reason comes *before* the non-ASCII dash on purpose: on a console
        # whose encoding cannot represent that character the tail of the line can
        # render as replacement characters, and the reason is the part an
        # operator has to be able to read.
        logger.warning(
            "Sentiment unavailable (%s) for %s — overlay skipped, response "
            "marked degraded", reason, ticker,
        )

    adjustment = sentiment * SENTIMENT_WEIGHT
    prob_adj = prob + adjustment
    prob_adj = max(0.0, min(1.0, prob_adj))
    signal_adj = "BUY" if prob_adj >= 0.50 else "SELL"

    return {
        "sentiment_score": round(sentiment, 4),
        "probability_adjusted": round(prob_adj, 4),
        "signal_adjusted": signal_adj,
        "degraded": degraded,
    }


def _mark_fundamentals(degraded: list[str], ticker: str, available: bool,
                       reason: Optional[str]) -> list[str]:
    """Name a fundamentals outage in ``degraded``, and log why.

    Five of the eight fundamental fields are required by every model's persisted
    contract (``pe_ratio``, ``dividend_yield``, ``market_cap``,
    ``debt_to_equity``, ``return_on_equity``), and an unavailable ``.info``
    materialises each as a constant-NaN column that ``align_to_schema`` accepts —
    it checks *presence*, and a column of ``NaN`` is present.  The prediction is
    therefore still returned, exactly as the sentiment overlay's is, and this
    function is the only thing that makes it distinguishable from a healthy one
    (Task 2.4).

    ``available`` is read from the status seam, never inferred from the values:
    an outage and a healthy fetch of a ticker whose fields Yahoo does not publish
    produce the same eight NaNs, so there is nothing in the numbers to detect.

    Returns:
        ``degraded`` with ``"fundamentals"`` appended when the source was
        unavailable, otherwise ``degraded`` unchanged.
    """
    if available:
        return degraded
    degraded.append("fundamentals")
    logger.warning(
        "Fundamentals unavailable (%s) for %s — five schema-required inputs are "
        "constant NaN in this prediction, response marked degraded",
        reason or "reason not reported", ticker,
    )
    return degraded

@app.get("/cache")
def cache_status():
    """Show prediction cache status."""
    conn = _get_cache_conn()
    today = str(date.today())
    # Today's *tickers* for the window serving currently uses (R1): one ticker
    # holds one row per (window, schema fingerprint), so a bare COUNT over the
    # table reports a coverage figure no single request path can achieve — after
    # a mid-day retrain that changes a model's columns, the same ticker is two
    # rows and ``coverage`` would read ``90/45 (200%)``.  The figure is consumed
    # as "how many of the 45 tickers" (see
    # monitoring/indonesian_stock_dashboard_v3.json), so it counts distinct
    # tickers, while ``total_cached`` below stays the honest row count of what is
    # actually stored.
    row = conn.execute(
        "SELECT COUNT(DISTINCT ticker) FROM cache WHERE date = ? AND window = ?",
        (today, CACHE_WINDOW_DEFAULT),
    ).fetchone()
    cached_count = row[0] if row else 0
    total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    conn.close()
    return {
        "cached_today": cached_count,
        "total_cached": total,
        "total_tickers": len(TICKERS),
        "coverage": f"{cached_count}/{len(TICKERS)} ({cached_count*100//len(TICKERS)}%)",
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """Predict BUY/SELL signal for a single ticker.

    Checks daily prediction cache first — cached path returns in <0.2s.
    On cache miss, downloads data + computes features (~5-15s).
    """
    try:
        global _fundamentals_cache

        # 1. Trigger background cache refresh
        background_tasks.add_task(refresh_daily_cache)

        # ── Fast path: check prediction cache ──────────────────────────
        # The key is (ticker, window, schema fingerprint), so `trained_cols` has
        # to be in hand first: a row cached for another model's column set is a
        # miss, not a vector to re-zip (F6).
        model, trained_cols = load_best_model(request.ticker)
        cached_row = _cache_get(request.ticker, trained_cols, CACHE_WINDOW_DEFAULT)

        if cached_row is not None:
            logger.debug("Cache HIT for %s — sub-200ms prediction", request.ticker)
            # The status is resolved on this path too, and *here* rather than
            # before the branch: the hit returns without ever building features,
            # so this is the only place its response can learn whether the
            # fundamentals behind the cached row were a measurement.  The row
            # cannot say — five of the eight fields are NaN for an outage and for
            # a field Yahoo does not publish alike — so the status is read, never
            # inferred (Task 2.4).  The values are not needed: the row holds them.
            #
            # Resolving it here instead of above the branch is also what keeps a
            # *miss* from paying for two ``.info`` calls: the slow path below
            # calls ``get_fundamentals`` once and uses both halves of the result.
            _values, fundamentals_available, fundamentals_reason = get_fundamentals(
                request.ticker
            )
            # The stored row is keyed by column name and was written for exactly
            # this schema, so the only reconciliation is align_to_schema — the
            # same call the slow path makes.  Nothing here fills a shortfall:
            # the `0.0` default that stood here was one half of F7, and it made
            # a hit disagree with the miss that produced the row.
            latest_df = _prediction_frame(cached_row, trained_cols)

            prediction = int(model.predict(latest_df)[0])
            probability = float(model.predict_proba(latest_df)[0][1])
            signal = "BUY" if prediction == 1 else "SELL"
            overlay = _apply_sentiment_overlay(request.ticker, probability)
            PREDICTION_COUNTER.labels(ticker=request.ticker, signal=signal).inc()
            CONFIDENCE_HISTOGRAM.labels(ticker=request.ticker).observe(probability)
            # ``drift_warning``/``drift_score`` are hardcoded on every response
            # path and ``_compute_psi`` is never called: PSI needs a reference
            # distribution, which nothing here stores (R5).  The fields are kept
            # for compatibility and the README now says plainly that they are not
            # computed; wiring them is a separate, tested change.
            return PredictResponse(
                ticker=request.ticker, prediction=prediction,
                probability_up=round(probability, 4), signal=signal,
                sentiment_score=overlay["sentiment_score"],
                probability_adjusted=overlay["probability_adjusted"],
                signal_adjusted=overlay["signal_adjusted"],
                drift_warning=False, drift_score=0.0,
                degraded=_mark_fundamentals(
                    list(overlay["degraded"]), request.ticker,
                    fundamentals_available, fundamentals_reason,
                ),
            )

        # ── Slow path: full pipeline (download + features + cache) ──────
        logger.info("Cache MISS for %s — running full pipeline", request.ticker)

        # 2. Async OHLCV download
        df = await asyncio.to_thread(
            yf.download, request.ticker, period="250d", auto_adjust=True, progress=False
        )
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data on Yahoo Finance for {request.ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = dropna(df)

        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Not enough trading days for TA calculation")

        # 3. Fundamentals (async + cached daily, with their availability)
        fundamentals, fundamentals_available, fundamentals_reason = \
            await asyncio.to_thread(get_fundamentals, request.ticker)

        # 4. Build features using shared functions
        latest = build_inference_features(df, request.ticker, fundamentals)

        # 5. Build the model input with the single frame builder.  It is
        #    ``align_to_schema`` for this computed frame, so behaviour is
        #    unchanged, but the miss path no longer builds its own input: the
        #    frame this request predicts on is the frame the builder produces for
        #    every other path, hit and batch included.  ``align_to_schema`` raises
        #    FeatureSchemaError for any column the model expects and this frame
        #    cannot produce; the NaN padding that used to stand here silently
        #    converted a missing feature into a value the model never saw.
        latest_df = _prediction_frame(latest, trained_cols)
        latest_row = latest_df.iloc[0]

        # 6. Cache the feature row for the next request (fast path) — keyed by
        #    the model's schema and the window, so the hit path can only read
        #    back values intended for exactly these columns (F6).  The row
        #    cached is the aligned one, so what the hit path rebuilds is what
        #    this request predicted on: the two paths converge here (F7).
        try:
            _cache_put(request.ticker, latest_row, trained_cols, CACHE_WINDOW_DEFAULT)
            logger.debug("Cached features for %s (%d cols)", request.ticker, len(trained_cols))
        except Exception as e:
            # Best-effort: a prediction without a cache entry is slower, not
            # wrong.  Reported at warning, because a cache that never fills is a
            # degradation that would otherwise be invisible.
            logger.warning("Failed to cache features for %s: %s", request.ticker, e)

        prediction = int(model.predict(latest_df)[0])
        probability = float(model.predict_proba(latest_df)[0][1])

        signal = "BUY" if prediction == 1 else "SELL"
        overlay = _apply_sentiment_overlay(request.ticker, probability)
        PREDICTION_COUNTER.labels(ticker=request.ticker, signal=signal).inc()
        CONFIDENCE_HISTOGRAM.labels(ticker=request.ticker).observe(probability)

        return PredictResponse(
            ticker=request.ticker,
            prediction=prediction,
            probability_up=round(probability, 4),
            signal=signal,
            sentiment_score=overlay["sentiment_score"],
            probability_adjusted=overlay["probability_adjusted"],
            signal_adjusted=overlay["signal_adjusted"],
            drift_warning=False, drift_score=0.0,
            degraded=_mark_fundamentals(
                list(overlay["degraded"]), request.ticker,
                fundamentals_available, fundamentals_reason,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Crash on %s: %s", request.ticker, e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest, background_tasks: BackgroundTasks):
    """Predict BUY/SELL signals for multiple tickers in one request.

    Fetches OHLCV data in parallel for all tickers.
    """
    if not request.tickers:
        raise HTTPException(status_code=400, detail="tickers list cannot be empty")

    # Trigger cache refresh
    background_tasks.add_task(refresh_daily_cache)

    async def predict_single(ticker: str) -> PredictResponse:
        """Predict for one ticker (used with asyncio.gather)."""
        try:
            df = await asyncio.to_thread(
                yf.download, ticker, period="250d", auto_adjust=True, progress=False
            )
            if df.empty:
                raise ValueError(f"No Yahoo Finance data for {ticker}")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df = dropna(df)
            if len(df) < 50:
                raise ValueError(f"Not enough data for {ticker}")

            fundamentals, fundamentals_available, fundamentals_reason = \
                await asyncio.to_thread(get_fundamentals, ticker)

            latest = build_inference_features(df, ticker, fundamentals)
            model, trained_cols = load_best_model(ticker)

            # Same contract, same failure mode, and the same builder as the
            # single-ticker path: a model whose columns this frame cannot satisfy
            # is an error, never a NaN-padded prediction.
            latest_df = _prediction_frame(latest, trained_cols)
            prediction = int(model.predict(latest_df)[0])
            probability = float(model.predict_proba(latest_df)[0][1])
            signal = "BUY" if prediction == 1 else "SELL"
            overlay = _apply_sentiment_overlay(ticker, probability)

            PREDICTION_COUNTER.labels(ticker=ticker, signal=signal).inc()
            CONFIDENCE_HISTOGRAM.labels(ticker=ticker).observe(probability)

            return PredictResponse(
                ticker=ticker,
                prediction=prediction,
                probability_up=round(probability, 4),
                signal=signal,
                sentiment_score=overlay["sentiment_score"],
                probability_adjusted=overlay["probability_adjusted"],
                signal_adjusted=overlay["signal_adjusted"],
                drift_warning=False, drift_score=0.0,
                degraded=_mark_fundamentals(
                    list(overlay["degraded"]), ticker,
                    fundamentals_available, fundamentals_reason,
                ),
            )
        except HTTPException:
            raise
        except Exception as e:
            # F8: this used to return a *signal-shaped* object — prediction 0
            # (= SELL), confidence 0.0, and the error text stuffed into the
            # ``signal`` field — under HTTP 200, so a caller reading
            # ``prediction`` traded on a failure.  A failure is now an error the
            # caller cannot mistake for a prediction, and the ticker is in the
            # body so one bad symbol in a batch is identifiable without logs.
            logger.warning("Batch predict failed for %s: %s", ticker, e)
            raise HTTPException(
                status_code=502,
                detail=f"Prediction failed for {ticker}: {e}",
            ) from e

    tasks = [predict_single(t) for t in request.tickers]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    return BatchPredictResponse(predictions=list(results))

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)