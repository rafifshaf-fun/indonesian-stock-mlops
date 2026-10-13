"""Task 1.8 — one prediction cache path, aligned by name (F6, F7).

Two audit findings meet in the prediction cache.

**F6 — a positional zip across model versions.**  The cache stored a bare
``numpy`` vector under a key of ``(ticker, date)`` and the request path rebuilt a
feature frame by *zipping that vector positionally* onto the current model's
column order, defaulting any shortfall to ``0.0``.  A vector produced by a
different model — a retrain, or another ticker's model — was therefore
*silently permuted* across features, and the resulting probability is a number
with no error attached to it.  The cache key now carries a fingerprint of the
schema, so values written for a different column set are a **miss**, and the
only reconciliation left is ``features.schema.align_to_schema``.

**F7 — the two paths disagreed on the same day.**  The cache-hit path filled a
column the model wanted but the vector lacked with ``0.0``, while the cache-miss
path filled it with ``NaN``.  The same ticker on the same day therefore had two
different answers (measured 0.187 against 0.303, straddling the 0.50 threshold).
No default of either kind survives: the cache is keyed and fingerprinted, so a
shortfall cannot arise, and ``align_to_schema`` raises rather than padding.

These tests are offline and never touch the real ``data/`` tree: each one points
``serve.PRED_CACHE_DB`` at a ``tmp_path`` file before the first connection.

``serve`` imports mlflow → opentelemetry, whose runtime-context loader still
calls the removed ``importlib.metadata`` dict interface on Python 3.11, and
FastAPI raises its own deprecation for ``@app.on_event("startup")``.  Both
messages are upstream of this repository, so the two filters below are scoped to
their exact text and to this module only, exactly as ``tests/test_serve_parity.py``
does; every other warning still escalates under ``filterwarnings = ["error"]``.
"""
import json
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:SelectableGroups dict interface is deprecated:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:(?s).*on_event is deprecated.*:DeprecationWarning"
    ),
]

SCHEMA = ["volatility_20d", "rsi"]


@pytest.fixture(autouse=True)
def cache_db(tmp_path, monkeypatch):
    """Point the cache at a per-test database, and hand back its path.

    Autouse so that no test in this module can reach the developer's real
    ``data/prediction_cache.db`` — a test that forgot to redirect it would
    otherwise read and write live cache state.
    """
    import serve

    path = tmp_path / "prediction_cache.db"
    monkeypatch.setattr(serve, "PRED_CACHE_DB", str(path))
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# THE BRIEF'S THREE TESTS — signature, window fingerprint, schema fingerprint
# ═══════════════════════════════════════════════════════════════════════════════

def test_fast_and_slow_path_agree(cache_db):
    import serve

    schema = ["volatility_20d", "rsi"]
    row = pd.Series({"volatility_20d": 9.0, "rsi": 0.1})
    serve._cache_put("BBCA.JK", row, schema, "0900")
    got = serve._cache_get("BBCA.JK", schema, "0900")
    assert got["volatility_20d"] == 9.0 and got["rsi"] == 0.1


def test_window_change_is_a_miss(cache_db):
    import serve

    serve._cache_put("BBCA.JK", pd.Series({"rsi": 1.0}), ["rsi"], "0900")
    assert serve._cache_get("BBCA.JK", ["rsi"], "1330") is None


def test_schema_change_is_a_miss(cache_db):
    import serve

    serve._cache_put("BBCA.JK", pd.Series({"rsi": 1.0}), ["rsi"], "0900")
    assert serve._cache_get("BBCA.JK", ["rsi", "volatility_20d"], "0900") is None


def test_schema_order_is_part_of_the_fingerprint(cache_db):
    """A reordered schema is a miss, not a permuted hit.

    ``align_to_schema``'s docstring: order is part of the contract, because
    XGBoost is fitted on a positionally-scaled matrix.  Two schemas carrying the
    same names in a different order are therefore *different* inputs, and the
    fingerprint must separate them.
    """
    import serve

    serve._cache_put("BBCA.JK", pd.Series({"rsi": 1.0, "volatility_20d": 2.0}),
                     SCHEMA, "0900")
    assert serve._cache_get("BBCA.JK", SCHEMA, "0900") is not None
    assert serve._cache_get("BBCA.JK", list(reversed(SCHEMA)), "0900") is None


def test_another_tickers_values_are_not_reachable(cache_db):
    """F6's concrete harm: one ticker's vector answered for another's model."""
    import serve

    serve._cache_put("BBCA.JK", pd.Series({"rsi": 1.0}), ["rsi"], "0900")
    assert serve._cache_get("BBRI.JK", ["rsi"], "0900") is None


def test_values_are_keyed_by_name_not_position(cache_db):
    """The row comes back keyed by name, so no caller can zip it positionally."""
    import serve

    serve._cache_put("BBCA.JK", pd.Series({"volatility_20d": 9.0, "rsi": 0.1}),
                     SCHEMA, "0900")
    got = serve._cache_get("BBCA.JK", SCHEMA, "0900")
    assert list(got.index) == SCHEMA
    assert got["volatility_20d"] == 9.0
    assert got["rsi"] == 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# R4 — the store is JSON, not pickle
# ═══════════════════════════════════════════════════════════════════════════════

def test_stored_bytes_are_json_keyed_by_column_name(cache_db):
    import serve

    serve._cache_put("BBCA.JK", pd.Series({"volatility_20d": 9.0, "rsi": 0.1}),
                     SCHEMA, "0900")

    conn = sqlite3.connect(str(cache_db))
    try:
        stored = conn.execute("SELECT features FROM cache").fetchone()[0]
    finally:
        conn.close()

    # A text column, and text that any consumer can read without executing it.
    assert isinstance(stored, str), "features are not stored as text"
    assert sorted(json.loads(stored)) == sorted(SCHEMA), (
        "the stored row is not keyed by column name"
    )
    assert not stored.startswith("gAS"), "the stored row is a pickle"


def test_nan_round_trips_as_nan(cache_db):
    """``NaN`` is a real feature value, not an error, and must survive the cache.

    XGBoost handles ``NaN`` at inference, and a served row legitimately carries
    it for the eight columns ``tests/test_serve_parity.py`` pins as pre-existing,
    so the cache must not turn one into a fabricated number — or into a miss.
    """
    import serve

    serve._cache_put("BBCA.JK", pd.Series({"rsi": 1.0, "volatility_20d": np.nan}),
                     SCHEMA, "0900")
    got = serve._cache_get("BBCA.JK", SCHEMA, "0900")
    assert got is not None, "a row carrying NaN was treated as absent"
    assert got["rsi"] == 1.0
    assert np.isnan(got["volatility_20d"])


# ═══════════════════════════════════════════════════════════════════════════════
# R5 — an old-shaped database is migrated, and a fresh one still works
# ═══════════════════════════════════════════════════════════════════════════════

def test_fresh_database_is_created(cache_db):
    """No file at all: connecting creates the new shape."""
    import serve

    assert not cache_db.exists()
    serve._cache_get("BBCA.JK", SCHEMA, "0900")
    assert cache_db.exists()
    # Ordered, not a set: ``INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)`` is
    # positional, so a table carrying the right *names* in another order is a
    # different table (fix round 1, Minor 3).
    assert serve._cache_columns(cache_db) == list(serve.CACHE_COLUMNS)
    assert serve._cache_primary_key(serve._cache_table_info(cache_db)) == \
        list(serve.CACHE_PRIMARY_KEY)


def test_pre_task_1_8_database_is_recreated_not_left_unqueryable(cache_db):
    """R5: ``CREATE TABLE IF NOT EXISTS`` cannot change an existing table.

    The previous version's ``cache`` table is ``(ticker, date, features BLOB,
    ohlcv BLOB)``.  Left in place, every query against the new columns fails, so
    the table is dropped and recreated on connect.  The contents are a
    regenerable cache, so dropping costs one slow request per ticker.
    """
    import serve

    conn = sqlite3.connect(str(cache_db))
    try:
        conn.execute("CREATE TABLE cache (ticker TEXT PRIMARY KEY, date TEXT, "
                     "features BLOB, ohlcv BLOB)")
        conn.execute("INSERT INTO cache VALUES (?, ?, ?, ?)",
                     ("BBCA.JK", "2020-01-01", b"\x80\x04gAS", b"\x80\x04N."))
        conn.commit()
    finally:
        conn.close()

    # The migration happens on the next connect, without raising.
    assert serve._cache_get("BBCA.JK", SCHEMA, "0900") is None
    assert serve._cache_columns(cache_db) == list(serve.CACHE_COLUMNS)
    assert serve._cache_primary_key(serve._cache_table_info(cache_db)) == \
        list(serve.CACHE_PRIMARY_KEY)

    # ...and the recreated table is fully usable in both directions.
    serve._cache_put("BBCA.JK", pd.Series({"volatility_20d": 9.0, "rsi": 0.1}),
                     SCHEMA, "0900")
    got = serve._cache_get("BBCA.JK", SCHEMA, "0900")
    assert got is not None and got["volatility_20d"] == 9.0


# ═══════════════════════════════════════════════════════════════════════════════
# F7 — the two paths agree: one row, one prediction matrix
# ═══════════════════════════════════════════════════════════════════════════════

def test_hit_path_frame_equals_miss_path_frame():
    """F7's measured defect (0.187 against 0.303) cannot recur.

    Every path — cache hit, cache miss, warmup, batch miss — hands a row and a
    schema to ``_prediction_frame``, and nothing else builds a model input.  A
    hit frame that differed from a miss frame for the same values would be the
    two-answers defect returning through a different door, so the two are pinned
    equal here, column order and values included.
    """
    import serve

    values = pd.Series({"volatility_20d": 9.0, "rsi": 0.1})
    hit = serve._prediction_frame(values, SCHEMA)

    # The miss path's row, after ``align_to_schema`` has put it in schema order.
    miss_row = pd.Series({"rsi": 0.1, "volatility_20d": 9.0})[SCHEMA]
    miss = serve._prediction_frame(miss_row, SCHEMA)

    pd.testing.assert_frame_equal(hit, miss)
    assert list(hit.columns) == SCHEMA


def test_cached_row_produces_the_same_matrix_as_the_row_that_was_served(cache_db):
    """The stored row, read back, is the frame the miss path predicted on.

    This is the F7 assertion end to end through the store: the values written on
    a miss and the values read on the following hit build byte-identical model
    inputs, so the fast path cannot answer differently from the slow one.
    """
    import serve

    served_row = pd.Series({"volatility_20d": 9.0, "rsi": 0.1})

    miss_frame = serve._prediction_frame(served_row, SCHEMA)
    serve._cache_put("BBCA.JK", served_row, SCHEMA, serve.CACHE_WINDOW_DEFAULT)
    hit_frame = serve._prediction_frame(
        serve._cache_get("BBCA.JK", SCHEMA, serve.CACHE_WINDOW_DEFAULT), SCHEMA
    )

    pd.testing.assert_frame_equal(hit_frame, miss_frame)


def test_prediction_frame_carries_no_default_for_a_missing_column():
    """No ``0.0`` — the default was one half of the two-paths-disagree defect.

    ``align_to_schema`` raises for a column the frame cannot produce; a
    ``NaN``/``0.0`` pad here would convert a missing feature into a value the
    model never saw.
    """
    import serve
    from features.schema import FeatureSchemaError

    short = pd.Series({"rsi": 0.1})
    with pytest.raises(FeatureSchemaError):
        serve._prediction_frame(short, SCHEMA)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX ROUND 1, IMPORTANT 1 — the date predicate has a test
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_row_from_another_day_is_a_miss(cache_db):
    """``AND date = ?`` is the only barrier to serving yesterday's features.

    The window and the schema are *redundantly* encoded — each is part of the
    primary key, so a mismatch cannot even reach the predicate.  The date is not:
    the key is ``(ticker, window, schema_hash)``, one key holds exactly one row,
    and yesterday's row therefore sits under today's key until the day's first
    write replaces it.  Nothing but the date predicate in ``_cache_get``
    separates the two, and without it a deployment that does not retrain would
    serve the previous session's features as today's, silently.

    The row below is written through raw ``sqlite3`` with the *correct*
    fingerprint for the key, so no other guard can answer for the predicate.
    """
    import serve

    stored = json.dumps({"volatility_20d": 9.0, "rsi": 0.1})
    conn = serve._get_cache_conn()   # create the table, then drive it raw
    conn.close()
    conn = sqlite3.connect(str(cache_db))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cache "
            "(ticker, window, schema_hash, date, features) VALUES (?, ?, ?, ?, ?)",
            ("BBCA.JK", serve.CACHE_WINDOW_DEFAULT,
             serve._cache_schema_hash(SCHEMA), "2020-01-01", stored),
        )
        conn.commit()

        # Direction 1: an old date under a correct, matching key is a miss.  The
        # only thing that changes below is the date.
        assert serve._cache_get("BBCA.JK", SCHEMA, serve.CACHE_WINDOW_DEFAULT) is None, (
            "a row dated 2020-01-01 was served as today's features"
        )

        # Direction 2: the identical row, dated today, is served — so the miss
        # above is the date predicate and not a fingerprint or shape rejection.
        conn.execute("UPDATE cache SET date = ?", (str(date.today()),))
        conn.commit()
    finally:
        conn.close()

    same_day = serve._cache_get("BBCA.JK", SCHEMA, serve.CACHE_WINDOW_DEFAULT)
    assert same_day is not None, "the same row dated today was not a hit"
    assert same_day["volatility_20d"] == 9.0
    assert same_day["rsi"] == 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# FIX ROUND 1, IMPORTANT 2 — every path, including miss and batch, uses the builder
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_computed_frame_is_built_by_the_single_builder():
    """The miss and batch paths' frame is the deleted expression, exactly.

    Those two paths call ``align_to_schema(latest, trained_cols)`` on the
    one-row frame ``build_inference_features`` returns; they now call
    ``_prediction_frame(latest, trained_cols)``, which is that same call for a
    frame input.  Pinned here so the convergence cannot quietly become a second
    construction with a different column set.
    """
    import serve
    from features.schema import align_to_schema

    computed = pd.DataFrame([{
        "volatility_20d": 9.0, "rsi": 0.1, "Close": 1234.5, "Volume": 10,
    }])

    expected = align_to_schema(computed, SCHEMA)      # what the miss path used to call
    got = serve._prediction_frame(computed, SCHEMA)

    pd.testing.assert_frame_equal(got, expected)
    assert list(got.columns) == SCHEMA                 # intermediates dropped, order kept
    assert got.iloc[0]["rsi"] == 0.1
    assert len(got) == 1


def test_only_the_builder_aligns_a_frame_to_a_model_schema():
    """The docstring's "every path builds its frame here" is load-bearing.

    ``_prediction_frame``'s docstring asserts that no other path builds a model
    input.  A future edit that re-introduces a direct ``align_to_schema`` call
    on a predicting path would make that claim false while leaving every other
    test green, so the claim itself is pinned: the module holds exactly one
    ``align_to_schema(...)`` call site and it is inside the builder.
    """
    import inspect
    import serve

    source = inspect.getsource(serve)
    assert source.count("align_to_schema(") == 1, (
        "a model input is built outside _prediction_frame: "
        f"{[l.strip() for l in source.splitlines() if 'align_to_schema(' in l]}"
    )
    assert "align_to_schema(" in inspect.getsource(serve._prediction_frame)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX ROUND 1, MINOR 3 — the migration compares order and the primary key
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_reordered_cache_table_is_migrated_not_accepted(cache_db):
    """Same five names, different order, is a different table.

    ``INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?)`` binds positionally,
    so a table whose columns are merely *present* would take the window as a
    ticker and the fingerprint as a window.  Comparing name sets (the pre-fix
    behaviour) accepts exactly that table.
    """
    import serve

    conn = sqlite3.connect(str(cache_db))
    try:
        conn.execute(
            "CREATE TABLE cache (features TEXT, date TEXT, schema_hash TEXT, "
            "window TEXT, ticker TEXT, PRIMARY KEY (ticker, window, schema_hash))"
        )
        conn.execute("INSERT INTO cache VALUES (?, ?, ?, ?, ?)",
                     ("marker", "2020-01-01", "stale", serve.CACHE_WINDOW_DEFAULT,
                      "BBCA.JK"))
        conn.commit()
    finally:
        conn.close()

    serve._cache_put("BBCA.JK", pd.Series({"rsi": 1.0}), ["rsi"],
                     serve.CACHE_WINDOW_DEFAULT)

    assert serve._cache_columns(cache_db) == list(serve.CACHE_COLUMNS)

    # The reordered table's row is gone (it was dropped, not reinterpreted) and
    # the new row landed under the columns it names.
    conn = sqlite3.connect(str(cache_db))
    try:
        rows = conn.execute(
            "SELECT ticker, window, schema_hash, date, features FROM cache"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1, rows
    ticker, window, schema_hash, dated, features = rows[0]
    assert ticker == "BBCA.JK"
    assert window == serve.CACHE_WINDOW_DEFAULT
    assert schema_hash == serve._cache_schema_hash(["rsi"])
    assert dated == str(date.today())
    assert json.loads(features) == {"rsi": 1.0}


def test_a_cache_table_without_the_primary_key_is_migrated(cache_db):
    """Right names, right order, no key — still not the table this code writes.

    ``INSERT OR REPLACE`` replaces *by primary key*.  Without it the same key is
    written as a second row, ``_cache_get``'s ``fetchone()`` becomes arbitrary,
    and "one key holds only today's row" stops being true.
    """
    import serve

    conn = sqlite3.connect(str(cache_db))
    try:
        conn.execute("CREATE TABLE cache (ticker TEXT, window TEXT, "
                     "schema_hash TEXT, date TEXT, features TEXT)")
        conn.execute("INSERT INTO cache VALUES (?, ?, ?, ?, ?)",
                     ("BBCA.JK", serve.CACHE_WINDOW_DEFAULT, "stale", "2020-01-01",
                      "{}"))
        conn.commit()
    finally:
        conn.close()

    serve._cache_put("BBCA.JK", pd.Series({"rsi": 1.0}), ["rsi"],
                     serve.CACHE_WINDOW_DEFAULT)

    assert serve._cache_primary_key(serve._cache_table_info(cache_db)) == \
        list(serve.CACHE_PRIMARY_KEY)
    conn = sqlite3.connect(str(cache_db))
    try:
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    finally:
        conn.close()
    assert count == 1, "the unkeyed table was not replaced"


# ═══════════════════════════════════════════════════════════════════════════════
# FIX ROUND 1, MINOR 4 — a non-numeric stored value is a miss, never a 500
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_non_numeric_stored_value_is_a_miss_not_an_exception(cache_db, caplog):
    """Only an external writer can produce one — and this store is going shared.

    ``pd.Series(..., dtype=float)`` raises on a parseable non-number, which
    would surface as a 500 rather than the warning-and-miss ``_cache_get``
    documents.  The column is named so the corruption is actionable.
    """
    import serve

    conn = serve._get_cache_conn()
    conn.close()
    conn = sqlite3.connect(str(cache_db))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cache "
            "(ticker, window, schema_hash, date, features) VALUES (?, ?, ?, ?, ?)",
            ("BBCA.JK", serve.CACHE_WINDOW_DEFAULT, serve._cache_schema_hash(SCHEMA),
             str(date.today()), json.dumps({"volatility_20d": 9.0, "rsi": "abc"})),
        )
        conn.commit()
    finally:
        conn.close()

    with caplog.at_level("WARNING"):
        got = serve._cache_get("BBCA.JK", SCHEMA, serve.CACHE_WINDOW_DEFAULT)

    assert got is None, "a non-numeric stored value was served instead of missing"
    assert any("rsi" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


# ═══════════════════════════════════════════════════════════════════════════════
# FIX ROUND 1, MINOR 5 — the write path releases its connection
# ═══════════════════════════════════════════════════════════════════════════════

def test_cache_put_closes_every_connection_it_opens(cache_db, monkeypatch):
    """``with conn:`` commits; it does not close (``__exit__`` ends a transaction).

    ``_cache_get`` and ``_cache_columns`` close explicitly; the write path did
    not, so every put leaked a handle for the process's lifetime.
    """
    import serve

    opened, closed = [], []

    class _Tracked(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

        def close(self):
            closed.append(self)
            super().close()

    real_connect = sqlite3.connect

    def tracked_connect(path, *args, **kwargs):
        kwargs.setdefault("factory", _Tracked)
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(serve.sqlite3, "connect", tracked_connect)

    serve._cache_put("BBCA.JK", pd.Series({"rsi": 1.0}), ["rsi"], "0900")

    assert opened, "the write path opened no connection at all"
    assert len(closed) == len(opened), (
        f"{len(opened)} connections opened, {len(closed)} closed"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIX ROUND 1, MINOR 6 — a degraded value names its column
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_non_numeric_feature_is_reported_with_its_column(cache_db, caplog):
    """``_as_float`` reads a non-number as ``NaN``; that must be visible.

    The constraint is that every degradation is reported at ``warning`` or
    above.  The pre-fix path raised into the caller's ``warning``; a silent
    ``NaN`` would be strictly less visible than what it replaced.
    """
    import serve

    with caplog.at_level("WARNING"):
        serve._cache_put("BBCA.JK", pd.Series({"rsi": "abc", "volatility_20d": 9.0}),
                         SCHEMA, "0900")

    messages = [r.getMessage() for r in caplog.records]
    assert any("rsi" in m for m in messages), messages

    got = serve._cache_get("BBCA.JK", SCHEMA, "0900")
    assert got is not None
    assert np.isnan(got["rsi"]), "a non-number is read as NaN, never as 0.0"
    assert got["volatility_20d"] == 9.0


# ═══════════════════════════════════════════════════════════════════════════════
# FIX ROUND 1, MINOR 7 — coverage counts tickers, not rows
# ═══════════════════════════════════════════════════════════════════════════════

def test_coverage_cannot_exceed_the_universe(cache_db):
    """A mid-day retrain must not inflate the reported coverage.

    Two schema fingerprints for the same ticker are two rows and one cached
    ticker.  Counting rows would report ``90/45 (200%)`` for a fully cached
    universe — and ``monitoring/indonesian_stock_dashboard_v3.json`` reads this
    figure as "how many of the 45 tickers".
    """
    import serve

    universe = list(serve.TICKERS)
    assert len(universe) == 45, len(universe)

    first = ["rsi"]
    second = ["rsi", "volatility_20d"]
    for ticker in universe:
        serve._cache_put(ticker, pd.Series({"rsi": 1.0}), first,
                         serve.CACHE_WINDOW_DEFAULT)
        serve._cache_put(ticker, pd.Series({"rsi": 1.0, "volatility_20d": 2.0}),
                         second, serve.CACHE_WINDOW_DEFAULT)

    status = serve.cache_status()

    assert status["total_cached"] == 2 * len(universe), status
    assert status["cached_today"] == len(universe), status
    assert status["coverage"] == f"{len(universe)}/{len(universe)} (100%)", status
    percent = int(status["coverage"].split("(")[1].rstrip("%)"))
    assert percent <= 100, status
