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
    assert set(serve._cache_columns(cache_db)) == set(serve.CACHE_COLUMNS)


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
    assert set(serve._cache_columns(cache_db)) == set(serve.CACHE_COLUMNS)

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
