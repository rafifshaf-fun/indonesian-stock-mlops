"""Task 1.7 — training writes the model index (F10).

The audit found ``mlruns/1/models/model_index.json`` committed with
``mapped_tickers: 0`` while 45 model folders sat beside it, and no way back:
``scripts/build_model_index.py`` never set a tracking URI, so every
``client.get_run(run_id)`` failed, a bare ``except Exception: pass`` hid it, and
the script silently fell through to matching a model's feature names against the
features CSV — an *invented* ticker.  Nothing wrote the index on the way out of
training either, so the file could only get staler.

These tests pin the replacement:

* ``write_model_index`` is the single writer, in ``model_store`` beside the
  reader, through the same ``index_path()`` accessor, with the file's shape
  unchanged.
* training records the folder MLflow actually created for each ticker
  (``ModelInfo.model_id`` — the relationship is measured, see
  ``train._logged_model_folder``), keeps it when ``log_model`` never ran (CI),
  and never invents one.
* an empty mapping is refused rather than written over a populated index: that
  write is how every prediction would be disabled while everything looked fine.
"""
from __future__ import annotations

import json
import os
import types
import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from model_store import load_model_index, write_model_index

# ``src/train.py`` imports mlflow, which imports mlflow.tracing, which imports
# opentelemetry, whose runtime-context loader still calls the removed
# ``importlib.metadata`` dict interface.  On Python 3.11 that emits
# ``DeprecationWarning: SelectableGroups dict interface is deprecated. Use
# select.`` from opentelemetry/util/_importlib_metadata.py:32 — upstream, not this
# repository.  The suite runs with ``filterwarnings = ["error"]`` (Task 1.1's
# deliberate choice), so without this filter importing the training module aborts
# every test below.  The same narrow filter, for the same reason, is declared in
# tests/test_feature_schema.py; it is scoped to that one message and category, so
# every other warning — including any this task's own code raises — escalates.
pytestmark = pytest.mark.filterwarnings(
    "ignore:SelectableGroups dict interface is deprecated:DeprecationWarning"
)


@pytest.fixture(autouse=True)
def _keep_off_the_real_store(monkeypatch, tmp_path):
    """No test in this module may read or write the tracked model store.

    ``model_dir`` is redirected for every test, so a stray call that forgets to
    pass a path still lands in ``tmp_path`` rather than in ``mlruns/``.
    """
    import model_store

    monkeypatch.setattr(model_store, "model_dir", lambda: tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# The brief's test — the writer exists and round-trips
# ═══════════════════════════════════════════════════════════════════════════════

def test_index_roundtrip(tmp_path, monkeypatch):
    import model_store
    p = tmp_path / "model_index.json"
    monkeypatch.setattr(model_store, "index_path", lambda: p)
    write_model_index({"BBCA.JK": "m-1"}, p)
    assert json.loads(p.read_text())["ticker_to_model"] == {"BBCA.JK": "m-1"}


def test_the_writer_defaults_to_the_one_path_accessor(tmp_path, monkeypatch, caplog):
    """R1: the writer and the reader share ``index_path()``, not two paths."""
    import model_store

    p = tmp_path / "model_index.json"
    monkeypatch.setattr(model_store, "index_path", lambda: p)

    with caplog.at_level("INFO"):
        write_model_index({"BBCA.JK": "m-1"})

    assert p.is_file(), "writing with no path must use index_path()"
    assert load_model_index() == {"BBCA.JK": "m-1"}
    assert any("now maps 1 ticker" in r.getMessage() for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════════
# The file's shape does not change
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_file_keeps_the_committed_shape_and_reads_back(tmp_path, monkeypatch):
    """The five committed keys stay, and the entry is a plain folder name.

    ``load_model_index`` is the only reader and it normalises both entry shapes;
    the writer emits the folder name itself, which is also what
    ``resolve_model_folder`` hands to ``mlruns/1/models/<folder>/artifacts``.
    """
    import model_store

    p = tmp_path / "model_index.json"
    monkeypatch.setattr(model_store, "index_path", lambda: p)

    write_model_index({"BBRI.JK": "m-2", "BBCA.JK": "m-1"})

    payload = json.loads(p.read_text())
    assert set(payload) == {
        "total_models", "indexed_models", "mapped_tickers", "errors", "ticker_to_model",
    }
    assert payload["total_models"] == 2
    assert payload["indexed_models"] == 2
    assert payload["mapped_tickers"] == 2
    assert payload["errors"] == 0
    assert payload["ticker_to_model"] == {"BBCA.JK": "m-1", "BBRI.JK": "m-2"}
    assert load_model_index() == {"BBCA.JK": "m-1", "BBRI.JK": "m-2"}


def test_a_run_keeps_the_mappings_it_did_not_retrain(tmp_path, monkeypatch):
    """Retraining one ticker must not unmap the other 44.

    ``python src/train.py --tune`` retrains ten tickers and ``--ticker X`` one.
    Replacing the file with just those would disable every other prediction
    silently — the same failure as writing it empty, only quieter.  A full
    retrain produces exactly this file anyway, because every ticker is named.
    """
    import model_store

    p = tmp_path / "model_index.json"
    monkeypatch.setattr(model_store, "index_path", lambda: p)
    write_model_index({"BBCA.JK": "m-1", "BBRI.JK": "m-2"})

    write_model_index({"BBRI.JK": "m-9"})

    assert load_model_index() == {"BBCA.JK": "m-1", "BBRI.JK": "m-9"}
    assert json.loads(p.read_text())["mapped_tickers"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# R3 — an empty mapping never overwrites a populated index
# ═══════════════════════════════════════════════════════════════════════════════

def test_an_empty_mapping_never_overwrites_a_populated_index(tmp_path, monkeypatch, caplog):
    """A CI-mode run logs no model, so it must leave the index alone.

    ``mlflow.xgboost.log_model`` only runs when ``IS_CI`` is false.  Writing the
    empty mapping it produces would set ``mapped_tickers: 0`` over a working
    index, and every prediction would fail — the exact failure this task exists
    to end.  The old file is asserted byte-for-byte: the write is skipped, not
    merely made harmless.
    """
    import model_store

    p = tmp_path / "model_index.json"
    # A *populated* index in the committed entry shape, at a formatting the
    # writer does not produce — so a rewrite of any kind changes the bytes.
    p.write_text(json.dumps({
        "total_models": 45, "indexed_models": 45, "mapped_tickers": 1, "errors": 0,
        "ticker_to_model": {"BBCA.JK": {"model_folder": "m-1", "match_score": 999}},
    }, indent=4), encoding="utf-8")
    before = p.read_bytes()
    monkeypatch.setattr(model_store, "index_path", lambda: p)

    with caplog.at_level("WARNING"):
        write_model_index({}, p)

    assert p.read_bytes() == before, "an empty mapping must not touch the index"
    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"], \
        "refusing to empty the index must be visible at warning or above"
    assert any("unchanged" in r.getMessage() for r in caplog.records)


def test_an_empty_index_may_still_be_written_to_the_empty_store(tmp_path, monkeypatch):
    """The guard refuses an *empty mapping*, not the act of writing.

    A store with no index yet is the ordinary first-retrain state; the writer
    creates the file (and its directory) for a run that logged models.
    """
    import model_store

    p = tmp_path / "nested" / "model_index.json"
    monkeypatch.setattr(model_store, "index_path", lambda: p)

    write_model_index({"BBCA.JK": "m-1"})

    assert json.loads(p.read_text())["ticker_to_model"] == {"BBCA.JK": "m-1"}


# ═══════════════════════════════════════════════════════════════════════════════
# R2 / R5 — training reports the folder MLflow created, and never invents one
# ═══════════════════════════════════════════════════════════════════════════════

RAW_TICKERS = ("BBCA.JK",)


@pytest.fixture
def train_module(monkeypatch):
    """``train`` imported with its module-level side effects contained.

    ``src/train.py`` calls ``warnings.filterwarnings("ignore")`` as it is
    imported, which under ``filterwarnings = ["error"]`` would hide the rest of
    the suite's signal; the snapshot is restored immediately.  The tracking URI
    it installs process-wide is restored on teardown too.
    """
    saved_filters = list(_warnings.filters)
    saved_uri = os.environ.get("MLFLOW_TRACKING_URI")
    import mlflow
    saved_tracking = mlflow.get_tracking_uri()

    import train

    _warnings.filters[:] = saved_filters
    monkeypatch.setattr(train, "_GLOBAL_SCHEMA_CACHE", {}, raising=False)

    try:
        yield train
    finally:
        _warnings.filters[:] = saved_filters
        if saved_uri is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = saved_uri
        mlflow.set_tracking_uri(saved_tracking)


def _synthetic_pool(n_rows: int = 200, tickers=RAW_TICKERS) -> pd.DataFrame:
    """A tiny pooled frame: six numeric features, two classes, one ticker."""
    rng = np.random.default_rng(20261013)
    frames = []
    for ticker in tickers:
        data = {f"feat_{i:03d}": rng.normal(size=n_rows) for i in range(6)}
        data["target"] = (rng.random(n_rows) > 0.5).astype(int)
        data["ticker"] = ticker
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


SCHEMA = [f"feat_{i:03d}" for i in range(6)]


class _FakeXgboostFlavor:
    """The one MLflow call that creates a logged model on disk."""

    def __init__(self, owner):
        self._owner = owner

    def log_model(self, model, artifact_path, **kwargs):
        self._owner.logged_models.append((artifact_path, model))
        return self._owner.model_info


class _FakeMlflow:
    """Records what ``train()`` logs, and never touches a real store."""

    def __init__(self, model_info):
        self.model_info = model_info
        self.params: dict = {}
        self.metrics: dict = {}
        self.tags: dict = {}
        self.logged_models: list = []
        self.xgboost = _FakeXgboostFlavor(self)

    def set_experiment(self, name):
        pass

    def start_run(self, run_name=None, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def log_params(self, params):
        self.params.update(params)

    def log_param(self, key, value):
        self.params[key] = value

    def log_metric(self, key, value):
        self.metrics[key] = value

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_artifact(self, path, artifact_path=None):
        pass


def _prepare_training(train, monkeypatch, tmp_path, model_info):
    """Point ``train`` at synthetic data and a recording MLflow."""
    monkeypatch.setattr(train, "load_features", lambda path: _synthetic_pool())
    fake = _FakeMlflow(model_info)
    monkeypatch.setattr(train, "mlflow", fake)

    def _png(name):
        path = tmp_path / name
        path.write_bytes(b"")
        return str(path)

    monkeypatch.setattr(train, "plot_feature_importance",
                        lambda model, names, ticker, top_n=20: _png(f"{ticker}_imp.png"))
    monkeypatch.setattr(train, "plot_confusion_matrix",
                        lambda y_true, y_pred, ticker: _png(f"{ticker}_cm.png"))
    return fake


def test_train_returns_the_folder_mlflow_created_for_the_model(
        train_module, monkeypatch, tmp_path):
    """R2: the parent learns the folder from ``ModelInfo``, not from a guess.

    ``mlflow.xgboost.log_model`` returns a ``ModelInfo`` whose ``model_id`` is
    the name MLflow gives the logged model's directory under
    ``mlruns/1/models/`` — measured against a real store, see
    ``train._logged_model_folder``.  ``train()`` therefore reports that id
    verbatim, and the caller has the ticker -> folder pair to write.
    """
    train = train_module
    monkeypatch.setattr(train, "IS_CI", False)
    persisted = []
    monkeypatch.setattr(train, "log_feature_schema",
                        lambda model_id, columns: persisted.append((model_id, list(columns))))

    model_id = "m-0123456789abcdef0123456789abcdef"
    fake = _prepare_training(
        train, monkeypatch, tmp_path, types.SimpleNamespace(model_id=model_id)
    )

    folder = train.train("BBCA.JK", feature_columns=SCHEMA)

    assert folder == model_id, "the index must name the folder MLflow created"
    assert fake.logged_models and fake.logged_models[0][0] == "model"
    assert persisted == [(model_id, SCHEMA)], "the contract lands beside the model"


def test_ci_mode_logs_no_model_and_reports_no_folder(train_module, monkeypatch, tmp_path):
    """R5: in CI ``log_model`` never runs, so there is nothing to map.

    ``ModelInfo`` does not exist on that path; asking for it must not raise an
    ``AttributeError`` out of ``train()``.  The ticker simply contributes no
    entry, and the index write is skipped rather than emptied (R3).
    """
    train = train_module
    monkeypatch.setattr(train, "IS_CI", True)
    monkeypatch.setattr(train, "log_feature_schema",
                        lambda model_id, columns: pytest.fail("no model was logged"))
    fake = _prepare_training(train, monkeypatch, tmp_path, types.SimpleNamespace())

    folder = train.train("BBCA.JK", feature_columns=SCHEMA)

    assert folder is None
    assert fake.logged_models == [], "CI mode must stay metrics-only"
    assert fake.metrics, "the run still reports its metrics"


def test_a_model_info_without_an_id_is_left_out_with_a_warning(
        train_module, monkeypatch, tmp_path, caplog):
    """A logged model whose id cannot be read is not guessed at (R2).

    Writing a folder name MLflow never reported would put a mapping in the index
    that points nowhere; leaving the ticker unmapped fails loudly at prediction
    time instead.
    """
    train = train_module
    monkeypatch.setattr(train, "IS_CI", False)
    monkeypatch.setattr(train, "log_feature_schema",
                        lambda model_id, columns: pytest.fail("nothing can be named"))
    _prepare_training(train, monkeypatch, tmp_path, types.SimpleNamespace(model_id=None))

    with caplog.at_level("WARNING"):
        folder = train.train("BBCA.JK", feature_columns=SCHEMA)

    assert folder is None
    assert any(
        r.levelname == "WARNING" and "BBCA.JK" in r.getMessage()
        and "no model id" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_the_worker_hands_the_parent_the_folder(train_module, monkeypatch, caplog):
    """The loky worker's return value carries the folder to the parent.

    ``_train_one`` runs in a separate process, so the only thing the parent can
    see is this tuple; it is the whole transport for ticker -> folder.
    """
    train = train_module
    monkeypatch.setattr(train, "train", lambda ticker, **kwargs: "m-worker-folder")
    assert train._train_one(("BBCA.JK", False, "features.csv", SCHEMA)) == \
        ("BBCA.JK", True, "m-worker-folder")

    def _boom(*args, **kwargs):
        raise RuntimeError("no model")

    monkeypatch.setattr(train, "train", _boom)
    with caplog.at_level("ERROR"):
        assert train._train_one(("BBCA.JK", False, "features.csv", SCHEMA)) == \
            ("BBCA.JK", False, None)


def test_the_parent_maps_only_the_workers_that_produced_a_folder(train_module):
    """The parallel aggregation reads the folder out of each worker's result.

    A failed worker and a CI-mode worker both report no folder, and neither may
    end up in the index; a successful one contributes exactly the id MLflow
    returned.
    """
    train = train_module

    entries = train._index_entries_from_results([
        ("BBCA.JK", True, "m-aaa"),
        ("BBRI.JK", False, None),
        ("BMRI.JK", True, None),          # CI mode: trained, but nothing logged
        ("TLKM.JK", True, "m-bbb"),
    ])

    assert entries == {"BBCA.JK": "m-aaa", "TLKM.JK": "m-bbb"}
