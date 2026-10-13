"""Task 1.6 — one model selector that raises (F2, F9).

The audit found the committed ``model_index.json`` mapping zero tickers while 45
model folders sat on disk, so the API's ``load_best_model`` fell through to
``sorted(os.listdir(models_dir))[0]`` and answered *every* ticker — including a
ticker that does not exist — with one folder's model.  Worse, the two consumers
disagreed by construction: the API's fallback scanned ascending, the
backtester's scanned ``reverse=True``, so the same ticker got different models
from the API and from ``cli.py backtest``.

These tests pin the single selector that replaces both fallbacks: the index is
the only source of a ticker's model, an unmapped ticker raises
``ModelNotFound``, and a request path never walks the store looking for a model
to borrow.

The legacy-model ruling (R1) is pinned too: all 45 models on disk predate Task
1.4's ``feature_schema.json``, so a missing contract falls back to the model's
own ``feature_names_in_`` — but at ``warning``, naming the ticker, because a
silent fallback is exactly the defect this task removes.
"""
import json

import pytest

from model_store import ModelNotFound, resolve_model_folder


# ═══════════════════════════════════════════════════════════════════════════════
# The brief's two tests — the fallback is gone
# ═══════════════════════════════════════════════════════════════════════════════

def test_unknown_ticker_raises(monkeypatch, tmp_path):
    import model_store
    monkeypatch.setattr(model_store, "load_model_index", lambda: {"BBCA.JK": "m-1"})
    with pytest.raises(ModelNotFound):
        resolve_model_folder("ZZZZ.JK")


def test_no_fallback_to_first_folder(monkeypatch):
    import model_store
    monkeypatch.setattr(model_store, "load_model_index", lambda: {})
    with pytest.raises(ModelNotFound):
        resolve_model_folder("BBCA.JK")


# ═══════════════════════════════════════════════════════════════════════════════
# F2 — the store is never scanned on a request path
# ═══════════════════════════════════════════════════════════════════════════════

def _store_with_one_loadable_model(tmp_path):
    """A store holding one *perfectly loadable* model folder and no index entry.

    This is the F2 shape exactly: 45 folders on disk, nothing mapped.  Both
    deleted fallbacks would have loaded this folder for any ticker.
    """
    artifact_dir = tmp_path / "m-1" / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
    return tmp_path


def test_unmapped_ticker_never_borrows_a_folder(monkeypatch, tmp_path):
    import model_store

    monkeypatch.setattr(model_store, "model_dir", lambda: _store_with_one_loadable_model(tmp_path))
    monkeypatch.setattr(model_store, "load_model_index", lambda: {})

    loads = []

    def _record(path):
        loads.append(path)
        raise AssertionError("the store was walked for a model to substitute")

    monkeypatch.setattr(model_store, "_load_xgboost_model", _record)

    with pytest.raises(ModelNotFound):
        model_store.load_model_for("BBCA.JK")
    assert loads == [], "no model may be loaded when the index maps no ticker"


def test_load_model_for_refuses_a_folder_the_index_does_not_name(monkeypatch, tmp_path):
    """A mapped ticker whose folder is absent raises; it never scans for another."""
    import model_store

    store = _store_with_one_loadable_model(tmp_path)
    monkeypatch.setattr(model_store, "model_dir", lambda: store)
    monkeypatch.setattr(model_store, "load_model_index", lambda: {"BBCA.JK": "m-404"})
    monkeypatch.setattr(
        model_store, "_load_xgboost_model",
        lambda path: pytest.fail(f"nothing may be loaded from {path}"),
    )

    with pytest.raises(ModelNotFound, match="m-404"):
        model_store.load_model_for("BBCA.JK")


# ═══════════════════════════════════════════════════════════════════════════════
# The path — constructed in exactly one place
# ═══════════════════════════════════════════════════════════════════════════════

def test_model_dir_is_absolute_and_is_the_store():
    """``model_dir()`` is resolved from model_store's own location, not the cwd.

    Every caller (serve, backtest, cli, predict) delegates here, so none of them
    can drift to its own relative base — which is how the API and the backtester
    came to disagree in the first place.
    """
    import model_store

    store = model_store.model_dir()
    assert store.is_absolute()
    assert store.name == "models"
    assert store.parent.name == "1"
    assert store.parent.parent.name == "mlruns"
    assert store == (model_store._REPO_ROOT / "mlruns" / "1" / "models").resolve()
    assert model_store.index_path() == store / "model_index.json"


def test_model_dir_is_independent_of_the_working_directory(monkeypatch, tmp_path):
    import model_store

    before = model_store.model_dir()
    monkeypatch.chdir(tmp_path)
    assert model_store.model_dir() == before


# ═══════════════════════════════════════════════════════════════════════════════
# The index — ticker -> folder, and every degradation is a warning
# ═══════════════════════════════════════════════════════════════════════════════

def _write_index(tmp_path, payload):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model_index.json"
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )
    return path


def test_index_normalises_the_on_disk_shape(monkeypatch, tmp_path):
    """The committed index holds entry dicts; the store exposes ``ticker -> folder``."""
    import model_store

    _write_index(tmp_path, {
        "total_models": 45, "mapped_tickers": 2, "errors": 0,
        "ticker_to_model": {
            "BBCA.JK": {"model_folder": "m-1", "match_score": 999, "n_features": 60},
            "BBRI.JK": {"model_folder": "m-2", "match_score": 999, "n_features": 60},
        },
    })
    monkeypatch.setattr(model_store, "model_dir", lambda: tmp_path)

    assert model_store.load_model_index() == {"BBCA.JK": "m-1", "BBRI.JK": "m-2"}


def test_empty_index_warns_and_maps_nothing(monkeypatch, tmp_path, caplog):
    """The committed state: the file exists, parses, and maps zero tickers."""
    import model_store

    _write_index(tmp_path, {
        "total_models": 1, "indexed_models": 0, "mapped_tickers": 0,
        "errors": 0, "ticker_to_model": {},
    })
    monkeypatch.setattr(model_store, "model_dir", lambda: tmp_path)

    with caplog.at_level("WARNING"):
        assert model_store.load_model_index() == {}
    assert any("no ticker" in r.message.lower() for r in caplog.records)


@pytest.mark.parametrize("payload", ["{not json", "[]", '{"ticker_to_model": 3}'])
def test_unusable_index_warns_rather_than_raising(monkeypatch, tmp_path, caplog, payload):
    """An external file read degrades to a safe default — visibly, not silently."""
    import model_store

    _write_index(tmp_path, payload)
    monkeypatch.setattr(model_store, "model_dir", lambda: tmp_path)

    with caplog.at_level("WARNING"):
        assert model_store.load_model_index() == {}
    assert caplog.records, "an unusable index must be visible at warning or above"


def test_missing_index_file_warns(monkeypatch, tmp_path, caplog):
    import model_store

    monkeypatch.setattr(model_store, "model_dir", lambda: tmp_path)
    with caplog.at_level("WARNING"):
        assert model_store.load_model_index() == {}
    assert any("unreadable" in r.message.lower() for r in caplog.records)


def test_entry_without_a_folder_is_dropped_with_a_warning(monkeypatch, tmp_path, caplog):
    import model_store

    _write_index(tmp_path, {
        "ticker_to_model": {"BBCA.JK": {"model_folder": "m-1"}, "BROKEN.JK": {"n_features": 3}},
    })
    monkeypatch.setattr(model_store, "model_dir", lambda: tmp_path)

    with caplog.at_level("WARNING"):
        assert model_store.load_model_index() == {"BBCA.JK": "m-1"}
    assert any("BROKEN.JK" in r.getMessage() for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════════
# R1 — prefer the persisted contract, fall back to the model's own list loudly
# ═══════════════════════════════════════════════════════════════════════════════

class _FakeModel:
    """Stands in for a loaded estimator; only its feature accessors matter."""

    def __init__(self, names):
        if names is not None:
            self.feature_names_in_ = list(names)


class _FakeWrapper:
    """``CalibratedClassifierCV``-shaped: the names live on ``.estimator``."""

    def __init__(self, names):
        self.estimator = _FakeModel(names)


def _mapped_store(tmp_path, monkeypatch, folder="m-1"):
    """Point model_store at a tmp store mapping BBCA.JK to ``folder``."""
    import model_store

    artifact_dir = tmp_path / folder / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
    monkeypatch.setattr(model_store, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(model_store, "load_model_index", lambda: {"BBCA.JK": folder})
    return artifact_dir


def test_schemaless_model_loads_from_its_own_names_with_a_warning(monkeypatch, tmp_path, caplog):
    """R1: 45/45 models on disk have no contract yet; they must still load."""
    import model_store

    artifact_dir = _mapped_store(tmp_path, monkeypatch)
    assert not (artifact_dir / "feature_schema.json").exists()
    model = _FakeModel(["rsi_14", "close"])
    monkeypatch.setattr(model_store, "_load_xgboost_model", lambda path: model)

    with caplog.at_level("WARNING"):
        loaded, columns = model_store.load_model_for("BBCA.JK")

    assert loaded is model
    assert columns == ["rsi_14", "close"]
    assert any(
        r.levelname == "WARNING" and "BBCA.JK" in r.getMessage()
        and "no persisted feature schema" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_wrapped_estimator_names_are_found(monkeypatch, tmp_path, caplog):
    import model_store

    _mapped_store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        model_store, "_load_xgboost_model", lambda path: _FakeWrapper(["rsi_14", "macd"])
    )

    with caplog.at_level("WARNING"):
        _, columns = model_store.load_model_for("BBCA.JK")
    assert columns == ["rsi_14", "macd"]


def test_numpy_feature_names_are_read_not_tested_for_truth(monkeypatch, tmp_path, caplog):
    """A fitted XGBClassifier stores ``feature_names_in_`` as a numpy array.

    ``if names:`` on a multi-element array raises ``ValueError: truth value of
    an array with more than one element is ambiguous``, so the real legacy
    models cannot be read that way — this test pins the conversion instead.
    Real models, not lists, are why the probe behind this task found the bug.
    """
    np = pytest.importorskip("numpy")

    import model_store

    _mapped_store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        model_store, "_load_xgboost_model",
        lambda path: _FakeModel(np.array(["rsi_14", "close", "macd"])),
    )

    with caplog.at_level("WARNING"):
        _, columns = model_store.load_model_for("BBCA.JK")
    assert columns == ["rsi_14", "close", "macd"]


def test_persisted_schema_wins_and_the_fallback_stays_quiet(monkeypatch, tmp_path, caplog):
    """The contract on disk is authoritative, order included."""
    from features.schema import save_feature_schema

    import model_store

    artifact_dir = _mapped_store(tmp_path, monkeypatch)
    save_feature_schema(["b", "a"], artifact_dir / "feature_schema.json")
    monkeypatch.setattr(
        model_store, "_load_xgboost_model", lambda path: _FakeModel(["a", "b"])
    )

    with caplog.at_level("WARNING"):
        _, columns = model_store.load_model_for("BBCA.JK")
    assert columns == ["b", "a"]
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_load_model_for_returns_the_persisted_schema(monkeypatch, tmp_path, caplog):
    from features.schema import save_feature_schema

    import model_store

    artifact_dir = _mapped_store(tmp_path, monkeypatch)
    save_feature_schema(["rsi_14", "close"], artifact_dir / "feature_schema.json")
    monkeypatch.setattr(
        model_store, "_load_xgboost_model", lambda path: _FakeModel(["something_else"])
    )

    with caplog.at_level("WARNING"):
        _, columns = model_store.load_model_for("BBCA.JK")
    assert columns == ["rsi_14", "close"]
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_malformed_persisted_schema_raises_instead_of_falling_back(monkeypatch, tmp_path):
    """Only an *absent* contract falls back; a corrupt one is an error."""
    from features.schema import FeatureSchemaError

    import model_store

    artifact_dir = _mapped_store(tmp_path, monkeypatch)
    (artifact_dir / "feature_schema.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(model_store, "_load_xgboost_model", lambda path: _FakeModel(["rsi_14"]))

    with pytest.raises(FeatureSchemaError, match="feature_schema.json"):
        model_store.load_model_for("BBCA.JK")


def test_model_without_any_feature_names_cannot_be_enforced(monkeypatch, tmp_path):
    from features.schema import FeatureSchemaError

    import model_store

    _mapped_store(tmp_path, monkeypatch)
    monkeypatch.setattr(model_store, "_load_xgboost_model", lambda path: _FakeModel(None))

    with pytest.raises(FeatureSchemaError, match="BBCA.JK"):
        model_store.load_model_for("BBCA.JK")
