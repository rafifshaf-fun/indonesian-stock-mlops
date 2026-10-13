"""Task 1.4 — one explicit feature schema per model (F5).

The audit found the 45 logged models carrying 45 *different* 60-name feature
lists (union 132, intersection 1) with nothing validating the column set a model
was fed: consumers silently padded absent names with ``NaN``, and any change to
a feature module invalidated all 45 models with no failure signal.  These tests
pin the contract that replaces that:

**The schema API** — ``save_feature_schema`` / ``load_feature_schema`` persist an
ordered column list as JSON, and ``align_to_schema`` is the only place where a
computed frame is allowed to meet a model's columns.

**The exclusion (R2)** — ``SERVING_UNPRODUCIBLE_*`` names every family serving
cannot reproduce, so the persisted schema can never demand one.  Measured against
the committed ``data/processed/features.parquet``: ``vp_*`` (19 columns) and
``google_trend`` (1) are live training signal that serving cannot build (5-minute
intraday downloads; a Google Trends fetch), while ``idn_*`` (10) is already dead
in training — ``fetch_idx_fundamentals`` is a no-network stub returning
``{field: np.nan}`` for all ten, so all ten are all-NaN in the parquet
(``src/features/fetchers.py:162-173``).

**The storage location (R4)** — the schema is written *beside* ``MLmodel`` in the
logged model's artifact directory.  That is the directory every existing consumer
already hands to ``mlflow.xgboost.load_model`` (``src/serve.py:82,241,262`` and
``scripts/build_model_index.py:95``), so the contract needs no new lookup path.
The mechanism was established by logging a throwaway model into a scratch store
and inspecting the resulting tree, never by assuming an MLflow layout; the
round-trip test at the bottom of this file pins it against the installed MLflow.
"""
import json
import os
import pathlib
import re
import types
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest

# ``features.schema``'s MLflow-backed persistence imports mlflow, which imports
# mlflow.tracing, which imports opentelemetry, whose runtime-context loader still
# calls the removed ``importlib.metadata`` dict interface.  On Python 3.11 that
# emits ``DeprecationWarning: SelectableGroups dict interface is deprecated. Use
# select.`` from opentelemetry/util/_importlib_metadata.py:32 — upstream, not this
# repository.  The suite runs with ``filterwarnings = ["error"]`` (Task 1.1's
# deliberate choice), so without this filter the import aborts the storage test.
#
# The second is upstream MLflow 3.13 too, and only *saving* a model reaches it:
# ``mlflow.xgboost.log_model`` lazily imports ``mlflow.pyfunc``, which imports
# ``mlflow.types.responses``, whose module-level pydantic type-hint inference
# warns "Any type hint is inferred as AnyType" about MLflow's own ChatAgent
# response types (mlflow/types/type_hints.py:185).  Nothing in this repository
# reaches it — serving and the model-index builder only ever *load* models — but
# the round-trip test below must save one to prove the storage mechanism.  The
# filter is scoped to that one message and category (in DOTALL form because the
# colour codes MLflow prefixes it with vary by environment): every other warning
# — including any warning this task's own code raises — still escalates.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:SelectableGroups dict interface is deprecated:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:(?s).*Any type hint is inferred as AnyType.*:UserWarning"
    ),
]

SCHEMA_FILENAME = "feature_schema.json"


# ═══════════════════════════════════════════════════════════════════════════════
# The brief's three tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_roundtrip(tmp_path):
    from features.schema import save_feature_schema, load_feature_schema

    p = tmp_path / SCHEMA_FILENAME
    save_feature_schema(["b", "a"], p)
    assert load_feature_schema(p) == ["b", "a"]  # order preserved


def test_missing_column_raises_not_pads():
    from features.schema import align_to_schema, FeatureSchemaError

    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(FeatureSchemaError, match="b"):
        align_to_schema(df, ["a", "b"])


def test_reordered_frame_is_reordered_not_permuted():
    from features.schema import align_to_schema

    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    out = align_to_schema(df, ["b", "a"])
    assert list(out.columns) == ["b", "a"]
    assert out["b"].iloc[0] == 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# R1 — raise on MISSING, silently drop EXTRA, never pad
# ═══════════════════════════════════════════════════════════════════════════════

def test_extra_frame_columns_are_dropped_not_an_error():
    """Serving's computed frame legitimately carries OHLCV and intermediates.

    Reading the brief literally — "raises listing every missing or extra column" —
    would make the function unusable at the serving call site, so the ruling is:
    absent schema columns raise, extra frame columns are dropped, and the result
    is exactly the schema's columns in the schema's order.  The schema-side
    guarantee (the schema never names a column serving cannot produce) is
    enforced at derivation time by ``candidate_columns``, not here.
    """
    from features.schema import align_to_schema

    df = pd.DataFrame(
        {"open": [1.0], "close": [2.0], "rsi_14": [3.0], "volume": [4.0]}
    )
    out = align_to_schema(df, ["rsi_14", "close"])

    assert list(out.columns) == ["rsi_14", "close"]
    assert out.iloc[0].tolist() == [3.0, 2.0]
    assert out.notna().all().all(), "align_to_schema must never insert NaN"


def test_every_missing_column_is_listed():
    from features.schema import align_to_schema, FeatureSchemaError

    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(FeatureSchemaError) as err:
        align_to_schema(df, ["a", "b", "c"])

    message = str(err.value)
    assert "b" in message and "c" in message, message
    assert not re.search(r"(?<![\w_])a(?![\w_])", message), (
        "a present column must not be reported as missing: " + message
    )


def test_align_is_a_no_op_when_the_frame_already_matches():
    from features.schema import align_to_schema

    df = pd.DataFrame({"b": [2.0], "a": [1.0]})
    out = align_to_schema(df, ["b", "a"])
    pd.testing.assert_frame_equal(out, df)


def test_saving_a_broken_schema_is_rejected(tmp_path):
    """A schema is an ordered, duplicate-free list of non-empty names."""
    from features.schema import save_feature_schema, FeatureSchemaError

    p = tmp_path / SCHEMA_FILENAME
    for broken in ([], ["a", "a"], ["a", ""], ["a", None]):
        with pytest.raises(FeatureSchemaError):
            save_feature_schema(broken, p)
    assert not p.exists(), "a rejected schema must not be written"


def test_loading_a_malformed_schema_is_rejected(tmp_path):
    from features.schema import load_feature_schema, FeatureSchemaError

    p = tmp_path / SCHEMA_FILENAME
    for malformed in ('{"a": 1}', '["a", 1]', "[]", '["a", "a"]', "null"):
        p.write_text(malformed)
        with pytest.raises(FeatureSchemaError):
            load_feature_schema(p)

    p.write_text(json.dumps(["ok", "fine"]))
    assert load_feature_schema(p) == ["ok", "fine"]


def test_loading_an_absent_schema_is_an_oserror_not_a_silent_empty(tmp_path):
    """A missing contract must not read as "no constraints"."""
    from features.schema import load_feature_schema

    with pytest.raises(OSError):
        load_feature_schema(tmp_path / "never-written.json")


# ═══════════════════════════════════════════════════════════════════════════════
# R2 — the contract excludes every family serving cannot produce
# ═══════════════════════════════════════════════════════════════════════════════

def test_exclusion_drops_exactly_the_unproducible_families():
    from features.schema import candidate_columns

    columns = [
        "close", "vp_poc", "rsi_14", "idn_roe", "google_trend",
        "vp_vwap_deviation", "idn_per", "volume", "macd",
    ]
    assert candidate_columns(columns) == ["close", "rsi_14", "volume", "macd"]


def test_exclusion_rules_are_the_documented_families():
    """The rule set is auditable: two prefixes and one exact name."""
    from features.schema import (
        SERVING_UNPRODUCIBLE_PREFIXES,
        SERVING_UNPRODUCIBLE_COLUMNS,
        is_unproducible,
        unproducible_reason,
    )

    assert set(SERVING_UNPRODUCIBLE_PREFIXES) == {"vp_", "idn_"}
    assert set(SERVING_UNPRODUCIBLE_COLUMNS) == {"google_trend"}

    # Reasons are not decoration: each entry is documented beside the family.
    for family, reason in {**SERVING_UNPRODUCIBLE_PREFIXES,
                           **SERVING_UNPRODUCIBLE_COLUMNS}.items():
        assert isinstance(reason, str) and len(reason) > 20, family

    assert is_unproducible("vp_anything_at_all")
    assert is_unproducible("idn_der")
    assert is_unproducible("google_trend")
    assert not is_unproducible("close")
    assert not is_unproducible("vp")          # a bare prefix is not a family member
    assert not is_unproducible("google_trends_sentiment")
    assert unproducible_reason("close") is None


def test_persisted_schema_names_no_unproducible_column(tmp_path):
    """The persisted contract can never demand a column serving cannot compute."""
    from features.schema import candidate_columns, save_feature_schema, load_feature_schema

    # A candidate list shaped like the real pooled frame: every excluded family
    # present, plus producible columns that must survive.
    candidates = (
        [f"vp_{i}" for i in range(19)]
        + ["idn_der", "idn_roe", "idn_roa", "idn_current_ratio", "idn_gpm",
           "idn_npm", "idn_eps_growth", "idn_bvps", "idn_per", "idn_pbv"]
        + ["google_trend"]
        + ["close", "rsi_14", "macd", "sector_return_1d"]
    )
    assert len(candidates) == 34

    p = tmp_path / SCHEMA_FILENAME
    save_feature_schema(candidate_columns(candidates), p)
    persisted = load_feature_schema(p)

    assert [c for c in persisted if c.startswith("vp_")] == []
    assert [c for c in persisted if c.startswith("idn_")] == []
    assert "google_trend" not in persisted
    assert persisted == ["close", "rsi_14", "macd", "sector_return_1d"]


# ═══════════════════════════════════════════════════════════════════════════════
# R4 — the schema is stored beside MLmodel in the logged model's directory
# ═══════════════════════════════════════════════════════════════════════════════

def _artifact_dir_of(model_id: str) -> pathlib.Path:
    """The on-disk artifact directory MLflow reports for a logged model.

    Resolved through ``MlflowClient.get_logged_model(...).artifact_location``,
    which is MLflow's own answer, so the test cannot pass by hardcoding a layout
    the installed version does not use.
    """
    from mlflow import MlflowClient

    location = MlflowClient().get_logged_model(model_id).artifact_location
    parsed = urlparse(location)
    assert parsed.scheme == "file", f"expected a file store, got {location!r}"
    path = parsed.path
    # file:///C:/... -> /C:/... on Windows
    if os.name == "nt" and path.startswith("/") and path[2:3] == ":":
        path = path[1:]
    return pathlib.Path(path)


@pytest.fixture
def scratch_mlflow(tmp_path, monkeypatch):
    """An isolated file-store MLflow in ``tmp_path`` — never the repo's store."""
    import mlflow

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    previous = mlflow.get_tracking_uri()
    store = tmp_path / "mlruns"
    store.mkdir()
    mlflow.set_tracking_uri(store.as_uri())
    mlflow.set_experiment("feature_schema_roundtrip")
    try:
        yield mlflow
    finally:
        mlflow.set_tracking_uri(previous)


def test_schema_round_trips_from_a_logged_models_directory(scratch_mlflow):
    """Persist through MLflow, read back from beside ``MLmodel``, order intact."""
    from mlflow import MlflowClient
    from xgboost import XGBClassifier

    from features.schema import (
        FEATURE_SCHEMA_FILENAME,
        load_feature_schema,
        log_feature_schema,
    )

    columns = ["rsi_14", "close", "sector_return_1d", "macd"]
    frame = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0, 4.0], "rsi_14": [0.1, 0.2, 0.3, 0.4],
         "macd": [0.0, 0.1, 0.0, -0.1],
         "sector_return_1d": [0.01, -0.01, 0.02, 0.0]}
    )
    target = pd.Series([0, 1, 0, 1])
    model = XGBClassifier(n_estimators=3, max_depth=2, verbosity=0)
    model.fit(frame, target)

    with scratch_mlflow.start_run():
        info = scratch_mlflow.xgboost.log_model(model, "model")
        assert info.model_id, "MLflow returned no logged-model id to attach to"
        log_feature_schema(info.model_id, columns)

    artifact_dir = _artifact_dir_of(info.model_id)
    assert (artifact_dir / "MLmodel").is_file(), artifact_dir

    on_disk = artifact_dir / FEATURE_SCHEMA_FILENAME
    assert on_disk.is_file(), (
        f"{FEATURE_SCHEMA_FILENAME} is not beside MLmodel in {artifact_dir}; "
        f"found {sorted(p.name for p in artifact_dir.iterdir())}"
    )
    assert load_feature_schema(on_disk) == columns

    # The consumer's own call — the exact one src/serve.py and
    # scripts/build_model_index.py make — still works from that directory.
    assert scratch_mlflow.xgboost.load_model(str(artifact_dir)) is not None

    # MLflow knows the file too, so nothing depends on scanning the raw tree.
    listed = {f.path for f in MlflowClient().list_logged_model_artifacts(info.model_id)}
    assert FEATURE_SCHEMA_FILENAME in listed, listed


# ═══════════════════════════════════════════════════════════════════════════════
# R5 — train.py selects once for every ticker and persists the contract
# ═══════════════════════════════════════════════════════════════════════════════

# ``train`` is imported inside a fixture, never at module scope: importing it
# calls ``warnings.filterwarnings("ignore")`` process-wide and points MLflow at
# the repository's own sqlite store, and neither may leak out of these tests.


def _synthetic_pool(n_tickers: int = 2, n_rows: int = 160) -> pd.DataFrame:
    """A Ticker x row frame shaped like ``data/processed/features.parquet``.

    Seventy producible numeric columns plus every serving-unproducible family —
    nineteen ``vp_*``, ten ``idn_*`` and ``google_trend`` — so a test can tell
    "the selection saw the pooled candidates" from "the selection saw
    everything".
    """
    rng = np.random.default_rng(20261013)
    columns = [f"feat_{i:03d}" for i in range(70)]
    columns += [f"vp_{i}" for i in range(19)]
    columns += [f"idn_{i}" for i in range(10)]
    columns += ["google_trend"]

    frames = []
    for t in range(n_tickers):
        data = {c: rng.normal(size=n_rows) for c in columns}
        data["target"] = (rng.random(n_rows) > 0.5).astype(int)
        data["ticker"] = f"TEST{t}.JK"
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def train_module(monkeypatch):
    """``train`` imported with its module-level side effects contained."""
    import warnings as _warnings

    saved_filters = list(_warnings.filters)
    saved_uri = os.environ.get("MLFLOW_TRACKING_URI")
    import mlflow
    saved_tracking = mlflow.get_tracking_uri()

    import train

    # train.py silences every warning process-wide as it is imported; under
    # filterwarnings = ["error"] that would hide the suite's own signal, so the
    # snapshot is restored before anything else runs.
    _warnings.filters[:] = saved_filters

    # shap registers its matplotlib colormaps on first import, and
    # shap/plots/colors/_colors.py:47 calls the already-deprecated
    # ``Colormap.set_bad``, which matplotlib answers with a
    # PendingDeprecationWarning.  Wholly upstream, and only on the first import
    # of shap: production never sees it because train.py runs with
    # ``filterwarnings("ignore")``, while this fixture deliberately restores the
    # suite's error escalation.  The import is therefore done here, once, under a
    # filter scoped to that single category — every other warning from shap, from
    # matplotlib or from this repository still escalates.
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", PendingDeprecationWarning)
        import shap  # noqa: F401  (pre-import only; the selection imports it itself)

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


class _FakeXgboostFlavor:
    def __init__(self, owner):
        self._owner = owner

    def log_model(self, model, artifact_path, **kwargs):
        self._owner.logged_models.append((artifact_path, model))
        return types.SimpleNamespace(model_id="m-fake0000000000000000000000000000")


class _FakeMlflow:
    """Records the calls train() makes, and never touches a real store."""

    def __init__(self):
        self.params: dict = {}
        self.metrics: dict = {}
        self.tags: dict = {}
        self.logged_models: list = []
        self.experiment = None
        self.run_name = None
        self.xgboost = _FakeXgboostFlavor(self)

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None, **kwargs):
        self.run_name = run_name
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


def test_global_selection_is_pooled_and_runs_once(train_module, monkeypatch):
    """One selection over every ticker's rows and the producible candidates."""
    train = train_module
    pool = _synthetic_pool()

    monkeypatch.setattr(train, "load_features", lambda path: pool)

    seen = []
    real_select = train._select_features_shap

    def counting_select(X, y, top_k=60):
        seen.append((X.shape, len(y)))
        return real_select(X, y, top_k=top_k)

    monkeypatch.setattr(train, "_select_features_shap", counting_select)

    schema = train.select_global_feature_schema("features.parquet")
    again = train.select_global_feature_schema("features.parquet")

    assert len(seen) == 1, "the selection must run once per run, not per ticker"
    assert seen[0][1] == len(pool), "selection must be pooled over every ticker"
    assert seen[0][0][1] == 70, "only producible candidates may be selected from"

    assert len(schema) == 60, "top_k=60 is kept"
    assert schema == again
    assert all(c.startswith("feat_") for c in schema), schema
    assert not [c for c in schema if c.startswith(("vp_", "idn_")) or c == "google_trend"]

    # And the same ordered list is a valid contract for every ticker's slice.
    from features.schema import align_to_schema

    for ticker in pool["ticker"].unique():
        X, _ = train.prepare_xy(pool[pool["ticker"] == ticker])
        assert list(align_to_schema(X, schema).columns) == schema


def test_train_trains_on_the_supplied_schema_without_reselecting(
        train_module, monkeypatch, tmp_path):
    """``feature_columns`` is the column set, and it is persisted beside the model."""
    train = train_module
    monkeypatch.setattr(train, "load_features",
                        lambda path: _synthetic_pool(n_tickers=1, n_rows=200))

    def boom(*args, **kwargs):
        raise AssertionError("a per-ticker selection ran — the run selects once")

    monkeypatch.setattr(train, "_select_features_shap", boom)

    fake = _FakeMlflow()
    monkeypatch.setattr(train, "mlflow", fake)
    monkeypatch.setattr(train, "IS_CI", False)

    persisted = []
    monkeypatch.setattr(train, "log_feature_schema",
                        lambda model_id, columns: persisted.append((model_id, list(columns))))

    def _png(name):
        path = tmp_path / name
        path.write_bytes(b"")
        return str(path)

    monkeypatch.setattr(train, "plot_feature_importance",
                        lambda model, names, ticker, top_n=20: _png(f"{ticker}_imp.png"))
    monkeypatch.setattr(train, "plot_confusion_matrix",
                        lambda y_true, y_pred, ticker: _png(f"{ticker}_cm.png"))

    schema = ["feat_000", "feat_001", "feat_002", "feat_003", "feat_004"]
    train.train("TEST0.JK", feature_columns=schema)

    # The model itself was fitted on the shared schema, in the schema's order —
    # not on a per-ticker selection.
    assert [path for path, _ in fake.logged_models] == ["model"]
    fitted = fake.logged_models[0][1]
    assert list(fitted.feature_names_in_) == schema
    assert fake.params["n_features"] == len(schema)
    assert fake.params["ticker"] == "TEST0.JK"
    assert persisted == [("m-fake0000000000000000000000000000", schema)]


def test_a_supplied_schema_naming_an_unproducible_column_is_rejected(
        train_module, monkeypatch):
    """A contract that names a column serving cannot build is refused, whatever
    its source — that is the defect this task removes."""
    train = train_module
    monkeypatch.setattr(train, "load_features",
                        lambda path: _synthetic_pool(n_tickers=1, n_rows=120))
    monkeypatch.setattr(train, "mlflow", _FakeMlflow())

    with pytest.raises(train.FeatureSchemaError, match="vp_poc"):
        train.train("TEST0.JK", feature_columns=["feat_000", "vp_poc"])
