"""Task 3.3 — evaluate out-of-sample: the split, the fleet, and the number.

The backtest has been running since Task 3.2, but **in-sample**: it predicts on
the same rows the models were fitted on, so the figure it prints measures
memorisation.  Bypassing the blocker produced +545.69% at a 97% win rate, which
is what fitting and testing on one dataset looks like.

This module pins the three seams that make an honest measurement possible, and
the two reporting rules that keep a number from being read as more than it is:

* ``train.filter_as_of`` — a cutoff that removes every row **after** it from
  training, and ``select_global_feature_schema``'s cutoff: the shared feature
  contract is also chosen from pre-cutoff rows only.
* ``model_store``'s model-root override — a holdout fleet can live beside the
  production one.  The default path must not move: every consumer
  (``serve.py``, ``predict.py``, ``cli.py``, ``backtest.py``) resolves models
  through this module, so an override that changes the default would silently
  repoint the API.
* ``backtest``'s skip report (R3) — the 100-row guard applies to *windowed*
  rows, so a short window can silently shrink the measured universe.  The run
  now names every ticker it could not measure and why, and says what universe
  the aggregate is over.
* ``backtest``'s buy-and-hold line (R4) — it used to count
  ``(total_return > 0).sum()``, a profitable-ticker count, under a label that
  read as a benchmark.  It is now a real close-to-close comparison.

Everything here is offline and fast.  No test reads ``data/processed``, fits a
production model, or gates the suite on the ~16-minute holdout retrain: the
seams are tested, not the fleet.
"""
from __future__ import annotations

import logging
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

# ``src/backtest.py`` imports mlflow, which imports mlflow.tracing, which imports
# opentelemetry, whose runtime-context loader still calls the removed
# ``importlib.metadata`` dict interface.  On Python 3.11 that emits
# ``DeprecationWarning: SelectableGroups dict interface is deprecated`` from
# opentelemetry — upstream, not this repository — and the suite runs with
# ``filterwarnings = ["error"]``, so without a filter the import aborts
# collection.  The block is scoped to that one message and category and is
# exited before any test runs, so every other warning still fails the run.  It
# is the same arrangement ``tests/test_backtest_contract.py`` documents.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="SelectableGroups dict interface is deprecated",
        category=DeprecationWarning,
    )
    import config
    import model_store
    import train

    import backtest

#: The repository's own artefacts, named from this file rather than from the
#: process's working directory.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Last business day of the synthetic frames below.
END_DATE = "2024-12-31"

#: The experiment name ``train`` imported with, restored after every test below.
_DEFAULT_EXPERIMENT_NAME = train.EXPERIMENT_NAME


@pytest.fixture(autouse=True)
def _isolate_the_store_override():
    """No test here may leak a model root or an experiment name into the suite.

    ``set_model_root`` and ``configure_fleet`` deliberately write process-wide
    environment state, so a test that uses them must clean up even when it
    fails.  ``monkeypatch.delenv`` on an absent variable records nothing to
    restore, which is how the holdout root once escaped into
    ``tests/test_model_store.py`` and failed ``model_dir()``'s default-path pin;
    this fixture snapshots and restores the two variables around every test.
    """
    keys = (model_store.MODEL_ROOT_ENV, train.EXPERIMENT_ENV)
    saved_env = {key: os.environ.get(key) for key in keys}
    saved_experiment = train.EXPERIMENT_NAME
    for key in keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        train.EXPERIMENT_NAME = saved_experiment


def _business_days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(end=END_DATE, periods=n)


def _priced_frame(ticker: str, n: int, start: float = 100.0,
                  end: float | None = None) -> pd.DataFrame:
    """One ticker's daily frame: a Close price (linear) plus the columns
    ``prepare_xy`` drops, which is all ``backtest_ticker`` needs."""
    close = np.linspace(start, start if end is None else end, n)
    return pd.DataFrame(
        {"Close": close, "ticker": ticker, "target": 0}, index=_business_days(n)
    )


class _ConstantModel:
    """A fake estimator that always answers the same probability."""

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.rows_seen = 0

    def predict_proba(self, frame):
        self.rows_seen += len(frame)
        return np.array([[1 - self.probability, self.probability]] * len(frame))


class _FakeClient:
    """Stands in for ``mlflow.MlflowClient`` — records experiment creation."""

    def __init__(self, existing=None) -> None:
        self.existing = existing
        self.created: list[tuple[str, str | None]] = []

    def get_experiment_by_name(self, name):
        return self.existing

    def create_experiment(self, name, artifact_location=None):
        self.created.append((name, artifact_location))
        return "7"


class _FakeMlflow:
    """Records what ``train``/``run_backtest`` log, and never touches a store."""

    def __init__(self, client=None) -> None:
        self.client = client or _FakeClient()
        self.params: dict = {}
        self.metrics: dict = {}
        self.tags: dict = {}
        self.runs: list[str] = []

    # ``train`` uses these.
    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None, **kwargs):
        self.runs.append(run_name)
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

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_artifact(self, path, artifact_path=None):
        pass

    # ``configure_fleet`` asks for a client.
    def MlflowClient(self):
        return self.client


class _TrainFakeMlflow(_FakeMlflow):
    """The subset ``train()`` calls; no client required."""


# ═══════════════════════════════════════════════════════════════════════════════
# The model-root override — a holdout fleet beside the production one (R1)
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_default_model_root_is_todays_production_fleet(monkeypatch):
    """With no override, the path is byte-for-byte the pre-task path.

    R1: every consumer resolves models through ``model_store``, so the default
    branch is the one thing that must not move.
    """
    monkeypatch.delenv(model_store.MODEL_ROOT_ENV, raising=False)

    assert model_store.model_dir() == (
        REPO_ROOT / config.MODEL_DIR / model_store.EXPERIMENT_ID / "models"
    ).resolve()
    assert model_store.index_path() == model_store.model_dir() / "model_index.json"


def test_the_override_moves_the_store_and_its_index(monkeypatch, tmp_path):
    monkeypatch.delenv(model_store.MODEL_ROOT_ENV, raising=False)
    fleet = tmp_path / "holdout" / "models"

    model_store.set_model_root(fleet)

    assert model_store.model_dir() == fleet.resolve()
    assert model_store.index_path() == fleet.resolve() / "model_index.json"

    model_store.set_model_root(None)
    assert model_store.model_dir() == (
        REPO_ROOT / config.MODEL_DIR / model_store.EXPERIMENT_ID / "models"
    ).resolve()


def test_a_relative_override_is_resolved_from_the_repository_not_the_cwd(
    monkeypatch, tmp_path
):
    """The store must not depend on where the process happens to run from.

    ``model_dir`` derives the default from ``model_store``'s own location for
    exactly this reason; the override follows the same rule.
    """
    monkeypatch.delenv(model_store.MODEL_ROOT_ENV, raising=False)
    monkeypatch.setenv(model_store.MODEL_ROOT_ENV, "mlruns/holdout-2025-06-12/models")
    monkeypatch.chdir(tmp_path)

    assert model_store.model_dir() == (
        REPO_ROOT / "mlruns" / "holdout-2025-06-12" / "models"
    ).resolve()


def test_the_override_is_an_environment_read_not_a_second_default(monkeypatch, tmp_path):
    """An empty override means "no override", never a store called ``""``."""
    monkeypatch.setenv(model_store.MODEL_ROOT_ENV, "")
    assert model_store.model_dir() == (
        REPO_ROOT / config.MODEL_DIR / model_store.EXPERIMENT_ID / "models"
    ).resolve()


def test_writing_a_holdout_index_leaves_the_production_index_byte_unchanged(
    monkeypatch, tmp_path
):
    """R1's proof at the seam: a holdout write must not touch the tracked index.

    The production index is the single file under ``mlruns/`` tracked in git and
    the one every prediction reads.  A holdout fleet writes its own index beside
    itself; the production one is asserted byte-for-byte, both while the
    override is active and after it is cleared.
    """
    monkeypatch.delenv(model_store.MODEL_ROOT_ENV, raising=False)
    production_index = model_store.index_path()
    before = production_index.read_bytes() if production_index.exists() else None

    fleet = tmp_path / "holdout" / "models"
    model_store.set_model_root(fleet)
    model_store.write_model_index({"BBCA.JK": "m-holdout"})

    assert (fleet / "model_index.json").is_file(), "the holdout index is not beside its fleet"
    assert model_store.load_model_index() == {"BBCA.JK": "m-holdout"}

    model_store.set_model_root(None)
    assert model_store.index_path() == production_index
    assert (production_index.read_bytes() if production_index.exists() else None) == before, (
        "writing a holdout index changed the production index"
    )


def test_the_production_fleet_directory_is_not_the_holdout_directory():
    """A sanity pin on the two paths, so a report cannot confuse the fleets."""
    production = (
        REPO_ROOT / config.MODEL_DIR / model_store.EXPERIMENT_ID / "models"
    ).resolve()
    holdout = (REPO_ROOT / "mlruns" / "holdout-2025-06-12" / "models").resolve()
    assert production != holdout
    assert holdout != production and production not in holdout.parents and holdout not in production.parents


# ── Training a fleet into its own root ────────────────────────────────────────

def test_configuring_a_fleet_points_the_index_and_the_experiment_at_one_directory(
    monkeypatch, tmp_path
):
    """A holdout run's index and its MLflow artifacts must name one directory.

    If they disagree, the index names folders that exist only in the other
    fleet — the measurement then silently measures nothing (or the production
    fleet).  The experiment is created with the fleet directory as its artifact
    location, so ``<root>/models/<model_id>`` is where the model is logged.
    """
    monkeypatch.delenv(model_store.MODEL_ROOT_ENV, raising=False)
    monkeypatch.delenv(train.EXPERIMENT_ENV, raising=False)
    monkeypatch.setattr(train, "EXPERIMENT_NAME", train.EXPERIMENT_NAME)
    fake = _FakeMlflow()
    monkeypatch.setattr(train, "mlflow", fake)

    train.configure_fleet("mlruns/holdout-2025-06-12/models", "idx-holdout")

    assert model_store.model_dir() == (
        REPO_ROOT / "mlruns" / "holdout-2025-06-12" / "models"
    ).resolve()
    assert fake.client.created == [
        ("idx-holdout", (REPO_ROOT / "mlruns" / "holdout-2025-06-12").as_uri())
    ]
    assert train.EXPERIMENT_NAME == "idx-holdout"
    assert os.environ[train.EXPERIMENT_ENV] == "idx-holdout"


def test_configuring_no_fleet_leaves_the_production_path_and_experiment_alone(
    monkeypatch
):
    monkeypatch.delenv(model_store.MODEL_ROOT_ENV, raising=False)
    monkeypatch.delenv(train.EXPERIMENT_ENV, raising=False)
    monkeypatch.setattr(train, "EXPERIMENT_NAME", train.EXPERIMENT_NAME)
    fake = _FakeMlflow()
    monkeypatch.setattr(train, "mlflow", fake)

    train.configure_fleet()

    assert model_store.model_dir() == (
        REPO_ROOT / config.MODEL_DIR / model_store.EXPERIMENT_ID / "models"
    ).resolve()
    assert train.EXPERIMENT_NAME == config.MLFLOW_EXPERIMENT
    assert fake.client.created == []


@pytest.mark.parametrize(
    "model_root, experiment",
    [("holdout/models", None), (None, "idx-holdout")],
)
def test_a_fleet_root_without_an_experiment_is_refused(monkeypatch, model_root, experiment):
    """Either half alone displaces something: the fleet or the index.

    ``--model-root`` alone would log models into the production experiment while
    writing the index into the holdout root (an index naming models that are not
    there); ``--experiment`` alone would log them elsewhere while writing the
    *production* index.  Both must be given together.
    """
    monkeypatch.delenv(model_store.MODEL_ROOT_ENV, raising=False)
    monkeypatch.delenv(train.EXPERIMENT_ENV, raising=False)
    monkeypatch.setattr(train, "mlflow", _FakeMlflow())

    with pytest.raises(ValueError, match="together"):
        train.configure_fleet(model_root, experiment)


def test_an_existing_experiment_is_reused_not_recreated(monkeypatch):
    monkeypatch.delenv(model_store.MODEL_ROOT_ENV, raising=False)
    monkeypatch.delenv(train.EXPERIMENT_ENV, raising=False)
    monkeypatch.setattr(train, "EXPERIMENT_NAME", train.EXPERIMENT_NAME)
    existing = object()
    fake = _FakeMlflow(client=_FakeClient(existing=existing))
    monkeypatch.setattr(train, "mlflow", fake)

    train.configure_fleet("mlruns/holdout-2025-06-12/models", "idx-holdout")

    assert fake.client.created == [], "an existing experiment must not be recreated"


def test_a_deleted_experiment_is_refused_not_silently_reused(monkeypatch):
    """A deleted experiment keeps its name; logging into it fails late and obscurely.

    The probe hit exactly this: after the first probe experiment was deleted,
    ``get_experiment_by_name`` still returned it, ``mlflow.set_experiment``
    raised inside ``train()``, and the run reported "no model was logged" — a
    misleading message for a name collision.  The refusal names the cause.
    """
    deleted = type("Deleted", (), {"lifecycle_stage": "deleted", "artifact_location": "x"})()
    fake = _FakeMlflow(client=_FakeClient(existing=deleted))
    monkeypatch.setattr(train, "mlflow", fake)

    with pytest.raises(ValueError, match="deleted"):
        train.configure_fleet("mlruns/holdout-2025-06-12/models", "idx-holdout")
    assert fake.client.created == [], "a deleted experiment must not be written to"


def test_an_index_naming_folders_absent_from_the_root_is_reported(monkeypatch, tmp_path):
    """The post-run guard: every entry must name a folder the fleet directory has.

    This is the failure the override makes possible — a fleet logged to one
    directory and an index written to another — and it must be loud, because the
    measurement would otherwise report "45 trained" over a fleet that cannot be
    loaded.
    """
    monkeypatch.delenv(model_store.MODEL_ROOT_ENV, raising=False)
    root = tmp_path / "models"
    (root / "m-present" / "artifacts").mkdir(parents=True)
    (root / "m-present" / "artifacts" / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")

    missing = train.missing_model_folders(
        {"BBCA.JK": "m-present", "BBRI.JK": "m-absent"}, root=root
    )

    assert missing == ["BBRI.JK"]


# ═══════════════════════════════════════════════════════════════════════════════
# The cutoff — training never sees a row after it
# ═══════════════════════════════════════════════════════════════════════════════

def _dated_frame(n: int = 300) -> pd.DataFrame:
    return pd.DataFrame({"Close": np.arange(n, dtype=float)}, index=_business_days(n))


def test_the_cutoff_keeps_rows_at_or_before_it():
    frame = _dated_frame(10)
    cutoff = frame.index[5]

    bounded = train.filter_as_of(frame, cutoff)

    assert len(bounded) == 6, "rows at or before the cutoff are kept, inclusively"
    assert bounded.index.max() == cutoff
    assert list(bounded.index) == list(frame.index[:6])


def test_no_cutoff_leaves_every_row():
    frame = _dated_frame(10)

    assert train.filter_as_of(frame).equals(frame), "the default path must not filter"


def test_a_cutoff_that_selects_no_rows_is_refused_loudly():
    frame = _dated_frame(10)

    with pytest.raises(ValueError) as excinfo:
        train.filter_as_of(frame, "2020-01-01")

    message = str(excinfo.value)
    assert "2020-01-01" in message
    assert "no rows" in message
    assert "2024" in message, "the refusal must name the range the frame covers"


def test_a_malformed_cutoff_is_refused():
    with pytest.raises(ValueError, match="as-of"):
        train.filter_as_of(_dated_frame(5), "not-a-date")


def _synthetic_pool(n_rows: int = 300) -> pd.DataFrame:
    """A pooled frame with a real date index, six numeric features, one ticker."""
    rng = np.random.default_rng(20261013)
    data = {f"feat_{i:03d}": rng.normal(size=n_rows) for i in range(6)}
    data["target"] = (rng.random(n_rows) > 0.5).astype(int)
    data["ticker"] = "BBCA.JK"
    frame = pd.DataFrame(data, index=_business_days(n_rows))
    return frame


def _prepare_training(monkeypatch, frame):
    """Point ``train`` at a synthetic frame and a recording MLflow."""
    fake = _TrainFakeMlflow()
    monkeypatch.setattr(train, "load_features", lambda path: frame)
    monkeypatch.setattr(train, "mlflow", fake)
    monkeypatch.setattr(train, "IS_CI", True)  # metrics only: no model, no plots
    return fake


@pytest.mark.parametrize("cutoff_offset, expected_rows", [(None, 300), (199, 200)])
def test_train_fits_only_the_rows_at_or_before_the_cutoff(
    monkeypatch, cutoff_offset, expected_rows
):
    """The whole point: the fitting frame never contains a post-cutoff row.

    ``n_rows`` is what the run reports, and the CV frame is what the estimator
    actually sees; both are asserted against the cutoff.
    """
    frame = _synthetic_pool()
    cutoff = None if cutoff_offset is None else frame.index[cutoff_offset]
    fake = _prepare_training(monkeypatch, frame)
    seen: list[pd.DataFrame] = []

    def _splits(X, n_splits=5, gap=10):
        seen.append(X.copy())
        n = len(X)
        return [(np.arange(n // 2), np.arange(n // 2, n))]

    monkeypatch.setattr(train, "_expanding_cv_splits", _splits)

    model_folder = train.train(
        "BBCA.JK", feature_columns=["feat_000"], blend_lgbm=False, as_of=cutoff
    )

    assert model_folder is None, "CI mode logs no model"
    assert fake.params["n_rows"] == expected_rows
    assert seen, "training never built a CV split"
    assert len(seen[0]) == expected_rows
    if cutoff is not None:
        assert seen[0].index.max() <= cutoff


def test_train_logs_to_the_runs_experiment_not_the_production_constant(monkeypatch):
    """Regression: ``train()`` must not hardcode the production experiment.

    The first probe of the holdout retrain logged its model into experiment 1 —
    the production one — because ``train()`` called
    ``mlflow.set_experiment(MLFLOW_EXPERIMENT)`` while the fleet configuration
    only changed the module global.  The model then landed in
    ``mlruns/1/models``, which is exactly what R1 forbids.  The post-run guard
    caught it; this test keeps the call site honest.
    """
    frame = _synthetic_pool()
    fake = _prepare_training(monkeypatch, frame)
    monkeypatch.setattr(train, "EXPERIMENT_NAME", "idx-holdout")

    train.train("BBCA.JK", feature_columns=["feat_000"], blend_lgbm=False)

    assert fake.experiment == "idx-holdout"


def test_the_shared_schema_is_selected_from_pre_cutoff_rows_only(monkeypatch):
    """The feature contract is part of training, so it takes the cutoff too.

    Choosing the 60 columns over rows that include the evaluation window would
    leak the future into the contract even when every fit is truncated.
    """
    frame = _synthetic_pool()
    cutoff = frame.index[199]
    seen: list[pd.DataFrame] = []

    def _select(X, y, top_k=60):
        seen.append(X.copy())
        return X[["feat_000"]]

    monkeypatch.setattr(train, "load_features", lambda path: frame)
    monkeypatch.setattr(train, "_select_features_shap", _select)
    monkeypatch.setattr(train, "_GLOBAL_SCHEMA_CACHE", {})

    columns = train.select_global_feature_schema("pooled.parquet", as_of=cutoff)

    assert columns == ["feat_000"]
    assert len(seen[0]) == 200, "the selection saw rows after the cutoff"
    assert seen[0].index.max() <= cutoff

    # A different cutoff is a different selection, not a cache hit.
    train.select_global_feature_schema("pooled.parquet", as_of=frame.index[99])
    assert len(seen) == 2, "the schema cache ignored the cutoff"
    assert len(seen[1]) == 100


@pytest.mark.parametrize("task, as_of", [
    (("BBCA.JK", False, "features.parquet", ["feat_000"]), None),
    (("BBCA.JK", False, "features.parquet", ["feat_000"], "2025-06-12"), "2025-06-12"),
])
def test_the_worker_carries_the_cutoff_across_the_process_boundary(
    monkeypatch, task, as_of
):
    """``_train_one`` runs in a loky worker, so the cutoff must travel in the task.

    The four-element form is kept working: a caller that predates the cutoff
    still builds a valid task, and it means "no cutoff".
    """
    recorded: dict = {}

    def _train(ticker, **kwargs):
        recorded.update(kwargs)
        return "m-worker-folder"

    monkeypatch.setattr(train, "train", _train)
    monkeypatch.setattr(train, "FEATURES_PATH", task[2])

    assert train._train_one(task) == ("BBCA.JK", True, "m-worker-folder")
    assert recorded["as_of"] == as_of


# ═══════════════════════════════════════════════════════════════════════════════
# The measurement reads the holdout fleet, not the production one
# ═══════════════════════════════════════════════════════════════════════════════

def _holdout_store(tmp_path, ticker: str = "ZZZZ.JK") -> pathlib.Path:
    """A one-ticker store laid out exactly like a logged fleet."""
    root = (tmp_path / "holdout" / "models").resolve()
    folder = root / "m-holdout" / "artifacts"
    folder.mkdir(parents=True)
    (folder / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
    (folder / "feature_schema.json").write_text('["Close"]\n', encoding="utf-8")
    (root / "model_index.json").write_text(
        '{"ticker_to_model": {"%s": {"model_folder": "m-holdout"}}}' % ticker,
        encoding="utf-8",
    )
    return root


def test_the_backtest_resolves_models_through_the_override(monkeypatch, tmp_path):
    """With ``--model-root`` set, selection reads that fleet and nowhere else."""
    monkeypatch.delenv(model_store.MODEL_ROOT_ENV, raising=False)
    root = _holdout_store(tmp_path)
    loaded: list[str] = []
    model = _ConstantModel(0.6)

    def _load(artifact_dir):
        loaded.append(str(artifact_dir))
        return model

    monkeypatch.setattr(model_store, "_load_xgboost_model", _load)
    monkeypatch.setattr(backtest, "_model_cache", {})
    monkeypatch.setattr(model_store, "_reported_index_diagnostics", set())

    model_store.set_model_root(root)
    try:
        result = backtest.backtest_ticker("ZZZZ.JK", _priced_frame("ZZZZ.JK", 120))
    finally:
        model_store.set_model_root(None)

    assert result is not None, "the holdout model was not resolved"
    assert result["metrics"]["n_trades"] >= 1
    assert loaded == [str(root / "m-holdout" / "artifacts")]
    assert model.rows_seen == 120


def test_without_the_override_the_holdout_ticker_does_not_resolve(monkeypatch, tmp_path):
    """The same run on the production store must not borrow the holdout model.

    ``ZZZZ.JK`` is in no production index, so the ticker is skipped — and the
    skip is recorded, which is the R3 report below.
    """
    monkeypatch.delenv(model_store.MODEL_ROOT_ENV, raising=False)
    _holdout_store(tmp_path)
    monkeypatch.setattr(backtest, "_model_cache", {})
    monkeypatch.setattr(model_store, "_reported_index_diagnostics", set())

    reasons: dict[str, str] = {}
    result = backtest.backtest_ticker(
        "ZZZZ.JK", _priced_frame("ZZZZ.JK", 120), skip_reasons=reasons
    )

    assert result is None
    assert "no model" in reasons["ZZZZ.JK"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# R3 — the run names the universe it actually measured
# ═══════════════════════════════════════════════════════════════════════════════

def _run(monkeypatch, tmp_path, frame, tickers, model):
    monkeypatch.setattr(backtest, "load_eval_frame", lambda **kwargs: frame)
    monkeypatch.setattr(backtest, "mlflow", _FakeMlflow())
    monkeypatch.setattr(backtest, "_model_cache", {})

    def _load(ticker):
        if ticker == "NOMODEL.JK":
            raise model_store.ModelNotFound("no model is mapped to it")
        return model, ["Close"]

    monkeypatch.setattr(backtest, "load_model_for", _load)
    monkeypatch.chdir(tmp_path)
    return backtest.run_backtest(tickers=tickers)


def test_a_run_names_every_ticker_it_could_not_measure_and_why(
    monkeypatch, caplog, tmp_path
):
    """The row guard shrinks the universe silently; the run must say so.

    ``SHORT.JK`` has fewer than ``min_rows`` rows *in the window* (the guard
    applies to windowed rows, Task 3.2), and ``NOMODEL.JK`` has no model.  Both
    are named with their reason, and the aggregate is stated to be over one
    ticker, not three.
    """
    frame = pd.concat([
        _priced_frame("SHORT.JK", 3),
        _priced_frame("FULL.JK", 120),
        _priced_frame("NOMODEL.JK", 120),
    ]).sort_index()
    caplog.set_level(logging.INFO)

    results = _run(
        monkeypatch, tmp_path, frame,
        ["SHORT.JK", "FULL.JK", "NOMODEL.JK"], _ConstantModel(0.6),
    )

    assert list(results["ticker"]) == ["FULL.JK"]
    skipped = results.attrs["skipped"]
    assert set(skipped) == {"SHORT.JK", "NOMODEL.JK"}
    assert "only 3 row" in skipped["SHORT.JK"]
    assert "no model" in skipped["NOMODEL.JK"].lower()

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Measured universe: 1 of 3" in text
    assert "SHORT.JK" in text and "NOMODEL.JK" in text
    assert any(
        record.levelname == "WARNING" and "shrank" in record.getMessage().lower()
        for record in caplog.records
    ), "a shrunken universe must be visible at warning"


def test_a_full_universe_reports_no_skips(monkeypatch, caplog, tmp_path):
    frame = _priced_frame("FULL.JK", 120)
    caplog.set_level(logging.INFO)

    results = _run(monkeypatch, tmp_path, frame, ["FULL.JK"], _ConstantModel(0.6))

    assert len(results) == 1
    assert results.attrs["skipped"] == {}
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Measured universe: 1 of 1" in text


# ═══════════════════════════════════════════════════════════════════════════════
# R4 — buy-and-hold is a benchmark, not a count of positive returns
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_buy_hold_return_is_the_windows_price_change(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest, "_model_cache", {})
    monkeypatch.setattr(model_store, "_reported_index_diagnostics", set())
    monkeypatch.setattr(backtest, "load_model_for", lambda ticker: (_ConstantModel(0.4), ["Close"]))

    result = backtest.backtest_ticker(
        "FALL.JK", _priced_frame("FALL.JK", 120, start=100.0, end=90.0)
    )

    metrics = result["metrics"]
    assert metrics["n_trades"] == 0, "a model that never says BUY must not trade"
    assert metrics["total_return_pct"] == 0.0
    assert metrics["buy_hold_return_pct"] == pytest.approx(-10.0, abs=0.05)
    assert metrics["beats_buy_hold"] is True


def test_the_summary_counter_compares_to_buy_hold_not_to_zero(
    monkeypatch, caplog, tmp_path
):
    """The old line counted ``total_return > 0``; here that count is 0 and the
    honest comparison is 1 (flat cash beat a 10% fall).  A regression to the old
    expression reports 0/1 and fails this test.
    """
    frame = _priced_frame("FALL.JK", 120, start=100.0, end=90.0)
    caplog.set_level(logging.INFO)

    results = _run(monkeypatch, tmp_path, frame, ["FALL.JK"], _ConstantModel(0.4))

    assert bool(results["beats_buy_hold"].iloc[0]) is True
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Tickers beating buy-hold: 1/1" in text
