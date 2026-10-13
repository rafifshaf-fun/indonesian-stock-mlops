"""The index rebuild's exit criterion, and the one reader outside ``model_store``.

``scripts/build_model_index.py`` scans the logged models on disk and writes the
index through ``model_store.write_model_index``.  Two consequences of that scan
are pinned here.

**The exit code is scoped to the universe, not to the store.**  Training never
prunes a logged-model folder, so a retrain takes the store from 45 folders to
90; the 45 folders in git keep run ids that a fresh tracking DB does not have,
and *always will*.  Judging the rebuild by "every folder resolved" therefore made
every post-retrain rebuild exit 1 on a healthy store — and ``start.bat``'s weekly
path treats a non-zero exit as a failed step, so the API container would never be
recreated after the retrain that populated the index.  Failure is now counted
against ``config.TICKERS``: an unmapped universe still exits 1 (today's state),
while stale folders are reported at ``warning`` and do not fail the command.

**``scripts/link_models.py`` reads both entry shapes.**  F10 changed an entry
from ``{"model_folder": ..., "match_score": ..., "n_features": ...}`` to the bare
folder name, because the two extra fields existed only to serve the deleted
feature-overlap guess.  That script is the only reader outside ``model_store``
and it still indexed ``info['model_folder']``, which raises
``TypeError: string indices must be integers`` on the new shape.

Both scripts are loaded by path: ``scripts/`` is not on ``pythonpath`` and the
application never imports them.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import pytest

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"

# ``scripts/build_model_index.py`` imports mlflow, which imports opentelemetry,
# whose runtime-context loader still calls the removed ``importlib.metadata``
# dict interface on Python 3.11 — a ``DeprecationWarning`` from upstream, and
# ``filterwarnings = ["error"]`` turns it into a collection error.  Same narrow
# filter, same reason, as ``tests/test_index_written.py`` and
# ``tests/test_feature_schema.py``.
pytestmark = pytest.mark.filterwarnings(
    "ignore:SelectableGroups dict interface is deprecated:DeprecationWarning"
)


def _load_script(name: str):
    """Import ``scripts/<name>.py`` by path, without touching ``sys.path``."""
    spec = importlib.util.spec_from_file_location(
        f"_rescue_script_{name}", SCRIPTS_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ═══════════════════════════════════════════════════════════════════════════════
# The rebuild's exit code — counted against the universe
# ═══════════════════════════════════════════════════════════════════════════════

class _FakeRun:
    """The two fields ``ticker_from_run`` reads."""

    def __init__(self, ticker: str):
        self.data = types.SimpleNamespace(
            tags={"mlflow.runName": f"xgb_{ticker}"}, params={}
        )


class _FakeClient:
    """A tracking client that knows only the run ids it was given."""

    def __init__(self, known: dict[str, str]):
        self._known = known
        self.asked: list[str] = []

    def get_run(self, run_id: str):
        self.asked.append(run_id)
        if run_id in self._known:
            return _FakeRun(self._known[run_id])
        raise RuntimeError(f"Run with id={run_id} not found")


def _logged_model(root: pathlib.Path, folder: str, run_id: str) -> None:
    """A logged-model folder exactly as MLflow lays one out."""
    artifacts = root / folder / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "MLmodel").write_text(f"run_id: {run_id}\n", encoding="utf-8")


@pytest.fixture
def rebuild(tmp_path, monkeypatch):
    """``build_model_index`` with its store and its tracking client redirected.

    ``model_store.model_dir`` is redirected too, so the writer it calls lands in
    ``tmp_path`` (``index_path()`` is derived from it) and the tracked
    ``mlruns/1/models/model_index.json`` cannot be touched by these tests.
    """
    import model_store

    module = _load_script("build_model_index")
    monkeypatch.setattr(model_store, "model_dir", lambda: tmp_path)
    # The script assigns the environment directly; pin the variable so teardown
    # restores whatever the suite had before.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///<redirected>")
    return module


def _install_client(monkeypatch, module, known: dict[str, str]) -> _FakeClient:
    client = _FakeClient(known)
    monkeypatch.setattr(module, "mlflow", types.SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        tracking=types.SimpleNamespace(MlflowClient=lambda: client),
    ))
    return client


def test_an_unmapped_universe_still_exits_one(rebuild, monkeypatch, tmp_path, caplog):
    """Today's artefacts: 45 folders, none resolvable, exit 1.

    The count that matters is ``0 of 45 tickers`` — expressed against the
    universe, which is what the command is judged on — not ``45 of 45 folders``.
    """
    for i in range(5):
        _logged_model(tmp_path, f"m-{i:04d}", f"run-{i}")
    _install_client(monkeypatch, rebuild, {})

    with caplog.at_level("WARNING"):
        status = rebuild.build_index()

    assert status == 1, "an index that maps no ticker is a failed rebuild"
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert sum("left unresolved" in message for message in warnings) == 5, warnings
    assert not (tmp_path / "model_index.json").exists(), \
        "a rebuild resolving nothing must not write an index"


def test_a_fully_mapped_universe_exits_zero_and_only_warns_about_stale_folders(
        rebuild, monkeypatch, tmp_path, caplog):
    """After a retrain: every ticker resolves, the stale folders are just noise.

    This is the state the fix exists for.  A retrain adds 45 folders to a store
    that already has 45 committed ones whose runs are absent from the tracking
    DB; those stale folders are reported per folder at ``warning``, and the
    command still exits 0, so ``start.bat`` reaches its container restart.
    """
    import config

    known: dict[str, str] = {}
    for i, ticker in enumerate(config.TICKERS):
        _logged_model(tmp_path, f"m-{i:04d}", f"run-{i}")
        known[f"run-{i}"] = ticker

    # Folders left behind by the previous store: their runs are not in the DB.
    for j in range(3):
        _logged_model(tmp_path, f"m-stale{j}", f"gone-{j}")
    # And one whose run resolves to a ticker outside the universe: the rebuild
    # must not map it, which is unresolved but not a stale folder.
    _logged_model(tmp_path, "m-outside", "run-outside")
    known["run-outside"] = "NOTA.JK"

    _install_client(monkeypatch, rebuild, known)

    with caplog.at_level("WARNING"):
        status = rebuild.build_index()

    assert status == 0, "a fully mapped universe is a healthy store, not a failure"

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert sum("left unresolved" in message for message in warnings) == 4, warnings
    assert any("NOTA.JK" in message for message in warnings), warnings

    payload = json.loads((tmp_path / "model_index.json").read_text(encoding="utf-8"))
    assert payload["mapped_tickers"] == len(config.TICKERS)
    assert set(payload["ticker_to_model"]) == set(config.TICKERS)
    assert payload["total_models"] == len(config.TICKERS) + 4
    assert payload["indexed_models"] == len(config.TICKERS)


def test_the_universe_count_is_printed_and_the_stale_folders_are_summarised(
        rebuild, monkeypatch, tmp_path, capsys):
    """The output has to say which number failed, or the exit code is a mystery.

    ``0/45`` and ``45 of 45`` are the universe; the folder count is context.
    """
    import config

    _logged_model(tmp_path, "m-a", "run-a")
    _install_client(monkeypatch, rebuild, {})

    assert rebuild.build_index() == 1

    out = capsys.readouterr().out
    assert f"0/{len(config.TICKERS)} ticker(s) in the universe" in out, out
    assert f"{len(config.TICKERS)} of {len(config.TICKERS)} ticker(s) in the universe" in out, out
    assert "1 logged model folder(s)" in out, out


# ═══════════════════════════════════════════════════════════════════════════════
# scripts/link_models.py — tolerant of both entry shapes
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def linker(tmp_path, monkeypatch):
    """``link_models`` with its three directories redirected into ``tmp_path``.

    ``os`` is replaced by a shim whose ``symlink`` always succeeds, so the test
    pins the *reader* on both platforms: on Windows without symlink rights the
    real call takes the copy fallback, which has a pre-existing bug of its own
    (``dst_artifacts`` is never created before the per-file copy) and is not
    what this task changes.
    """
    import os

    module = _load_script("link_models")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    index_file = models_dir / "model_index.json"
    by_ticker = tmp_path / "by_ticker"
    monkeypatch.setattr(module, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(module, "INDEX_FILE", str(index_file))
    monkeypatch.setattr(module, "BY_TICKER_DIR", str(by_ticker))
    monkeypatch.setattr(module, "os", types.SimpleNamespace(
        path=os.path, makedirs=os.makedirs, unlink=os.unlink,
        symlink=lambda *args, **kwargs: None,
    ))
    return module, models_dir, index_file


def test_the_link_script_reads_both_entry_shapes(linker, capsys):
    """A bare folder name and the legacy dict both resolve to a folder.

    F10 changed the entry to the bare string the writer produces.  This script
    is the only reader outside ``model_store`` (which normalises both), and it
    used to index the entry as a dict — raising ``TypeError`` the first time a
    retrain wrote the new shape.
    """
    module, models_dir, index_file = linker
    for folder in ("m-bare", "m-dict"):
        _logged_model(models_dir, folder, "run-1")
    index_file.write_text(json.dumps({"ticker_to_model": {
        "BBCA.JK": "m-bare",
        "BBRI.JK": {"model_folder": "m-dict", "match_score": 0.91, "n_features": 60},
    }}), encoding="utf-8")

    module.link_models()

    out = capsys.readouterr().out
    assert "m-bare" in out, out
    assert "m-dict" in out, out
    assert "Done: 2 tickers linked" in out, out


def test_an_entry_that_names_no_folder_is_skipped_not_a_crash(linker, capsys):
    """A malformed entry is reported and skipped; the other tickers still link."""
    module, models_dir, index_file = linker
    _logged_model(models_dir, "m-bare", "run-1")
    index_file.write_text(json.dumps({"ticker_to_model": {
        "BBCA.JK": "m-bare",
        "TLKM.JK": {"match_score": 0.91},
        "ASII.JK": None,
    }}), encoding="utf-8")

    module.link_models()

    out = capsys.readouterr().out
    assert "m-bare" in out, out
    assert "SKIP TLKM.JK" in out, out
    assert "SKIP ASII.JK" in out, out
    assert "Done: 1 tickers linked" in out, out
