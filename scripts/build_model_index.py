#!/usr/bin/env python
"""Rebuild the ticker-to-model index from the logged models' run metadata.

Scans every logged model under ``mlruns/1/models/``, reads the ``run_id`` from
each ``MLmodel``, asks MLflow which ticker that run was for, and writes the
mapping through ``model_store.write_model_index`` — the index's one writer, so
this command and training produce the same file shape and share the same
"never write an empty index" guard.

A model whose ticker cannot be resolved from MLflow metadata is reported at
``warning``.  The script **exits non-zero when a ticker in the universe is left
without a model** — never merely because a logged-model folder from an earlier
retrain could not be resolved, since those folders outlive the run ids a fresh
tracking DB keeps and can never be resolved by anything this command does.  It
never falls back to guessing a ticker from the model's own feature names against
the features CSV: that guess invented ticker identity, which is how the committed
index came to name models training never associated with those tickers (F10).
Retraining is the remedy — training writes the index itself now.

Usage:
    python scripts/build_model_index.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import mlflow
import yaml

import model_store
from config import MLFLOW_DB, TICKERS, get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: MLflow's run metadata lives in the sqlite tracking store.  Without this URI
#: ``MlflowClient()`` resolves the *file* store (``./mlruns``) and every
#: ``get_run`` fails — the failure the old bare ``except Exception: pass`` hid,
#: which is why the script could resolve nothing and guessed instead.
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, MLFLOW_DB)}"

#: The run name training gives a ticker's run (``src/train.py``).
RUN_NAME_PREFIX = "xgb_"

#: How many unresolved folders to list before the tail is summarised.
REPORT_LIMIT = 5


def ticker_from_run(client, run_id: str) -> str:
    """The ticker MLflow recorded for ``run_id``, or ``""`` when it has none.

    Read from the run's name and tags/params, exactly as training wrote them.
    Raises whatever ``get_run`` raises — the caller reports it and counts the
    model as unresolved rather than continuing quietly.
    """
    run = client.get_run(run_id)

    run_name = run.data.tags.get("mlflow.runName", "")
    if run_name.startswith(RUN_NAME_PREFIX):
        ticker = run_name[len(RUN_NAME_PREFIX):]
        if ticker:
            return ticker

    for source in (run.data.tags, run.data.params):
        ticker = source.get("ticker", "")
        if ticker:
            return ticker
    return ""


def build_index() -> int:
    """Write the index and return the process exit status.

    Returns:
        ``0`` when every ticker in the universe is mapped; ``1`` when any ticker
        in the universe is left without a model.

    **The criterion is the universe, not the folders on disk.**  Training never
    prunes a logged model's folder, so a retrain leaves the previous store beside
    the new one: those folders keep run ids that a later tracking DB does not
    have, and no rebuild can ever resolve them.  Judging this command by "every
    folder resolved" therefore made every post-retrain rebuild exit ``1`` on a
    healthy index — and ``start.bat``'s weekly path treats a non-zero exit as a
    failed step, so the container restart that publishes the new index never ran.
    Unresolvable folders are reported at ``warning`` and do not fail the command;
    a ticker with no model still does.
    """
    models_dir = model_store.model_dir()
    folders = sorted(path.name for path in models_dir.iterdir() if path.is_dir())
    print(f"Scanning {len(folders)} model folder(s) under {models_dir}...")

    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("Resolving runs through %s", MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    entries: dict[str, str] = {}
    unresolved: list[tuple[str, str]] = []
    errors = 0
    total = 0

    for folder in folders:
        mlmodel_file = models_dir / folder / "artifacts" / "MLmodel"
        if not mlmodel_file.is_file():
            continue

        total += 1
        run_id = ""
        try:
            with open(mlmodel_file, encoding="utf-8") as handle:
                run_id = (yaml.safe_load(handle) or {}).get("run_id", "") or ""
            ticker = ticker_from_run(client, run_id) if run_id else ""
            if not ticker:
                reason = (
                    f"run {run_id} records no ticker in its run name, tags or params"
                    if run_id else "its MLmodel records no run_id"
                )
            elif ticker not in TICKERS:
                reason = (
                    f"run {run_id} records ticker {ticker!r}, which is not in the "
                    f"{len(TICKERS)}-ticker universe"
                )
                ticker = ""
            else:
                reason = ""
        except Exception as exc:
            errors += 1
            ticker = ""
            reason = f"MLflow could not be read for run {run_id or '<none>'}: {exc}"

        if ticker:
            entries[ticker] = folder
            continue

        unresolved.append((folder, reason))
        logger.warning("%s: %s — left unresolved", folder, reason)

    # The failure this command exists to report, counted against the thing the
    # index has to serve: a ticker in the universe with no model to resolve.
    missing = [ticker for ticker in TICKERS if ticker not in entries]

    print(
        f"Resolved {len(entries)}/{len(TICKERS)} ticker(s) in the universe "
        f"from {total} logged model folder(s)"
    )

    # The shared writer, not a local json.dump: one file shape, and the guard
    # that refuses to empty a populated index (R3) applies to this command too.
    # With nothing resolved it writes nothing at all, so a failed rebuild cannot
    # destroy an index that is still serving predictions.
    model_store.write_model_index(entries, total_models=total, errors=errors)

    if unresolved:
        # Stale folders from an earlier retrain, a run whose metadata cannot be
        # read, or a run for a ticker outside the universe.  Reported, and
        # reported loudly, but never the reason to fail: they cannot be resolved
        # by anything this command does.
        print(
            f"\n{len(unresolved)} logged model folder(s) resolved to no ticker in "
            f"the universe — stale folders from earlier runs, or unreadable run "
            f"metadata.  They do not fail this command:"
        )
        for folder, reason in unresolved[:REPORT_LIMIT]:
            print(f"  {folder}: {reason}")
        if len(unresolved) > REPORT_LIMIT:
            print(f"  ... and {len(unresolved) - REPORT_LIMIT} more (see the warnings above)")

    if not missing:
        print(
            f"\nEvery one of the {len(TICKERS)} ticker(s) in the universe resolves "
            f"to a model; the index was written"
        )
        return 0

    print(
        f"\n{len(missing)} of {len(TICKERS)} ticker(s) in the universe have no "
        f"model the index can name:"
    )
    for ticker in missing[:REPORT_LIMIT]:
        print(f"  {ticker}")
    if len(missing) > REPORT_LIMIT:
        print(f"  ... and {len(missing) - REPORT_LIMIT} more")

    print(
        "\nGuessing a ticker from the model's own feature names is gone: a model "
        "whose ticker MLflow does not record cannot be mapped.\n"
        "Remedy: retrain, so training writes the index itself —\n"
        "    python src/train.py --parallel --features data/processed/features.csv"
    )
    return 1


if __name__ == '__main__':
    sys.exit(build_index())
