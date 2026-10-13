"""One explicit feature schema per model — the feature contract (F5).

The audit found the 45 logged models carrying 45 *different* 60-name feature
lists (union 132, intersection 1) with nothing checking the column set a model
was fed.  Consumers padded absent names with ``NaN`` — moving a predicted
probability by ~0.10 against the 0.50 boundary — and a change to any feature
module invalidated all 45 models with no failure signal at all.

This module is the single place that contract is defined, persisted and applied:

``save_feature_schema`` / ``load_feature_schema``
    An ordered, duplicate-free list of column names, stored as
    ``feature_schema.json`` beside each logged model.  Order is part of the
    contract: XGBoost is fitted on a positionally-scaled matrix, so a reordered
    frame is a different model input.

``align_to_schema``
    The only place a *computed* frame meets a model's columns.  It raises
    :class:`FeatureSchemaError` for every schema column the frame lacks, drops
    frame columns the schema does not name (serving legitimately computes OHLCV
    and other intermediates the model never sees), and returns exactly the
    schema's columns in the schema's order.  It never pads and never reindexes,
    so no missing feature can be silently converted into ``NaN``.

``candidate_columns``
    Applied when the schema is *derived*, so the contract can never name a
    column serving cannot produce.  See the exclusion table below.

Storage location (R4): the schema is written **beside** ``MLmodel`` in the
logged model's artifact directory, i.e.
``mlruns/1/models/<model_id>/artifacts/feature_schema.json``.  That is exactly
the directory every existing consumer already passes to
``mlflow.xgboost.load_model`` (``src/serve.py:82,241,262`` and
``scripts/build_model_index.py:95``), so reading the contract needs no new
lookup path.  Measured on the committed store: ``model_index.json`` maps a
ticker to a ``model_folder``, and each of those folders is itself an MLflow
logged-model directory (``mlruns/1/models/m-<uuid>/artifacts/MLmodel``), not a
run artifact directory — ``mlruns/1/<run_id>/artifacts/`` holds only the
importance and confusion-matrix PNGs.  The path is also gitignored
(``mlruns/**/artifacts/``), so the contract lives in the store alone.

The mechanism — ``MlflowClient.log_model_artifact`` on the
``ModelInfo.model_id`` returned by ``mlflow.xgboost.log_model`` — was
established by logging a throwaway model into a scratch store and inspecting
the resulting tree, not by assuming an MLflow layout: ``mlflow.log_artifact``
with ``artifact_path="model"`` writes into the *run's* artifact directory
(``<experiment>/<run_id>/artifacts/model/``), which carries no ``MLmodel`` at
all in MLflow 3.x, and ``log_model(extra_files=[...])`` nests the file one
level down in ``<model dir>/extra_files/``.
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
from collections.abc import Mapping
from typing import Iterable, Sequence

import pandas as pd

#: File name of the persisted contract, beside ``MLmodel`` in a model directory.
FEATURE_SCHEMA_FILENAME = "feature_schema.json"


class FeatureSchemaError(RuntimeError):
    """A feature schema is malformed, or a frame does not satisfy it."""


# ═══════════════════════════════════════════════════════════════════════════════
# THE EXCLUSION — families serving cannot reproduce (R2)
# ═══════════════════════════════════════════════════════════════════════════════
# Every entry below is measured against the committed
# ``data/processed/features.parquet`` (191 numeric candidate columns, 45
# tickers).  The schema is derived with these families removed *before* the
# global SHAP selection, so no model can be trained on, or served, a column that
# only exists in the training artefact.  Removing a family here is the only
# mechanism: ``train.py`` reads the already-engineered parquet and has neither
# raw OHLCV nor IHSG data with which to recompute what serving could produce.

#: Column-name prefixes excluded from every schema, with the measured reason.
SERVING_UNPRODUCIBLE_PREFIXES: dict[str, str] = {
    "vp_": (
        "19 columns, live in training (not all-NaN in features.parquet) and "
        "requires 5-minute intraday downloads, so serving cannot reproduce it"
    ),
    "idn_": (
        "10 columns, already dead in training: fetch_idx_fundamentals "
        "(src/features/fetchers.py:162-173) is a no-network stub returning "
        "{field: np.nan} for all ten, so all ten are all-NaN in the parquet — "
        "there is nothing to reproduce"
    ),
}

#: Exact column names excluded from every schema, with the measured reason.
SERVING_UNPRODUCIBLE_COLUMNS: dict[str, str] = {
    "google_trend": (
        "1 column, live in training (98 distinct values spanning 0-100 in "
        "features.parquet) and would be a fabricated constant at inference; "
        "serving passes google_trend=None so the column is not materialised"
    ),
}


def unproducible_reason(column: str) -> str | None:
    """Why ``column`` is excluded from the contract, or ``None`` if it is not.

    Exact-name entries win over prefix entries, and prefixes are matched as
    literal family prefixes (``"vp"`` alone is not a member).
    """
    if column in SERVING_UNPRODUCIBLE_COLUMNS:
        return SERVING_UNPRODUCIBLE_COLUMNS[column]
    for prefix, reason in SERVING_UNPRODUCIBLE_PREFIXES.items():
        if column.startswith(prefix):
            return reason
    return None


def is_unproducible(column: str) -> bool:
    """True when ``column`` belongs to a family serving cannot produce."""
    return unproducible_reason(column) is not None


def candidate_columns(columns: Iterable[str]) -> list[str]:
    """Drop every unproducible family, preserving the input order.

    This is what ``train.py`` applies to the pooled candidate frame before the
    global SHAP selection, so the ordered list it yields is the training
    column set *and* the persisted contract.
    """
    return [c for c in columns if not is_unproducible(c)]


# ═══════════════════════════════════════════════════════════════════════════════
# PERSIST / LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def _validated(columns: Sequence[str], *, source: str = "schema") -> list[str]:
    """Return ``columns`` as a list, or raise ``FeatureSchemaError``.

    A schema is an ordered, duplicate-free list of non-empty strings: a
    duplicate would make ``align_to_schema`` select a frame twice over and a
    blank name would never match a computed column.
    """
    if isinstance(columns, (str, bytes, Mapping)):
        raise FeatureSchemaError(f"{source} must be a sequence of column names")
    try:
        names = list(columns)
    except TypeError as exc:
        raise FeatureSchemaError(f"{source} must be a sequence of column names") from exc
    if not names:
        raise FeatureSchemaError(f"{source} is empty — a model with no features")
    for name in names:
        if not isinstance(name, str) or not name:
            raise FeatureSchemaError(f"{source} contains a non-name entry: {name!r}")
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise FeatureSchemaError(f"{source} repeats column(s): {duplicates}")
    return names


def save_feature_schema(columns: Sequence[str], path: str | os.PathLike) -> None:
    """Persist ``columns`` as JSON at ``path``, order included.

    The file is written only after validation, so a rejected schema never
    leaves a partial contract on disk.
    """
    names = _validated(columns)
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")


def load_feature_schema(path: str | os.PathLike) -> list[str]:
    """Read a persisted contract, preserving its order.

    A missing file raises ``FileNotFoundError`` (an ``OSError``) rather than
    returning an empty schema: an absent contract must not read as "no
    constraints", which is the silent padding this task exists to remove.
    """
    target = pathlib.Path(path)
    raw = target.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise FeatureSchemaError(f"{target} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise FeatureSchemaError(
            f"{target} must hold a JSON list of column names, found "
            f"{type(parsed).__name__}"
        )
    return _validated(parsed, source=f"{target}")


# ═══════════════════════════════════════════════════════════════════════════════
# APPLY
# ═══════════════════════════════════════════════════════════════════════════════

def align_to_schema(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return ``df`` restricted to ``columns``, in ``columns``' exact order.

    Raises:
        FeatureSchemaError: if the frame lacks any column the schema names —
            every absent column is listed — or if ``columns`` is not a valid
            schema.

    Frame columns the schema does not name are dropped silently: the serving
    path computes OHLCV and other intermediates the model never sees, and the
    schema-side guarantee (that the schema names only producible columns) is
    enforced at derivation time by :func:`candidate_columns`, not here.
    """
    names = _validated(columns)
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise FeatureSchemaError(
            f"frame is missing {len(missing)} of {len(names)} schema column(s): "
            f"{missing} (frame carries {len(df.columns)} columns)"
        )
    return df[names]


# ═══════════════════════════════════════════════════════════════════════════════
# PERSIST BESIDE A LOGGED MODEL (R4)
# ═══════════════════════════════════════════════════════════════════════════════

def log_feature_schema(model_id: str, columns: Sequence[str]) -> None:
    """Write the contract beside ``MLmodel`` in a logged model's directory.

    ``mlflow.xgboost.log_model`` creates the logged-model directory itself, so
    the schema cannot be placed there first.  ``MlflowClient.log_model_artifact``
    is called *with the model id MLflow reported* and copies the file to
    ``<model artifact dir>/feature_schema.json`` — the directory
    ``mlflow.xgboost.load_model`` is already given by every consumer.

    A missing ``model_id`` is a ``FeatureSchemaError``: a model without a
    persisted contract is exactly the state this task removes, and a caller
    that cannot supply one must not silently produce an unenforced model.
    """
    if not model_id or not isinstance(model_id, str):
        raise FeatureSchemaError(
            f"cannot persist a feature schema without a logged model id (got {model_id!r})"
        )
    from mlflow import MlflowClient  # local import: keeps this module pandas-only

    with tempfile.TemporaryDirectory(prefix="feature_schema_") as tmp:
        staged = os.path.join(tmp, FEATURE_SCHEMA_FILENAME)
        save_feature_schema(columns, staged)
        MlflowClient().log_model_artifact(model_id, staged)
