"""The one model selector: index in, model plus contract out, or an error.

The audit found two independent, silent substitutions in model selection.

**F2 — one model answered for every ticker.**  The committed
``model_index.json`` maps zero tickers while 45 model folders sit on disk, so
``serve.load_best_model`` fell through to ``sorted(os.listdir(models_dir))[0]``
and served the alphabetically first folder for *any* ticker — including a ticker
that does not exist, which returned a prediction rather than an error.

**F9 — the API and the backtester disagreed by construction.**  The API's
fallback scanned ascending, the backtester's scanned ``sorted(..., reverse=True)``,
so the same ticker resolved to different models depending on which entry point
asked.

This module is the single replacement for both.  Its contract:

``load_model_index()``
    ``ticker -> model folder``, read from ``model_index.json`` in the store and
    normalised from the file's ``{"model_folder": ...}`` entry shape.  Every way
    the file can be unusable — absent, unreadable, not JSON, not an object, no
    ``ticker_to_model`` mapping — is a ``warning`` and an **empty** mapping, so a
    degradation is visible and still cannot serve anything.

``resolve_model_folder(ticker)`` / ``load_model_for(ticker)``
    The request path.  An unmapped ticker raises :class:`ModelNotFound`; a
    mapped ticker whose folder or ``MLmodel`` is missing also raises.  There is
    no fallback to "the first folder", to a previously cached model, or to any
    other ticker's model: a wrong answer is worse than no answer.

    ``load_model_for`` returns ``(model, feature_columns)``, where the columns
    are the model's feature contract — the persisted
    ``feature_schema.json`` (Task 1.4, written beside ``MLmodel``) when it
    exists, otherwise the model's own ``feature_names_in_`` **with a
    ``warning`` naming the ticker**.  All 45 models on disk today predate the
    persisted contract, so requiring it would make the fleet unloadable before
    Task 2.3 retrains; falling back quietly would recreate the silent
    substitution this task removes.  Callers pass their computed frame through
    ``features.schema.align_to_schema`` with those columns, so a model that
    expects a column serving cannot build fails with ``FeatureSchemaError``
    instead of being handed a padded ``NaN``.

``model_dir()`` / ``index_path()``
    The store's absolute path, resolved from *this file's* location.  Every
    caller — ``src/serve.py``, ``src/backtest.py``, ``cli.py`` and
    ``predict.py`` — routes its model selection through this module, so none of
    them builds the path from its own relative base and the four cannot drift
    apart again.

``write_model_index(entries, path=None)``
    The index's **only** writer, beside the reader and through the same
    :func:`index_path`.  Nothing wrote it on the way out of training, so the
    committed file could only get staler; now ``src/train.py`` calls this once,
    in the parent process, after every ticker has been trained and logged, and
    ``scripts/build_model_index.py`` goes through it too so the file has one
    shape and one guard wherever it is written from.  An empty mapping is
    refused at ``warning`` rather than written: the CI path logs no model at
    all, and ``mapped_tickers: 0`` over a populated index disables every
    prediction while the file still looks like a valid index.

**R4 — where a directory scan is allowed.**  ``scripts/build_model_index.py``
is an explicit rebuild command: it scans the store, reads each logged model's
run metadata and writes the index through the writer above.  That scan is
legitimate and stays there.  It is forbidden
here: neither :func:`resolve_model_folder` nor :func:`load_model_for` ever
calls ``os.listdir`` on the store.  A request path may only read the index and
the single folder the index names.
"""
from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping
from typing import Any

import config
from features.schema import (
    FEATURE_SCHEMA_FILENAME,
    FeatureSchemaError,
    load_feature_schema,
)

logger = config.get_logger(__name__)

#: File name of the ``ticker -> model folder`` index inside the model store.
INDEX_FILENAME = "model_index.json"

#: The MLflow experiment id whose logged models *are* the model store.  Pinned
#: here and nowhere else in application code: renumbering it moves the store,
#: which the rescue plan's constraints forbid.
EXPERIMENT_ID = "1"

#: The marker file MLflow writes in a logged model's artifact directory.
MLMODEL_FILENAME = "MLmodel"

#: Repository root, derived from this module rather than from the caller's cwd.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Index diagnostics already emitted, so a degradation that recurs on *every*
#: request is reported once rather than once per request.  ``resolve_model_folder``
#: reads the index on each uncached request — deliberately, so a rebuilt index
#: reaches a running process without a restart — which meant a broken index
#: re-logged the identical line on every ``/predict``: a per-request, unbounded
#: flood on the very endpoint that was failing.
#:
#: Keyed by the diagnostic's identity (kind, path, and any detail that makes the
#: message text different), so a *different* degradation — another failure mode,
#: a changed failure, or the same one for another index — is still reported.
#: Only the message is deduplicated; the read is untouched.
_reported_index_diagnostics: set[tuple[Any, ...]] = set()


def _warn_once(key: tuple[Any, ...], message: str, *args: Any) -> None:
    """Emit ``message`` at ``warning`` the first time ``key`` is diagnosed.

    The first occurrence is kept verbatim — same level, same text, still naming
    the path and the remedy.  Later occurrences of the *same* diagnostic are
    dropped; a different one passes straight through.
    """
    if key in _reported_index_diagnostics:
        return
    _reported_index_diagnostics.add(key)
    logger.warning(message, *args)


class ModelNotFound(LookupError):
    """No model is mapped to the requested ticker, or its artifacts are gone.

    A ``LookupError``: asking for a model the store does not have is a lookup
    failure, never an invitation to hand back a different model.
    """


def model_dir() -> pathlib.Path:
    """Absolute path of the model store.

    Resolved from this file's own location, so the four call sites and the
    rebuild command cannot drift to different bases (F9) or depend on the
    process's working directory.
    """
    return (_REPO_ROOT / config.MODEL_DIR / EXPERIMENT_ID / "models").resolve()


def index_path() -> pathlib.Path:
    """Absolute path of the model index inside :func:`model_dir`."""
    return model_dir() / INDEX_FILENAME


def load_model_index() -> dict[str, str]:
    """Map ``ticker -> model folder`` from the store's index.

    Returns:
        The mapping, or ``{}`` when the index cannot be used.  The file's entry
        shape (``{"model_folder": ..., "match_score": ..., "n_features": ...}``,
        the shape the pre-F10 rebuild wrote) is normalised to the folder name;
        an entry without one is dropped.

    Every degradation logs at ``warning`` — once per process per *distinct*
    diagnostic (see :func:`_warn_once`), because this function runs on the
    request path and an index that stays broken would otherwise re-log the same
    line on every request.  An empty mapping means every lookup raises
    :class:`ModelNotFound`, and that must be visible rather than read as "no
    models exist".
    """
    return _read_index_entries(index_path())


def _discard(*_args: Any, **_kwargs: Any) -> None:
    """Diagnostic sink for :func:`_read_index_entries`'s quiet mode."""


def _read_index_entries(path: pathlib.Path, *, quiet: bool = False) -> dict[str, str]:
    """Parse an index file into ``ticker -> model folder``.

    Args:
        path: The index file to read.
        quiet: Suppress the diagnostics.  :func:`write_model_index` reads the
            file only to avoid dropping mappings it is not rewriting, and the
            state it is about to overwrite is not a live degradation: reporting
            it as one would bury the write's own outcome.

    Returns:
        The mapping, or ``{}`` when the file cannot be used — absent, unreadable,
        not JSON, not an object, or carrying no ``ticker_to_model`` mapping.
    """
    warn = _discard if quiet else _warn_once
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        warn(
            ("unreadable", str(path), type(exc).__name__),
            "Model index %s is unreadable (%s) — no ticker can be resolved", path, exc,
        )
        return {}

    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        warn(
            ("not-json", str(path), str(exc)),
            "Model index %s is not valid JSON (%s) — no ticker can be resolved",
            path, exc,
        )
        return {}

    if not isinstance(parsed, dict):
        warn(
            ("not-an-object", str(path), type(parsed).__name__),
            "Model index %s must hold a JSON object, found %s — no ticker can be resolved",
            path, type(parsed).__name__,
        )
        return {}

    entries = parsed.get("ticker_to_model")
    if not isinstance(entries, dict):
        warn(
            ("no-ticker-mapping", str(path), type(entries).__name__),
            "Model index %s carries no ticker_to_model mapping — no ticker can be resolved",
            path,
        )
        return {}

    index: dict[str, str] = {}
    for ticker, entry in entries.items():
        folder = entry.get("model_folder") if isinstance(entry, dict) else entry
        if isinstance(folder, str) and folder:
            index[str(ticker)] = folder
        else:
            warn(
                ("entry-without-folder", str(path), str(ticker)),
                "Model index entry for %s names no model folder (%r) — ignored", ticker, entry
            )

    if not index:
        warn(
            ("empty-index", str(path)),
            "Model index %s maps no tickers — every prediction will fail until "
            "scripts/build_model_index.py is run", path,
        )
    return index


def write_model_index(entries: Mapping[str, str], path: pathlib.Path | None = None, *,
                      total_models: int | None = None, errors: int = 0) -> None:
    """Write ``ticker -> model folder`` into the store's index (F10).

    This is the index's only writer.  ``src/train.py`` calls it once, in the
    parent process, after every ticker has been trained and logged, so the file
    cannot go stale behind a retrain; ``scripts/build_model_index.py`` — the
    explicit rebuild — writes through it as well, so the index has one shape and
    one guard wherever it is written from.

    The folder name is never derived here.  Training passes the id MLflow
    reported for the model it just logged (``ModelInfo.model_id``, the name of
    the directory under ``mlruns/1/models/``) and the rebuild passes the folder
    it resolved a run to; an entry whose folder cannot be resolved is not
    written at all.

    Args:
        entries: ``ticker -> model folder`` pairs produced by *this* run.
        path: Where to write.  Defaults to :func:`index_path` — the same
            accessor :func:`load_model_index` reads through.
        total_models: How many models the run examined, when that is more than
            it could map: the rebuild scans every folder and may fail to resolve
            some.  Defaults to the number of entries.
        errors: How many models the run could not resolve or load.

    Merging: mappings already in ``path`` for tickers this run did **not**
    retrain are carried over.  A full retrain names every ticker and so writes
    exactly the file a wholesale replace would; a partial one (``--tune``
    covers ten tickers and ``--ticker`` one) would otherwise unmap the rest and
    silently disable their predictions — the same failure as writing the file
    empty, one step earlier.

    R3: an **empty** mapping is refused, at ``warning``, and the file is left
    untouched.  In CI no model is logged at all (``log_model`` runs only outside
    CI), and overwriting a populated index with ``mapped_tickers: 0`` would
    disable every prediction while the file still looked like a valid index.

    A write that fails is a ``warning`` and nothing else: the run's models are
    already logged, and the index is rewritten by the next run.
    """
    target = index_path() if path is None else pathlib.Path(path)
    named = {str(ticker): str(folder) for ticker, folder in entries.items() if folder}

    if not named:
        logger.warning(
            "Refusing to write an empty model index: this run produced no "
            "ticker -> model mapping (no model was logged, or none could be "
            "resolved). %s is left unchanged — an index that maps nothing "
            "disables every prediction", target,
        )
        return

    mapping = {**_read_index_entries(target, quiet=True), **named}
    payload = {
        "total_models": len(named) if total_models is None else int(total_models),
        "indexed_models": len(named),
        "mapped_tickers": len(mapping),
        "errors": int(errors),
        "ticker_to_model": {ticker: mapping[ticker] for ticker in sorted(mapping)},
    }

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except OSError as exc:
        logger.warning(
            "Model index %s could not be written (%s) — %d ticker(s) stay "
            "unmapped, and every prediction for them fails until it is rebuilt",
            target, exc, len(named),
        )
        return

    carried = len(mapping) - len(named)
    logger.info(
        "Model index %s now maps %d ticker(s): %d from this run%s",
        target, len(mapping), len(named),
        f", {carried} kept from earlier runs" if carried else "",
    )


def resolve_model_folder(ticker: str) -> str:
    """Return the model folder the index maps ``ticker`` to.

    Raises:
        ModelNotFound: when the index has no entry for ``ticker``.

    This never scans the store.  The two deleted fallbacks both did, and both
    answered an unmapped ticker with a model trained on something else.
    """
    index = load_model_index()
    try:
        return index[ticker]
    except KeyError:
        raise ModelNotFound(
            f"no model is mapped to {ticker!r}: {index_path()} maps "
            f"{len(index)} ticker(s)"
        ) from None


def _load_xgboost_model(artifact_dir: pathlib.Path) -> Any:
    """Load the logged XGBoost model from ``artifact_dir``.

    The only ``mlflow.xgboost.load_model`` call on a request path.  Imported
    locally so that importing this module — by a test, or by ``cli.py``'s
    argument parsing — does not pay MLflow's import cost.
    """
    import mlflow.xgboost

    return mlflow.xgboost.load_model(str(artifact_dir))


def _as_name_list(names: Any) -> list[str]:
    """``list[str]`` from a sequence of names, or ``[]`` when there is none.

    Both accessors below return a *numpy array* in practice — a fitted
    ``XGBClassifier`` stores ``feature_names_in_`` as ``np.ndarray`` — so
    truthiness ("``if names:``") cannot be used on them: an array of more than
    one element raises ``ValueError: truth value ... is ambiguous``.  The
    conversion is the check.
    """
    if names is None:
        return []
    try:
        return [str(name) for name in names]
    except TypeError:
        return []


def _model_feature_names(model: Any) -> list[str]:
    """Return ``model``'s own feature list, unwrapping a wrapped estimator.

    Tried in order: the estimator's ``feature_names_in_``, a wrapped
    estimator's (``CalibratedClassifierCV.estimator``), a booster's
    ``feature_names``, then a bare ``feature_names`` attribute.  Empty means the
    model exposes no column contract at all, which cannot be enforced.
    """
    if model is not None:
        for provider in (model, getattr(model, "estimator", None)):
            if provider is None:
                continue
            names = _as_name_list(getattr(provider, "feature_names_in_", None))
            if names:
                return names

        get_booster = getattr(model, "get_booster", None)
        if callable(get_booster):
            names = _as_name_list(getattr(get_booster(), "feature_names", None))
            if names:
                return names

        names = _as_name_list(getattr(model, "feature_names", None))
        if names:
            return names

    return []


def _feature_columns(ticker: str, artifact_dir: pathlib.Path, model: Any) -> list[str]:
    """The model's feature contract, from disk when persisted.

    R1: an *absent* contract falls back to the model's own names with a
    ``warning`` naming the ticker — a legacy model must stay loadable, but the
    gap must be visible.  A *malformed* or *unenforceable* contract raises:
    only absence is tolerated, never a contract that cannot be trusted.
    """
    schema_path = artifact_dir / FEATURE_SCHEMA_FILENAME
    try:
        return load_feature_schema(schema_path)
    except FileNotFoundError:
        logger.warning(
            "Model for %s has no persisted feature schema at %s — falling back to "
            "the model's own feature_names_in_; its column contract is unverified "
            "until the next retrain", ticker, schema_path,
        )
    except FeatureSchemaError as exc:
        raise FeatureSchemaError(
            f"model for {ticker} carries a malformed feature schema at "
            f"{schema_path}: {exc}"
        ) from exc

    columns = _model_feature_names(model)
    if not columns:
        raise FeatureSchemaError(
            f"model for {ticker} has no persisted feature schema at {schema_path} "
            f"and exposes no feature names of its own — its column contract cannot "
            f"be enforced"
        )
    return columns


def load_model_for(ticker: str) -> tuple[Any, list[str]]:
    """Load ``ticker``'s model and its feature contract from the local store.

    Args:
        ticker: An exact ticker key, e.g. ``"BBCA.JK"``.

    Returns:
        ``(model, feature_columns)``.  Callers pass their computed frame through
        ``features.schema.align_to_schema(frame, feature_columns)`` — never a
        positional zip and never a ``NaN`` pad — so a model whose contract the
        frame cannot satisfy fails with ``FeatureSchemaError``.

    Raises:
        ModelNotFound: the index maps no model to ``ticker``, or the folder it
            names has no ``MLmodel``.  Another ticker's model is never
            substituted.
        FeatureSchemaError: the model's persisted feature contract is malformed,
            or the model exposes no feature names and there is no contract.
    """
    folder = resolve_model_folder(ticker)
    artifact_dir = model_dir() / folder / "artifacts"
    mlmodel_path = artifact_dir / MLMODEL_FILENAME
    if not mlmodel_path.is_file():
        raise ModelNotFound(
            f"the index maps {ticker!r} to {folder!r}, but {mlmodel_path} does not exist"
        )

    model = _load_xgboost_model(artifact_dir)
    return model, _feature_columns(ticker, artifact_dir, model)
