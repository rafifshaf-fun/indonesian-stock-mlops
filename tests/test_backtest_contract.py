"""Task 3.2 — the backtest runs on the model's own contract, over a chosen window.

Before this task the backtest measured nothing: it recomputed training's
*panel-wide* feature pruning on a **single ticker's** frame, which dropped 10-17
of the 60 columns the model's persisted ``feature_schema.json`` requires, so
``align_to_schema`` raised and the ticker was skipped — 0 of 45 ran, no trades,
no return, no costs.  It also read ``data/processed/features.csv``, the CI-mode
artefact the models were **not** trained on, while training reads
``data/processed/features.parquet``.

**The number this makes computable is still in-sample.**  Every row the
backtest predicts on is a row the model was fitted on, so a positive return here
is a statement about memorisation, not about a strategy.  Task 3.3 supplies the
holdout; nothing in this module asserts that money is made — the tests assert
that the run *happens*, that trades are simulated, and that the frame handed to
the model is exactly its persisted contract, in order.  An assertion that the
return is positive would encode a lie.
"""
from __future__ import annotations

import inspect
import logging
import pathlib
import warnings

import pandas as pd
import pytest

# ``src/backtest.py`` imports mlflow, which imports mlflow.tracing, which imports
# opentelemetry, whose runtime-context loader still calls the removed
# ``importlib.metadata`` dict interface.  On Python 3.11 that emits
# ``DeprecationWarning: SelectableGroups dict interface is deprecated. Use
# select.`` — upstream, not this repository — and the suite runs with
# ``filterwarnings = ["error"]``, so it aborts collection without a filter.  The
# block is scoped to that one message and category, exits before any test runs,
# and (unlike ``pytestmark``) covers the import itself, which is the same
# arrangement ``tests/test_backtest_costs.py`` documents and needs because
# ``src/backtest.py`` calls ``warnings.filterwarnings("ignore")`` at import and
# ``catch_warnings`` is what restores the suite's escalation afterwards.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="SelectableGroups dict interface is deprecated",
        category=DeprecationWarning,
    )
    import config

    import backtest

#: The repository's own artefacts, resolved from this file rather than from the
#: process's working directory.  The backtest itself resolves its input paths
#: from the cwd (``config`` holds relative paths), so it runs from the repo root;
#: these constants let the tests *name* the same files from anywhere.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATURES_PARQUET = REPO_ROOT / config.DATA_PROCESSED_PATH
CI_FEATURES_CSV = REPO_ROOT / config.DATA_PROCESSED_CSV_PATH

#: One ticker with a mapped, retrained model and the full 60-name contract.
TICKER = "BBCA.JK"

#: The bounded window every data-backed test evaluates.  One year of trading
#: days keeps the simulation short while leaving it long enough to trade, and it
#: is far narrower than the six-year history the models were fitted on — the
#: seam Task 3.3's as-of cutoff needs.
WINDOW_START = "2024-01-01"
WINDOW_END = "2024-12-31"


def _require(path: pathlib.Path, what: str) -> pathlib.Path:
    """Fail with an actionable message when a committed input is missing."""
    if not path.exists():
        pytest.fail(
            f"{what} is missing at {path} — this test measures the backtest "
            f"against the repository's own fleet and feature artefact and "
            f"cannot run without it"
        )
    return path


def _frame(*dates: str) -> pd.DataFrame:
    """A minimal daily frame whose only content is its index and a price."""
    return pd.DataFrame(
        {"Close": [100.0] * len(dates)}, index=pd.to_datetime(list(dates))
    )


@pytest.fixture(scope="module")
def evaluation_frame() -> pd.DataFrame:
    """The evaluation frame the backtest loads, bounded to the test window.

    Read once for the whole module: the artefact is 59 MB / 64k rows, and every
    data-backed test below measures this same frame.
    """
    _require(FEATURES_PARQUET, "the committed feature parquet")
    return backtest.load_eval_frame(start=WINDOW_START, end=WINDOW_END)


# ═══════════════════════════════════════════════════════════════════════════════
# The bounded evaluation window — the seam Task 3.3 depends on
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_window_keeps_only_the_rows_inside_it():
    frame = _frame("2024-01-02", "2024-01-05", "2024-01-10")

    bounded = backtest.filter_eval_window(
        frame, start="2024-01-03", end="2024-01-06"
    )

    assert list(bounded.index) == [pd.Timestamp("2024-01-05")]


def test_both_window_bounds_are_inclusive():
    """``--start``/``--end`` name the first and last day *evaluated*."""
    frame = _frame("2024-01-02", "2024-01-05", "2024-01-10")

    bounded = backtest.filter_eval_window(
        frame, start="2024-01-05", end="2024-01-05"
    )

    assert list(bounded.index) == [pd.Timestamp("2024-01-05")]


def test_either_bound_can_be_given_alone():
    frame = _frame("2024-01-02", "2024-01-05", "2024-01-10")

    assert list(
        backtest.filter_eval_window(frame, start="2024-01-05").index
    ) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-10")]
    assert list(
        backtest.filter_eval_window(frame, end="2024-01-05").index
    ) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05")]


def test_no_window_leaves_every_row():
    frame = _frame("2024-01-02", "2024-01-05", "2024-01-10")

    unbounded = backtest.filter_eval_window(frame)

    assert unbounded.equals(frame)


def test_a_window_that_selects_no_rows_is_refused_loudly():
    """Zero trades from an empty window is the failure this seam must name.

    A silent empty result is indistinguishable from a strategy that never
    traded, which is why the refusal carries the requested window *and* the
    range the frame actually covers.
    """
    frame = _frame("2024-01-02", "2024-01-10")

    with pytest.raises(ValueError) as excinfo:
        backtest.filter_eval_window(frame, start="2030-01-01")

    message = str(excinfo.value)
    assert "no rows" in message
    assert "2030-01-01" in message
    assert "2024-01-02" in message and "2024-01-10" in message


def test_a_malformed_bound_is_refused_with_the_option_named():
    frame = _frame("2024-01-02")

    with pytest.raises(ValueError) as excinfo:
        backtest.filter_eval_window(frame, start="not-a-date")

    assert "start" in str(excinfo.value)


def test_run_backtest_takes_the_evaluation_window():
    """Task 3.3's as-of cutoff calls the run entry point, so the bound lives there."""
    parameters = inspect.signature(backtest.run_backtest).parameters

    assert "start" in parameters
    assert "end" in parameters


# ═══════════════════════════════════════════════════════════════════════════════
# The artefact the models were trained on — not the CI-mode one
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_backtest_does_not_read_the_ci_artefact_path():
    """The CSV path was the second half of why every ticker was skipped."""
    assert not hasattr(backtest, "DATA_PROCESSED_CSV_PATH")


def test_the_evaluation_frame_is_the_full_mode_training_artefact(evaluation_frame):
    """``vp_*`` exists only in the parquet, so its presence names the source.

    ``build_feature_set(mode="ci")`` skips ``volume_profile``, ``news_sentiment``
    and ``google_trends`` and writes the CSV; the fleet was trained on the
    parquet.  The 19 ``vp_*`` columns are the parquet-only difference, so a
    frame carrying them cannot have come from the CSV.
    """
    ci_columns = set(
        pd.read_csv(_require(CI_FEATURES_CSV, "the CI-mode feature CSV"), nrows=0).columns
    )

    parquet_only = sorted(set(evaluation_frame.columns) - ci_columns)

    assert parquet_only, (
        "the evaluation frame carries no column the CI-mode CSV lacks — it is "
        "not the full-mode artefact training used"
    )
    assert "vp_poc" in parquet_only


def test_the_evaluation_frame_is_bounded_to_the_window(evaluation_frame):
    assert not evaluation_frame.empty
    assert evaluation_frame.index.min() >= pd.Timestamp(WINDOW_START)
    assert evaluation_frame.index.max() <= pd.Timestamp(WINDOW_END)
    assert set(evaluation_frame["ticker"].unique()) == set(config.TICKERS)


# ═══════════════════════════════════════════════════════════════════════════════
# The run itself — one ticker, one window, on its own persisted contract
# ═══════════════════════════════════════════════════════════════════════════════

class _RecordingModel:
    """Delegates to the real estimator and records every frame it is handed."""

    def __init__(self, inner):
        self.inner = inner
        self.columns = []

    def predict_proba(self, frame):
        self.columns.append(list(frame.columns))
        return self.inner.predict_proba(frame)


def test_a_bounded_run_trades_on_the_models_own_contract(
    monkeypatch, caplog, evaluation_frame
):
    """The run this task exists to make possible — and it is **in-sample**.

    Every row of ``evaluation_frame`` is a row the model was fitted on, so the
    return this run produces measures memorisation, not a strategy.  Nothing
    here asserts that money is made: the assertions are that the ticker is not
    skipped, that trades are simulated, and that the frame handed to the model
    is exactly the persisted contract, in the contract's order.  An assertion
    that the return is positive would encode a lie — Task 3.3 supplies the
    holdout that would make such a claim meaningful.
    """
    from model_store import load_model_for

    model, columns = load_model_for(TICKER)
    recorder = _RecordingModel(model)
    monkeypatch.setattr(backtest, "_model_cache", {})
    monkeypatch.setattr(backtest, "load_model_for", lambda ticker: (recorder, columns))
    caplog.set_level(logging.WARNING)

    rows = evaluation_frame[evaluation_frame["ticker"] == TICKER]
    assert len(rows) >= backtest.TRAINING_CONFIG["min_rows"], (
        f"the test window carries only {len(rows)} rows for {TICKER}; the "
        f"backtest would skip it on the row guard, not on the contract"
    )

    result = backtest.backtest_ticker(TICKER, evaluation_frame)

    assert result is not None, (
        f"{TICKER} was skipped — the evaluation frame cannot satisfy its model's "
        f"feature contract, which is the defect this task removes"
    )
    assert result["trades"], "the run simulated no trades at all"
    assert result["metrics"]["n_trades"] >= 1, "no trade was ever closed"

    assert recorder.columns, "the model was never asked to predict"
    # One frame per row, and every one of them the same contract: pruning a
    # single ticker's frame is exactly what used to drop 10-17 of these columns.
    assert all(seen == list(columns) for seen in recorder.columns), (
        "the frame handed to the model is not its persisted contract, in order"
    )
    assert len(columns) == 60, "the fleet's contracts are 60 ordered names"
    assert not [
        record for record in caplog.records if "contract cannot be met" in record.getMessage()
    ], "the ticker was skipped for a contract mismatch"
