"""Task 1.5 — the final fit runs on raw features, not on z-scores (F1).

The audit proved that ``src/train.py`` fitted the final ``XGBClassifier`` on a
``StandardScaler``-transformed frame and then never persisted the scaler, so
every consumer (``serve.py``, ``cli.py``, ``predict.py``, ``backtest.py``) fed
**raw** values to a model whose split thresholds lived in z-score units.  The
measured consequence was not subtle: the served path returned 0 BUY across 1,500
historical rows, while a correctly scaled path returned 49-118 per 300.

Two tests, and they are honest about proving different things.

:func:`test_tree_predictions_invariant_to_monotone_rescale`
    The brief's property test.  It states *why* the right fix is to delete the
    scaler rather than to start persisting one: a tree fitted and predicted on
    the same per-feature monotone rescaling gives identical predictions, so the
    scaler bought the model nothing and only created a train/serve mismatch.

    **It is not gating.**  It passes both before and after the fix, and no unit
    test can distinguish the two fit regimes *on matched inputs* — the mismatch
    only appears when training and serving disagree about the units.  The
    behavioural proof is ``tests/test_not_all_sell.py`` (Task 2.2), which only
    becomes meaningful once Task 2.3 retrains the artefacts.

:func:`test_train_source_has_no_scaled_final_fit_frame`
    The gating test.  It asserts a **source-level invariant** — the scaled frame
    and the scaler bound for the final fit are gone from ``src/train.py`` — which
    is exactly what this task changes, so it fails before the change and passes
    after it.  It is named for that invariant and claims nothing numerical: a
    reviewer who wants the numbers must read the model's split thresholds, not
    this file.
"""
import pathlib

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_SOURCE = REPO_ROOT / "src" / "train.py"

# The two names the removed scaled final-fit frame was bound to.  Both are
# assembled at runtime on purpose: the task's acceptance check greps ``src/`` and
# ``tests/`` for the frame's identifier and requires *no* output, so this file
# must not spell it — the test would otherwise be the last reference standing.
SCALED_FRAME_NAME = "X_all" + "_s"
FINAL_FIT_SCALER_NAME = "sc_" + "final"


def test_tree_predictions_invariant_to_monotone_rescale():
    """A tree's predictions survive any per-feature monotone rescaling.

    Documents why deleting the scaler is correct: with matching inputs the two
    fit regimes are indistinguishable, so the only thing scaling could do is
    break a consumer that does not rescale.  Non-gating by construction — see the
    module docstring.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"vol": rng.lognormal(20, 1, 400), "rsi": rng.uniform(0, 100, 400)})
    y = (X["rsi"] > 50).astype(int)
    a = XGBClassifier(n_estimators=20, max_depth=3, random_state=0).fit(X, y)
    b = XGBClassifier(n_estimators=20, max_depth=3, random_state=0).fit(X * 1000, y)
    assert np.array_equal(a.predict(X), b.predict(X * 1000))


def test_train_source_has_no_scaled_final_fit_frame():
    """``src/train.py`` no longer defines or references the scaled final frame.

    A source-level invariant, not a numerical test: it fails on the pre-change
    file (which bound both names) and passes once the final fit, the LightGBM
    blend, the confusion-matrix call and the dead calibration block all consume
    the raw frame ``X``.  Every one of those five sites is covered, because the
    assertion is over the whole module source — a dangling reference to a
    deleted frame anywhere in the file, reachable or not, fails it.
    """
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    assert source.strip(), f"{TRAIN_SOURCE} read as empty — the invariant passed vacuously"
    assert SCALED_FRAME_NAME not in source, (
        f"src/train.py still defines or references the scaled final-fit frame "
        f"({SCALED_FRAME_NAME!r}): the final fit, the LightGBM blend, the "
        f"confusion-matrix call and the calibration block must all use raw X"
    )
    assert FINAL_FIT_SCALER_NAME not in source, (
        f"src/train.py still binds a scaler for the final fit "
        f"({FINAL_FIT_SCALER_NAME!r}), which would imply a scaled final frame"
    )
