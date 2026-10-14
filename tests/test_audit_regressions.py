"""Task 2.2 — the audit's critical and high findings, as executable pins.

Each test below is named for the finding it pins and asserts a *measured*
property of the live tree, not a restatement of the fix's prose:

F1  ``test_not_all_sell_on_history``
    The audit's worst finding: the served path answered 0 BUY across 1,500
    historical rows (5 tickers x 300) because models fitted on a lost
    standardisation were fed raw features.  This drives the **real** retrained
    models over the **real** committed ``data/processed/features.parquet``,
    each frame aligned through ``features.schema.align_to_schema`` to the
    ticker's own persisted contract, and asserts that **both** outcomes occur
    (``total_buys > 0`` *and* ``total_sells > 0``).  A floor on BUY alone is not
    enough: the mirror defect — an all-BUY fleet — is produced for real by an
    all-zeros feature frame (7,735/7,735 BUY), so the gate must exclude both
    degenerate constants.  The same pure function is applied to **each ticker's
    own** tallies as well as to the pooled ones, because five independently
    trained models are pooled here: one model collapsing to a single outcome
    while the other four stay mixed is invisible to a pooled count.  The
    per-ticker and total rates are printed so the evidence survives in the test
    log; the assertion is the gate.

F2  ``test_distinct_tickers_use_distinct_models``
    Two tickers resolve to **different** model folders, load genuinely
    different model artefacts (distinct serialised bytes), and disagree about
    the same rows.  Schema equality is explicitly *not* treated as evidence:
    all 45 persisted schemas are identical by design.

F3/F5  ``test_no_all_nan_model_feature``
    Measured over **every mapped ticker** — all 45 models, in both directions.
    The previous version looped over three tickers while its name, its brief and
    its report claimed the fleet, and that gap hid a real finding, so the whole
    fleet is now paid for deliberately (~45 s).  Each model's persisted schema
    is checked against *that ticker's own* rows of the committed training frame
    — a fleet-wide column check cannot see a column that is alive for 38
    tickers and all-NaN for the seven banks — and against the frame the
    **serving builder actually produces** for that ticker from the on-disk
    universe, so the model's input is real at inference rather than NaN-padded.
    Two measured sets come out of it, both pinned exactly in this module: 15
    tickers carry a schema column their own training rows are all-NaN for
    (``TRAINING_ALL_NAN_BY_TICKER`` — the fields yfinance does not publish for
    them, plus ADMR's ``sector_*``), and 5 tickers are served an all-NaN
    ``ob_nearest_bearish_distance`` that training *did* populate
    (``WINDOW_ALL_NAN_BY_TICKER`` — the served request window, ``period="250d"``
    of *calendar* days, with no bearish order block inside it).  The window is
    measured on the calendar extent the request uses, not on 250 rows: the two
    differ by a factor of ~1.5, and the row-count version omitted CTRA.JK, whose
    only bearish block in the last 248 rows sits 233 rows back.  A third set is
    the mirror skew, also pinned:
    ``REVIVED_BY_SERVING`` — three ADMR sector columns serving supplies for a
    model fitted on them as constant NaN.

F6/F7  ``test_cache_by_name``
    A real round-trip through ``_cache_put``/``_cache_get`` keyed by column name
    with a schema and a window: values come back under the names they were
    written for, and a schema or window mismatch is a **miss**, never a
    positional permutation of another model's column order.

F8  ``test_degraded_flag_present``
    A healthy ``/predict`` says ``degraded == []``; a sentiment outage — driven
    through the real ``features.fetchers.fetch_news_sentiment_status`` seam, not
    a stub that raises — says ``degraded == ["sentiment"]``.  The dead
    calibration symbol is absent from ``src/train.py`` and ``src/serve.py``,
    asserted **structurally** over the AST so a comment naming it costs nothing.

F11 ``test_all_imports_declared``
    Every third-party module the repository's ``*.py`` files import maps to a
    distribution declared in ``requirements.txt`` (or the dev/docker lists).
    ``tests/test_manifest.py`` — which the plan's Task 0.3 referenced — does not
    exist; this is its real home.

F12 ``test_batch_failure_is_error``
    ``/predict/batch`` answers a ticker it cannot predict with an HTTP error
    carrying no signal-shaped payload (``prediction``, ``signal``,
    ``probability_up``).  The feature-build failure — the branch the audit
    measured — is exactly a **502**; it used to answer ``prediction: 0`` (SELL)
    under 200.  An unmapped ticker is exactly a **404** — the handler re-raises
    ``HTTPException`` before its generic handler — and is pinned here as
    *unanswerable* rather than substituted from another ticker.  Neither status
    is tolerated as the other's alternative.

Everything here is offline: no test touches the network, and the autouse
``offline`` fixture turns any attempt into a loud failure rather than a silent
``except``.  Two tests read the repository's own committed data — that is the
point of this file, and it is why they are the only place the retrained fleet is
exercised end to end.

``serve`` imports mlflow → opentelemetry, whose runtime-context loader still
calls the removed ``importlib.metadata`` dict interface on Python 3.11, and
FastAPI raises its own deprecation for ``@app.on_event("startup")``.  Both are
upstream of this repository, so the two filters below are scoped to their exact
message and to this module only, exactly as ``tests/test_prediction_cache.py``
and ``tests/test_serve_parity.py`` already do; every other warning still
escalates under ``filterwarnings = ["error"]``.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import sys
import time
import types
import warnings

import numpy as np
import pandas as pd
import pytest

# ``fastapi.testclient`` → ``starlette.testclient`` emits a collection-time
# ``StarletteDeprecationWarning`` when ``httpx2`` is absent.  The filter is
# installed when this module is *imported*, which is the statement below, so the
# import is wrapped narrowly — the same wrapper ``test_degradation_visible.py``
# uses, and it re-asserts that the wrapper captured nothing else.
with warnings.catch_warnings(record=True) as _import_warnings:
    warnings.simplefilter("always")
    from fastapi.testclient import TestClient

_IMPORT_WARNINGS = [str(w.message) for w in _import_warnings]
EXPECTED_IMPORT_WARNING = (
    "Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` "
    "instead."
)

# ``src/news_sentiment.py`` calls ``warnings.filterwarnings("ignore")`` at import
# time, which would otherwise disable this suite's ``filterwarnings = ["error"]``
# for every test collected after it.  ``catch_warnings`` saves and restores the
# filter list, so the module's global ignore is discarded when the block exits.
# A module-scope import also means the "vaderSentiment not installed" notice is
# emitted once at collection rather than from whichever test touches the seam
# first — the same arrangement ``test_degradation_visible.py`` documents.
with warnings.catch_warnings():
    import news_sentiment

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:SelectableGroups dict interface is deprecated:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:(?s).*on_event is deprecated.*:DeprecationWarning"
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# The repository's own inputs — the fleet and the committed feature artefact
# ═══════════════════════════════════════════════════════════════════════════════

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATURES_PARQUET = _REPO_ROOT / "data" / "processed" / "features.parquet"
STOCKS_CSV = _REPO_ROOT / "data" / "raw" / "stocks.csv"
IHSG_CSV = _REPO_ROOT / "data" / "raw" / "ihsg.csv"
TRAIN_SOURCE = _REPO_ROOT / "src" / "train.py"
SERVE_SOURCE = _REPO_ROOT / "src" / "serve.py"

#: The five tickers the audit measured its 1,500 historical rows over.  All five
#: are LQ45 members with a mapped model, and three of them (BBCA/BBRI/BMRI) sit
#: in one sector so the sector-relative columns are computable.
HISTORY_TICKERS = ("BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK")

#: ``fetch_fundamentals``'s eight keys — the fields the builder injects as
#: columns (``inject_macro_features`` assigns each from a scalar) and therefore
#: the fields yfinance can leave unpublished for a whole ticker.  Direction 2 of
#: F3 feeds each ticker *its own* values from the committed artefact rather than
#: one constant dict for all 45: see :func:`_fundamentals_for`.
FUNDAMENTAL_FIELDS = (
    "pe_ratio",
    "pb_ratio",
    "dividend_yield",
    "market_cap",
    "debt_to_equity",
    "return_on_equity",
    "revenue_growth",
    "profit_margins",
)

#: Measured: schema columns that are all-NaN across a ticker's **own** rows of
#: the committed training frame.  This is the per-ticker fact that a fleet-wide
#: column check cannot see — ``debt_to_equity`` is alive for 38 tickers and
#: all-NaN for the seven banks — and the first version of F3's Direction 1
#: checked only the global column, so it reported none of this.
#:
#: Every entry is the same shape: yfinance publishes no value for the field, so
#: training had none, and the column reaches the model as a constant NaN.  The
#: groups behind the map:
#:
#: * ``debt_to_equity`` — the seven banks (BBCA, BBRI, BMRI, BBNI, BBTN, BRIS,
#:   ARTO);  ``totalDebt/equity`` is not published for them.
#: * ``dividend_yield`` — the seven non-payers (AMMN, ARTO, BREN, BRPT, GOTO,
#:   MBMA, MDKA).
#: * ``pe_ratio`` — the three tickers with no trailing P/E (EXCL, GOTO, MDKA).
#: * ``sector_relative_strength`` / ``sector_return_1d`` /
#:   ``sector_volume_ratio`` — ADMR.JK.  ``compute_cross_stock_features`` only
#:   computes the sector family when the ticker's sector has more than one member
#:   (``src/features/market.py:149``); ADMR's rows carry none of the three, so
#:   its model was fitted with them constant-NaN while serving now computes them
#:   (ADMR sits in ``energy_mining`` with twelve peers today).
#:
#: A column listed here is a **dead feature for that model** — it is in the
#: schema but was constant-NaN across every row the model was fitted on, so the
#: model can only have learned a default branch for it.  That is a real quality
#: finding, but it is not the audit's F3 class (a feature the model expects that
#: is populated in training and absent at inference, which is
#: :data:`WINDOW_ALL_NAN_BY_TICKER`).  Pinned exactly, so the corrected
#: per-ticker check cannot quietly stop reporting it, and so the map cannot be
#: widened to excuse a *new* dead column.
TRAINING_ALL_NAN_BY_TICKER = {
    "ADMR.JK": ("sector_relative_strength", "sector_return_1d", "sector_volume_ratio"),
    "AMMN.JK": ("dividend_yield",),
    "ARTO.JK": ("debt_to_equity", "dividend_yield"),
    "BBCA.JK": ("debt_to_equity",),
    "BBNI.JK": ("debt_to_equity",),
    "BBRI.JK": ("debt_to_equity",),
    "BBTN.JK": ("debt_to_equity",),
    "BMRI.JK": ("debt_to_equity",),
    "BREN.JK": ("dividend_yield",),
    "BRIS.JK": ("debt_to_equity",),
    "BRPT.JK": ("dividend_yield",),
    "EXCL.JK": ("pe_ratio",),
    "GOTO.JK": ("dividend_yield", "pe_ratio"),
    "MBMA.JK": ("dividend_yield",),
    "MDKA.JK": ("dividend_yield", "pe_ratio"),
}

#: Measured: the schema column that is all-NaN in the served row while that
#: ticker's own training rows **do** carry it.  This is the finding the
#: three-ticker sample hid, and it is the audit's F3 class in the narrow sense
#: that matters: a feature the model was trained with a real value for, absent at
#: inference.
#:
#: ``ob_nearest_bearish_distance`` is ``(close - ob_bearish_level.ffill()) /
#: close`` (``src/features/ict.py:136``).  The ``ffill`` carries the most recent
#: bearish order-block level forward across a gap of any length, so the column is
#: populated in 72–99 % of each of these tickers' committed rows.  A request,
#: though, carries ``yf.download(period="250d")`` rows — the latest block cited
#: is ``src/serve.py:1057``; ``src/serve.py:122`` and ``src/serve.py:964`` issue
#: the same call on the warm-up and single-ticker paths — and the ICT family is
#: recomputed on *that* frame, so when the window holds no bearish order block
#: there is no level to carry and the served value is NaN.
#:
#: ``period="250d"`` is **calendar** days, not rows.  The committed universe
#: snapshot ends ``2026-06-15``, so the served window opens 250 calendar days
#: earlier at ``2025-10-08``; that span holds **164** of the snapshot's 1,553
#: dated rows (the ~90 that fall out are weekends and IDX holidays).  Then
#: ``ta.utils.dropna`` (``src/serve.py:578``), which the universe loader applies
#: per ticker, drops the four zero-volume rows inside that span, so the frame
#: ``build_inference_features`` is actually handed holds **160** rows for every
#: ticker in this snapshot.  Measured on the window production uses — the frame
#: ``load_universe_context`` yields, cut at ``2025-10-08``:
#:
#: | ticker  | tail(250) rows | served window | last bearish block before the final row | first tail(n) that finds it | value there |
#: |---------|---------------:|--------------:|----------------------------------------:|----------------------------:|------------:|
#: | ADMR.JK | NaN            | NaN           | 297 rows                                | tail(350)                   | 0.5521      |
#: | AMRT.JK | NaN            | NaN           | 368 rows                                | tail(400)                   | -1.2020     |
#: | CTRA.JK | -0.5651        | NaN           | 248 rows                                | tail(248)                   | -0.5651     |
#: | GOTO.JK | NaN            | NaN           | 256 rows                                | tail(275)                   | -0.64       |
#: | MAPA.JK | NaN            | NaN           | 279 rows                                | tail(300)                   | -0.0938     |
#:
#: The two windows are kept apart deliberately.  ``tail(250)`` is the previous,
#: wrong measurement: it is *deeper* than production's 160 rows, yet it is
#: shallower than a 250-trading-day history, and for CTRA it lands on the far
#: side of the block at lag 233 — which is why CTRA was missing from the map
#: while ADMR, AMRT, GOTO and MAPA were not.  CTRA is the one ticker whose
#: verdict *flips* between the two windows, and it flips in the direction that
#: matters: the deeper, wrong window hid a served NaN.  It is the sharpest
#: demonstration in this file that the window definition is load-bearing.
#:
#: Verdict, and why this is a pinned exception rather than a widened assertion:
#:
#: * it is **not an absent column** — training populated it for 72–99 % of these
#:   tickers' rows (CTRA 1498/1545, ADMR 808/1051, AMRT 1530/1548, GOTO
#:   713/989, MAPA 1525/1539), and the same value reappears as soon as the
#:   window is deep enough, so the served NaN is a property of the request
#:   window;
#: * it is **not** the deferred index-alignment bug at
#:   ``src/features/ict.py:117-129``.  That bug reindexes ``ms_mss_*`` and
#:   ``pd_*`` — RangeIndex-ed ``pd.Series`` — onto the DatetimeIndex and makes
#:   *those* columns all-NaN everywhere; ``ob_bearish_level`` is a numpy array,
#:   position-aligned, and the column is alive in training.  Same module, same
#:   ``ob_*`` family, **different cause**;
#: * it *is* a genuine, narrow train/serve skew: training handed the model a
#:   number for this feature on this row and serving hands it NaN, which XGBoost
#:   reads as its default branch.  The production fix — seed the order-block level
#:   from history deeper than the request window, or compute the ICT family on the
#:   ticker's full available history — is a change to ``src/features/ict.py``,
#:   deliberately not made: the controller deferred every ``ict.py`` change so it
#:   cannot confound the Phase 3 measurement, and this task may not touch source.
#:
#: The map is asserted to name **exactly** these five tickers and one column
#: each, each column alive in that ticker's own training rows, so it cannot be
#: widened into a blanket exemption for a genuinely dead column.  The count is
#: stated because it moved: the first measured version named **4** tickers x 1
#: column = 4 pairs over 4 tickers, and the corrected window names **5** x 1 =
#: 5 pairs over 5 tickers.  CTRA.JK is the entire growth, and its justification
#: is the same measured one its four neighbours carry: a schema column, alive in
#: its own training rows, all-NaN in the frame the request actually builds, and
#: alive again the moment the window is deepened one row past the block.
#: The list grew because the *measurement* was corrected, not to fit an
#: observation — which is why the guards below still have to pass for it.
WINDOW_ALL_NAN_BY_TICKER = {
    "ADMR.JK": ("ob_nearest_bearish_distance",),
    "AMRT.JK": ("ob_nearest_bearish_distance",),
    "CTRA.JK": ("ob_nearest_bearish_distance",),
    "GOTO.JK": ("ob_nearest_bearish_distance",),
    "MAPA.JK": ("ob_nearest_bearish_distance",),
}

#: Measured: schema columns that are all-NaN across a ticker's own training rows
#: and **alive in the served row** — the third case, and the mirror of
#: :data:`WINDOW_ALL_NAN_BY_TICKER`.  A column serving cannot supply is a
#: finding; so is a column serving *invents* for a model that was never fitted on
#: it.
#:
#: ``ADMR.JK``'s three sector columns, and the evidence is in the repository:
#:
#: * the committed training artefact (``data/processed/features.parquet``) has
#:   ADMR's rows with no sector family at all, while all twelve of its sector
#:   peers carry live values in the same artefact — ADMR is the only ticker in
#:   the fleet with an all-NaN ``sector_return_1d``;
#: * ``4bf00db`` (2026-10-13) added ``ADMR.JK`` to
#:   ``SECTORS["energy_mining"]``.  Before that commit ADMR was in ``TICKERS``
#:   and in the name map but in **no** sector, so
#:   ``compute_cross_stock_features`` found ``sector is None`` for it and never
#:   computed the family (``src/features/market.py:147-149``);
#: * the artefact predates it, and the retrain that produced the current fleet
#:   (``433765f``, 2026-10-14) was fitted on that artefact, so ADMR's model saw
#:   these three columns as constant NaN.  Serving reads today's ``SECTORS``,
#:   finds twelve peers, and computes real values for them.
#:
#: This is the project rule "``TICKERS`` and ``SECTORS`` must stay in sync"
#: arriving as a train/serve skew.  The remedy is a rebuild of the training
#: artefact — a data-side fix, not a source change — so it is reported rather
#: than papered over.  Pinned so a *new* revival cannot pass unnoticed.
REVIVED_BY_SERVING = {
    "ADMR.JK": ("sector_relative_strength", "sector_return_1d", "sector_volume_ratio"),
}


def _pair_count(mapping: dict[str, tuple[str, ...]]) -> int:
    """How many ticker/column pairs a documented map of this module names."""
    return sum(len(columns) for columns in mapping.values())


#: The six ICT columns the audit deferred, all-NaN in the training parquet
#: because of the index-alignment bug at ``src/features/ict.py``.  They are named
#: here only so F3 can assert the global SHAP selection retained *none* of them:
#: the finding this test records is that the schemas are clean, not that the
#: columns are.
DEFERRED_ICT_COLUMNS = frozenset({
    "ms_mss_bullish",
    "ms_mss_bearish",
    "pd_premium_zone",
    "pd_discount_zone",
    "pd_equilibrium_distance",
    "pd_ote_zone",
})


# ═══════════════════════════════════════════════════════════════════════════════
# Offline harness
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """No test in this module may reach the network or the live cache.

    The stubs raise *and* record every attempt, and the record is asserted empty
    after each test: every production fetch is wrapped in ``try/except``, so a
    raising stub alone could be swallowed and the test would pass anyway.

    ``serve.PRED_CACHE_DB`` is redirected per test for the same reason
    ``tests/test_prediction_cache.py`` does it — no test may read or write the
    developer's real ``data/prediction_cache.db``.
    """
    import requests
    import yfinance as yf

    import serve

    attempts: list[str] = []

    def explode(*args, **kwargs):
        attempts.append(f"{args[:1]!r}{kwargs!r}"[:120])
        raise AssertionError(
            "the audit-regression suite must never use the network"
        )

    monkeypatch.setattr(requests, "get", explode)
    monkeypatch.setattr(requests, "post", explode)
    monkeypatch.setattr(yf, "download", explode)
    monkeypatch.setattr(yf, "Ticker", explode)
    monkeypatch.setattr(serve, "PRED_CACHE_DB", str(tmp_path / "prediction_cache.db"))

    yield

    assert attempts == [], f"network calls attempted during this test: {attempts}"


def _require(path: pathlib.Path, what: str) -> pathlib.Path:
    """Fail with an actionable message when a committed input is missing."""
    if not path.exists():
        pytest.fail(
            f"{what} is missing at {path} — this test measures the retrained "
            f"fleet against the repository's own artefacts and cannot run "
            f"without it"
        )
    return path


@pytest.fixture(scope="module")
def features_history() -> pd.DataFrame:
    """The committed feature artefact, read once for the whole module."""
    _require(FEATURES_PARQUET, "the committed feature parquet")
    return pd.read_parquet(FEATURES_PARQUET)


@pytest.fixture(scope="module")
def mapped_models() -> dict[str, str]:
    """``ticker -> model folder`` from the store's index, once per module."""
    from model_store import load_model_index

    index = load_model_index()
    assert index, (
        "the model index maps no tickers — every prediction fails by design; "
        "run scripts/build_model_index.py"
    )
    return index


def _aligned_history(history: pd.DataFrame, ticker: str, columns: list[str]):
    """``ticker``'s historical rows, aligned to its model's own contract.

    This is the alignment the served path performs (``align_to_schema`` in
    ``serve._prediction_frame``), applied to the committed training artefact
    rather than to a freshly downloaded frame — which is what makes the audit's
    measurement reproducible offline.
    """
    from features.schema import align_to_schema

    rows = history[history["ticker"] == ticker]
    assert not rows.empty, f"the committed parquet carries no rows for {ticker}"
    return align_to_schema(rows, columns)


def _fundamentals_for(rows: pd.DataFrame, ticker: str) -> dict:
    """``ticker``'s own fundamentals, as the committed training frame recorded them.

    ``rows`` is that ticker's slice of ``data/processed/features.parquet``.
    ``inject_macro_features`` assigns each field with ``df[col] = val`` from a
    scalar, so a fundamental column is constant across a ticker's rows *by
    construction* — asserted here rather than assumed, because if it ever stopped
    being constant then "the value training used" would no longer be one value.
    The value that matters for F3's Direction 2 is that ticker's own, and it is
    ``NaN`` for the fields yfinance does not publish for it (every bank's
    ``debt_to_equity``, every non-payer's ``dividend_yield``).

    Passing one constant dict for all 45 tickers — what the first draft did —
    makes the served measurement *easier* than production for exactly those
    tickers: it fills in the columns ``fetch_fundamentals`` leaves ``NaN``, so a
    per-ticker all-NaN fundamental column could never be seen at inference even
    though a real request would produce it.
    """
    fundamentals: dict[str, float] = {}
    for field in FUNDAMENTAL_FIELDS:
        assert field in rows.columns, (
            f"the committed training frame carries no {field!r} column, so a "
            f"request's fundamentals cannot be modelled from it"
        )
        distinct = rows[field].dropna().unique()
        assert len(distinct) <= 1, (
            f"{field!r} takes {len(distinct)} distinct values across "
            f"{ticker}'s committed rows ({distinct[:5]}) — the training builder "
            f"writes it from a scalar, so fundamentals are no longer constant "
            f"per ticker and 'the value training used' is not one value"
        )
        fundamentals[field] = rows[field].iloc[-1]
    return fundamentals


# ═══════════════════════════════════════════════════════════════════════════════
# F1 — the served path is no longer all-SELL on history
# ═══════════════════════════════════════════════════════════════════════════════

def _degenerate_outcome(total_buys: int, total_sells: int,
                        total_rows: int) -> str | None:
    """F1's gate: a fleet that answers only one way is degenerate, either way.

    The audit's defect was an all-SELL fleet — 0 BUY across its 1,500 rows.  Its
    mirror image, an all-BUY fleet, is the *same* class of defect and is not
    hypothetical: replacing every feature with ``0`` and predicting over these
    same rows returns 7,735/7,735 BUY.  A floor on BUY alone therefore passes
    that frame, which is why the gate is "both outcomes occur" rather than
    "BUY occurs".

    No percentage band is asserted.  "Both outcomes occur" is the honest,
    non-arbitrary statement of non-degeneracy, and how far the fleet sits from
    the extremes is what the printed rates are for; a band around whatever the
    data happens to give would be a threshold to tune rather than a fact to
    state.

    Returns ``None`` when both outcomes occur, otherwise a message naming the
    degeneracy.  It is a pure function of the tallies so both directions can be
    shown falsifiable without a second prediction pass.
    """
    if total_rows <= 0:
        return "the measurement covered no historical rows"
    if total_buys == 0:
        return (
            f"0 BUY across {total_rows} historical rows — the audit's degenerate "
            f"all-SELL fleet"
        )
    if total_sells == 0:
        return (
            f"0 SELL across {total_rows} historical rows — a degenerate all-BUY "
            f"fleet, the mirror of the audit's defect"
        )
    return None


def test_not_all_sell_on_history(features_history, mapped_models):
    """F1: the real models produce both outcomes on the real historical rows.

    The audit's headline finding was ``0 BUY across 1,500 historical rows``:
    models fitted on a standardisation that was lost between training and
    serving were fed raw features, and every row came back SELL.  This test
    replays the *measurement* — historical rows in, predictions out — against
    the retrained fleet, with each frame aligned to that ticker's persisted
    ``feature_schema.json`` exactly as serving aligns it.

    The assertion is that the model produces **both** outcomes across the sample
    (``total_buys > 0 and total_sells > 0``), not a floor on one side, and is
    not tuned to the data: a count of zero on either side is a finding about the
    models, not a threshold to lower.  A floor on BUY alone would pass the
    audit's defect mirrored — a degenerate all-BUY fleet, which an all-zeros
    feature frame produces for real — so both degenerate constants are excluded
    here.  The gate is applied **per ticker as well as pooled**: these five
    models were trained independently, so one of them collapsing to a single
    outcome while the other four stay mixed would hide inside a pooled count.
    The measured rates are printed because the number a human needs is not
    "greater than zero" but "how far from either extreme", and a collapse back
    toward the audit's 0.0 (or toward its mirror) would be visible in the log
    long before it reached the gate.
    """
    from model_store import load_model_for

    total_rows = 0
    total_buys = 0
    per_ticker: dict[str, tuple[int, int]] = {}

    for ticker in HISTORY_TICKERS:
        assert ticker in mapped_models, (
            f"{ticker} is not mapped in the model index; the F1 measurement "
            f"cannot be taken over the audit's sample"
        )
        model, columns = load_model_for(ticker)
        aligned = _aligned_history(features_history, ticker, columns)

        predictions = model.predict(aligned)
        assert len(predictions) == len(aligned)
        buys = int((np.asarray(predictions) == 1).sum())

        per_ticker[ticker] = (buys, len(aligned))
        total_buys += buys
        total_rows += len(aligned)

    total_sells = total_rows - total_buys

    report = " | ".join(
        f"{ticker}: {buys}/{rows} = {buys / rows:.4f}"
        for ticker, (buys, rows) in per_ticker.items()
    )
    print(
        f"\nF1 BUY rate per ticker: {report}\n"
        f"F1 BUY rate in total:   {total_buys}/{total_rows} = "
        f"{total_buys / total_rows:.4f}\n"
        f"F1 SELL rate in total:  {total_sells}/{total_rows} = "
        f"{total_sells / total_rows:.4f}"
    )

    # The gate is falsifiable in both directions, demonstrated here rather than
    # only claimed: an all-BUY and an all-SELL sample are each rejected while a
    # mixed one is accepted.  (The counts are arbitrary — only the shape
    # matters — so this is not a rate threshold smuggled in as a demo.)
    assert _degenerate_outcome(total_rows, 0, total_rows) is not None
    assert _degenerate_outcome(0, total_rows, total_rows) is not None
    assert _degenerate_outcome(4, 6, 10) is None

    # Per ticker as well as pooled.  The pooled count above cannot see one
    # collapsed model among four mixed ones, and the per-ticker tallies are
    # already in hand, so the same pure function is applied to each of them.
    per_ticker_degeneracy = {
        ticker: _degenerate_outcome(buys, rows - buys, rows)
        for ticker, (buys, rows) in per_ticker.items()
    }
    collapsed = {
        ticker: message
        for ticker, message in per_ticker_degeneracy.items()
        if message is not None
    }
    assert not collapsed, (
        f"per-ticker degeneracy: {collapsed}. Tallies: {per_ticker}. A model "
        f"that answers only one way on every one of its own historical rows has "
        f"collapsed to a constant, so the audit's F1 defect is back for that "
        f"ticker — or its mirror — even though the pooled "
        f"{total_buys}/{total_rows} count looks mixed."
    )

    degeneracy = _degenerate_outcome(total_buys, total_sells, total_rows)
    assert degeneracy is None, (
        f"{degeneracy}: {per_ticker}. The served path has collapsed to a single "
        f"constant outcome, so the audit's F1 defect is back — or its mirror "
        f"image, which the old ``buys > 0`` floor could not see."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# F2 — two tickers do not share a model
# ═══════════════════════════════════════════════════════════════════════════════

def test_distinct_tickers_use_distinct_models(
        features_history, mapped_models, monkeypatch):
    """F2: distinct tickers resolve to distinct folders and distinct artefacts.

    The audit's substitution — ``sorted(os.listdir(models_dir))[0]`` answering
    for *any* ticker — is what this pins against, so the assertions are about
    *identity*, not about the numbers a model happens to return.

    Four independent facts, strongest last:

    1. the index maps the two tickers to different folder names;
    2. ``load_model_for`` returns two different objects from two different
       artifact directories;
    3. the serialised booster bytes differ, so they are different trained
       models and not two loads of one;
    4. the loader is handed exactly the two artifact directories the index names,
       in order, and the loads it performs through the **real** loader return two
       distinct objects — so neither ticker is answered from the other's folder.

    The 45 persisted schemas are identical by design (Task 1.4 froze one column
    contract for the whole fleet), so schema equality is asserted here only to
    document that it is *not* used as evidence of distinctness.
    """
    import model_store
    from model_store import load_model_for, resolve_model_folder

    first, second = "BBCA.JK", "BBRI.JK"
    for ticker in (first, second):
        assert ticker in mapped_models, f"{ticker} is not mapped in the index"

    folder_a = resolve_model_folder(first)
    folder_b = resolve_model_folder(second)
    assert folder_a != folder_b, (
        f"{first} and {second} both resolve to {folder_a!r} — one model is "
        f"answering for two tickers"
    )

    model_a, columns_a = load_model_for(first)
    model_b, columns_b = load_model_for(second)
    assert model_a is not model_b

    # The contract is shared by design; saying so here keeps it from being
    # mistaken for the distinctness evidence below.
    assert list(columns_a) == list(columns_b), (
        "the 45 persisted schemas were identical when this task measured them; "
        "if they now differ, the fleet was retrained with divergent contracts"
    )

    artifacts_a = model_store.model_dir() / folder_a / "artifacts"
    artifacts_b = model_store.model_dir() / folder_b / "artifacts"
    assert artifacts_a != artifacts_b

    assert hashlib_sha256(artifacts_a) != hashlib_sha256(artifacts_b), (
        f"the models mapped to {first} and {second} carry identical serialised "
        f"bytes at {artifacts_a} and {artifacts_b} — one artefact is serving "
        f"both tickers"
    )

    # Each load opens the artifact directory *its own ticker* names — pinned by
    # recording what ``model_store`` hands the loader, not merely that the two
    # folder names differ.  The recorder **delegates to the real loader**: an
    # earlier version returned ``model_a``/``model_b`` from the stub, which made
    # the ``model_c is not model_d`` assertion below a tautology about the stub's
    # own two return values rather than a fact about two real loads.
    opened: list[pathlib.Path] = []
    real_load = model_store._load_xgboost_model

    def recording_load(artifact_dir):
        opened.append(pathlib.Path(artifact_dir))
        return real_load(artifact_dir)

    monkeypatch.setattr(model_store, "_load_xgboost_model", recording_load)
    model_c, columns_c = load_model_for(first)
    model_d, columns_d = load_model_for(second)

    assert opened == [artifacts_a, artifacts_b], (
        f"the loader was handed {opened}, not the folders the index names for "
        f"{first} and {second} ({[artifacts_a, artifacts_b]})"
    )
    assert model_c is not model_d, (
        f"two loads from {artifacts_a} and {artifacts_b} returned the same "
        f"object — the two directories hold one model"
    )
    assert list(columns_c) == list(columns_a)
    assert list(columns_d) == list(columns_b)

    # Two genuinely different models must not agree about every row; if they did,
    # distinctness would be an on-disk fact with no consequence in the response.
    rows_a = _aligned_history(features_history, first, columns_a)
    predictions_a = np.asarray(model_a.predict(rows_a))
    predictions_b = np.asarray(model_b.predict(rows_a))
    disagreements = int((predictions_a != predictions_b).sum())
    print(
        f"\nF2: {first}'s model and {second}'s model disagree on "
        f"{disagreements}/{len(rows_a)} of {first}'s historical rows"
    )
    assert disagreements > 0, (
        f"{first}'s and {second}'s models agree on all {len(rows_a)} rows "
        f"despite being different artefacts — the fleet may be one model "
        f"copied 45 times"
    )


def hashlib_sha256(artifacts: pathlib.Path) -> str:
    """SHA-256 of an MLflow model directory's serialised booster.

    ``MLmodel`` names the serialised model file in its ``python_function.data``
    entry (``model.ubj`` for the XGBoost flavour recorded here); the
    ``flavors.xgboost.data`` entry names the same file.  The file is read, not
    the directory listing, because two folders holding byte-identical boosters
    are the defect this helper exists to detect.
    """
    import hashlib

    model_file = artifacts / "model.ubj"
    if not model_file.is_file():
        # Fall back to whatever ``MLmodel`` says the serialised payload is, so a
        # future MLflow version that renames it does not silently weaken this.
        names = re.findall(
            r"^\s+data:\s*(\S+\.(?:ubj|json|model|bin))\s*$",
            (artifacts / "MLmodel").read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        candidates = [artifacts / name for name in names if (artifacts / name).is_file()]
        assert candidates, f"no serialised model found in {artifacts}"
        model_file = candidates[0]
    return hashlib.sha256(model_file.read_bytes()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# F3 / F5 — no retained schema column is all-NaN at inference
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_all_nan_model_feature(features_history, mapped_models, monkeypatch):
    """F3/F5: no schema column is all-NaN for a ticker — over the whole fleet.

    The scope is **every mapped ticker** (all 45 models), in both directions.
    The first version of this test looped over three tickers while its name, the
    brief's requirement ("no feature the model expects is all-NaN at inference")
    and the report's §4 claimed the fleet; that gap is exactly what hid the
    finding below, so the fleet is measured in full even though it costs ~45 s.
    A documented sample would have to be believed; this does not.

    Measured on the real artefacts:

    * **Direction 1 — the training frame, *that ticker's own* rows.**  Checking
      a column fleet-wide is not enough: ``debt_to_equity`` is alive for 38
      tickers and all-NaN for the seven banks, and ``sector_*`` are all-NaN for
      ADMR.JK, so the global column is alive while the rows one model was
      actually fitted on carry nothing.  15 of the 45 tickers carry such a
      column; they are pinned exactly in ``TRAINING_ALL_NAN_BY_TICKER`` and are
      pre-existing gaps in the training *data* (fields yfinance does not publish
      for those tickers), not serving defects.
    * **Direction 2 — the frame the serving builder actually produces**, one row
      per ticker from the window the request itself uses — the calendar-day
      extent of ``period="250d"``, not the last 250 rows — with each ticker's
      *own* fundamentals (a single constant dict fills in the columns
      ``fetch_fundamentals`` leaves ``NaN``, which would make the measurement
      easier than production).  ``align_to_schema`` raising
      ``FeatureSchemaError`` for an absent column is the F5 half, and the model
      predicts on the aligned row, so it is a real model input.

    Both directions together are the finding, and it fits in one line: for every
    ticker, the schema columns all-NaN at inference are exactly the columns its
    own training rows are all-NaN for plus the window-truncated
    ``ob_nearest_bearish_distance`` of five tickers, minus the three ADMR sector
    columns serving revives.  All three sets are pinned, with their evidence, in
    ``TRAINING_ALL_NAN_BY_TICKER``, ``WINDOW_ALL_NAN_BY_TICKER`` and
    ``REVIVED_BY_SERVING``; the assertion is the accounting, per ticker, over the
    whole fleet.
    """
    import serve
    from features.schema import align_to_schema
    from model_store import load_model_for

    # ── Direction 1: the training artefact, each ticker's own rows ───────────
    training_all_nan: dict[str, tuple[str, ...]] = {}
    retained_deferred: dict[str, list[str]] = {}
    for ticker in mapped_models:
        rows = features_history[features_history["ticker"] == ticker]
        assert not rows.empty, f"the committed parquet carries no rows for {ticker}"
        _, columns = load_model_for(ticker)
        missing = [c for c in columns if c not in features_history.columns]
        assert not missing, (
            f"{ticker}'s persisted schema names {len(missing)} column(s) the "
            f"committed training artefact does not carry: {missing}"
        )
        empty = tuple(sorted(c for c in columns if rows[c].isna().all()))
        if empty:
            training_all_nan[ticker] = empty
        deferred = sorted(set(DEFERRED_ICT_COLUMNS) & set(columns))
        if deferred:
            retained_deferred[ticker] = deferred

    # That the check is *per ticker* is the point, demonstrated rather than
    # claimed: the fleet-wide column is alive while the seven banks' own rows are
    # not, so a global all-NaN test (what this direction used to be) reports
    # nothing at all.
    assert not features_history["debt_to_equity"].isna().all(), (
        "debt_to_equity is now all-NaN for every ticker, so the per-ticker "
        "demonstration below no longer distinguishes the two checks"
    )
    assert features_history.loc[
        features_history["ticker"] == "BBCA.JK", "debt_to_equity"
    ].isna().all(), (
        "BBCA.JK's own rows now carry debt_to_equity — re-measure "
        "TRAINING_ALL_NAN_BY_TICKER"
    )

    assert training_all_nan == TRAINING_ALL_NAN_BY_TICKER, (
        f"the per-ticker all-NaN training columns changed: measured "
        f"{training_all_nan}, documented {TRAINING_ALL_NAN_BY_TICKER}. This is a "
        f"measured fact about the committed fleet, not a contract: re-measure it "
        f"and update the constant deliberately. A column that is all-NaN across a "
        f"ticker's own rows is a dead feature for that model — it is in the "
        f"schema but every row the model was fitted on carried NaN."
    )

    # The other half of this finding, which the file previously only claimed in
    # prose: the six ICT columns whose index alignment is the *deferred* bug are
    # all-NaN in the training artefact, and the global SHAP selection retained
    # none of them.  That is why no exception is needed for them, so it is
    # asserted rather than assumed.
    assert not retained_deferred, (
        f"a persisted schema retains a deferred ICT column: {retained_deferred}. "
        f"Those six columns ({sorted(DEFERRED_ICT_COLUMNS)}) are all-NaN in the "
        f"committed training artefact because of the index-alignment bug at "
        f"src/features/ict.py:117-129 (deliberately deferred), so a model whose "
        f"schema names one is fed a permanently absent feature."
    )

    # ── Direction 2: the frame the serving builder actually produces ─────────
    _require(STOCKS_CSV, "the committed universe snapshot")
    _require(IHSG_CSV, "the committed IHSG snapshot")

    # Real macro inputs, taken from the same artefact training used.  In
    # production these are warming daily caches, not training columns; sourcing
    # them here keeps the measurement offline without fabricating a value that a
    # schema column would then be judged on.
    usdidr = features_history.groupby(features_history.index)[
        ["usdidr_rate", "usdidr_return"]
    ].first()
    bi_rate = pd.Series(6.0, index=usdidr.index, name="bi_rate_official").ffill()

    # ``monkeypatch``, not assignment: the universe context is cached on the
    # module and a bare ``serve._universe_context = None`` would leak into every
    # later test in the session.
    monkeypatch.setattr(serve, "_usdidr_cache", usdidr)
    monkeypatch.setattr(serve, "_fred_cache", pd.DataFrame())
    monkeypatch.setattr(serve, "_bi_cache", bi_rate)
    monkeypatch.setattr(serve, "_universe_context", None)
    # Sentiment is stubbed to the value the real fetcher returns with no
    # ``NEWSAPI_KEY`` — and stubbing it keeps the real module from writing its
    # ``data/sentiment/`` cache into the repository during a test run.
    import features.fetchers as fetchers

    monkeypatch.setattr(fetchers, "fetch_news_sentiment", lambda ticker: 0.0)

    # The *real* universe loader, reading the on-disk snapshots, so the frame
    # judged below is the frame a request would compute.
    _raw, _ihsg, all_ticker_data = serve.load_universe_context()
    missing_tickers = [t for t in mapped_models if t not in all_ticker_data]
    assert not missing_tickers, (
        f"the committed universe snapshot carries no OHLCV for {missing_tickers}, "
        f"so Direction 2 cannot measure the fleet"
    )

    # ── The served window ────────────────────────────────────────────────────
    # A request downloads ``yf.download(ticker, period="250d")`` — **calendar**
    # days, not rows (``src/serve.py:122``, ``:964``, ``:1057``) — so the frame
    # the builder is handed is the calendar extent of ``250d``, which in the
    # committed snapshot is a fraction of 250 rows.  Deriving it from the last
    # N rows instead is not a smaller version of the same measurement: for
    # CTRA.JK the two windows land on opposite sides of a bearish order block
    # 233 rows back, and the row-count window hides a served NaN.  So the window
    # is located by date and the row count is measured, never assumed.
    #
    # ``SERVED_LOOKBACK_DAYS`` is transcribed from the three ``period="250d"``
    # call sites and cross-checked against them below, so a change there fails
    # here rather than silently re-scoping the map.
    serve_text = SERVE_SOURCE.read_text(encoding="utf-8")
    periods = re.findall(r'period="(\d+)d"', serve_text)
    assert periods == ["250"] * 3, (
        f"src/serve.py's download periods are now {periods}, not three "
        f"period=\"250d\" call sites; the served window this test measures — and "
        f"so WINDOW_ALL_NAN_BY_TICKER, which was measured on it — is defined by "
        f"them. Re-derive the map from the new window, then update this pin."
    )
    SERVED_LOOKBACK_DAYS = int(periods[0])

    benchmark = all_ticker_data["BBCA.JK"].index
    served_cutoff = benchmark[-1] - pd.Timedelta(days=SERVED_LOOKBACK_DAYS)
    # The raw panel still carries every dated row — including the zero-volume
    # ones ``ta.utils.dropna`` removes — so this is the calendar extent a
    # ``period="250d"`` download returns before the loader touches it.  It is
    # the number a reader reaches for first, and the one the previous version of
    # this test assumed the *served* frame to be.
    raw_window_rows = int((_raw.index >= served_cutoff).sum())
    assert raw_window_rows > 0, "the snapshot ends before the served window opens"

    # ``load_universe_context`` runs ``ta.utils.dropna`` per ticker, which drops
    # zero-volume rows as well as NaN ones, so the frame the builder receives is
    # this many rows.  Measured, and bounded rather than pinned to an exact
    # number: the count is a property of the IDX calendar inside the span, so a
    # committed snapshot a few days fresher can move it by a row or two without
    # anything being wrong.  The bound is what makes "250 calendar days" mean
    # something here — a byte-identical check would break on the next ingest and
    # teach the reader to edit the constant.
    served_window_rows = int((benchmark >= served_cutoff).sum())
    assert 156 <= served_window_rows <= 172, (
        f"the served window (last {SERVED_LOOKBACK_DAYS} calendar days of the "
        f"committed snapshot, ending {benchmark[-1].date()}) holds "
        f"{served_window_rows} rows, outside the measured 156-172 band. Either "
        f"the snapshot changed length or the request window changed: re-derive "
        f"WINDOW_ALL_NAN_BY_TICKER before touching this bound."
    )
    print(
        f"\nF3 served window: {SERVED_LOOKBACK_DAYS} calendar days "
        f"({served_cutoff.date()} .. {benchmark[-1].date()}) = {raw_window_rows} "
        f"dated rows in the snapshot, {served_window_rows} after the loader's "
        f"dropna — the frame build_inference_features is handed"
    )

    inference_all_nan: dict[str, tuple[str, ...]] = {}
    revived_by_serving: dict[str, tuple[str, ...]] = {}
    built_widths: set[int] = set()
    checked: set[str] = set()
    started = time.perf_counter()

    for ticker in mapped_models:
        rows = features_history[features_history["ticker"] == ticker]
        model, columns = load_model_for(ticker)
        # The window a request builds: the ticker's own rows — the loader has
        # already applied ``dropna`` — cut at the calendar cutoff the request
        # uses.  Not ``tail(250)`` and not ``tail(served_window_rows)``: those
        # are the same length but different dates.
        price_history = all_ticker_data[ticker]
        price_history = price_history[price_history.index >= served_cutoff].copy()
        price_history.columns = [str(c) for c in price_history.columns]
        assert len(price_history) == served_window_rows, (
            f"{ticker} carries {len(price_history)} rows in the served window, "
            f"not the {served_window_rows} every other ticker carries — the "
            f"snapshot is ragged and the window is not the request's"
        )

        frame = serve.build_inference_features(
            price_history, ticker, _fundamentals_for(rows, ticker)
        )
        assert not frame.empty, f"the serving builder produced no frame for {ticker}"
        built_widths.add(int(frame.shape[1]))

        # ``align_to_schema`` raises ``FeatureSchemaError`` naming every absent
        # column, which is the F5 half of this finding.
        aligned = align_to_schema(frame, columns)

        empty = tuple(sorted(c for c in columns if aligned[c].isna().all()))
        if empty:
            inference_all_nan[ticker] = empty

        # The mirror case: a column that is all-NaN across this ticker's training
        # rows and *populated* in the served row.  Collected so it is pinned too —
        # a value serving supplies for a model that never saw one is the same
        # class of skew as a value training supplied and serving cannot.
        revived = tuple(sorted(set(training_all_nan.get(ticker, ())) - set(empty)))
        if revived:
            revived_by_serving[ticker] = revived

        # A window exception is a *truncation of a real series*, never a column
        # the fleet did not have: it must be in this model's schema and alive in
        # this ticker's own training rows.  Asserted here so the map can never be
        # widened into a blanket exemption for a genuinely dead column — and the
        # contradiction with TRAINING_ALL_NAN_BY_TICKER is what would catch a
        # reader who "simplified" the two maps into one.
        #
        # The ``live > 0`` guard is the one addition that a genuinely
        # unjustified entry cannot reach: every column that is dead in a
        # ticker's own rows is already in TRAINING_ALL_NAN_BY_TICKER, so an
        # entry claiming one is rejected by the contradiction check first.  It
        # is kept as the belt to that braces — it is the assertion that states
        # the definition ("training had the column") in the map's own terms.
        for column in WINDOW_ALL_NAN_BY_TICKER.get(ticker, ()):
            assert column in columns, (
                f"WINDOW_ALL_NAN_BY_TICKER names {column!r} for {ticker}, which "
                f"is not in that model's schema"
            )
            assert column not in TRAINING_ALL_NAN_BY_TICKER.get(ticker, ()), (
                f"{column!r} is listed as a window exception for {ticker} *and* "
                f"as all-NaN in its training rows — the two are different "
                f"findings and the window case is defined by training having the "
                f"column"
            )
            live = int(rows[column].notna().sum())
            assert live > 0, (
                f"{ticker}'s training rows carry no value for the window-excepted "
                f"{column!r}, so it is not a window truncation but a column the "
                f"fleet never had — the audit's F3 class"
            )
            print(
                f"\nF3 {ticker}: {column} is all-NaN in the "
                f"{served_window_rows}-row served window "
                f"({served_cutoff.date()}, period=\"{SERVED_LOOKBACK_DAYS}d\") but "
                f"live in {live}/{len(rows)} of that ticker's own training rows — "
                f"the window truncation pinned in WINDOW_ALL_NAN_BY_TICKER"
            )

        # And the row is genuinely predictable: an all-NaN frame would raise
        # here rather than pass an empty-looking assertion.
        prediction = np.asarray(model.predict(aligned))
        assert prediction.shape == (1,), (ticker, prediction.shape)
        checked.add(ticker)

    elapsed = time.perf_counter() - started
    print(
        f"\nF3 direction 2 (the whole fleet): {len(checked)} tickers in "
        f"{elapsed:.1f}s — {len(checked) - len(inference_all_nan)} with no all-NaN "
        f"schema column; {_pair_count(TRAINING_ALL_NAN_BY_TICKER)} ticker/column "
        f"pairs all-NaN in training over {len(TRAINING_ALL_NAN_BY_TICKER)} "
        f"tickers; {_pair_count(WINDOW_ALL_NAN_BY_TICKER)} window-truncated over "
        f"{len(WINDOW_ALL_NAN_BY_TICKER)} (window: "
        f"{SERVED_LOOKBACK_DAYS} calendar days = {served_window_rows} rows); "
        f"{_pair_count(REVIVED_BY_SERVING)} revived by serving over "
        f"{len(REVIVED_BY_SERVING)}; built column counts {sorted(built_widths)}"
    )
    assert checked == set(mapped_models), (
        f"Direction 2 measured {sorted(checked)}, not the whole fleet "
        f"{sorted(mapped_models)}"
    )

    # The three documented maps together are the assertion: for every ticker, the
    # columns all-NaN at inference are exactly the columns its own training rows
    # are all-NaN for, plus the documented window truncations, minus the columns
    # serving revives.  Anything outside that accounting — a column training
    # populated and serving cannot, for a ticker not in the window map — is the
    # audit's F3 class and fails here.
    expected_inference_all_nan: dict[str, tuple[str, ...]] = {}
    for ticker in (set(TRAINING_ALL_NAN_BY_TICKER)
                   | set(WINDOW_ALL_NAN_BY_TICKER)
                   | set(REVIVED_BY_SERVING)):
        columns = tuple(sorted(
            (set(TRAINING_ALL_NAN_BY_TICKER.get(ticker, ()))
             | set(WINDOW_ALL_NAN_BY_TICKER.get(ticker, ())))
            - set(REVIVED_BY_SERVING.get(ticker, ()))
        ))
        if columns:
            expected_inference_all_nan[ticker] = columns

    assert inference_all_nan == expected_inference_all_nan, (
        f"the fleet's inference-time all-NaN columns changed: measured "
        f"{inference_all_nan}, documented {expected_inference_all_nan}. A schema "
        f"column that is all-NaN at inference and alive in the ticker's own "
        f"training rows is exactly the audit's F3 class: measure it, report it, "
        f"and if it is a real window effect add it to WINDOW_ALL_NAN_BY_TICKER "
        f"with its evidence — do not widen the assertion for the fleet."
    )

    assert revived_by_serving == REVIVED_BY_SERVING, (
        f"the columns the serving builder supplies for a ticker whose own training "
        f"rows are all-NaN changed: measured {revived_by_serving}, documented "
        f"{REVIVED_BY_SERVING}. This is the mirror of the audit's F3 class: the "
        f"model was fitted with the column constant-NaN and is now served real "
        f"values it was never trained on. Measure it, report it, and change the "
        f"map only with that evidence."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# F6 / F7 — the cache is keyed by name, schema and window
# ═══════════════════════════════════════════════════════════════════════════════

def test_cache_by_name():
    """F6/F7: a real round-trip by name; a mismatch is a miss, not a permutation.

    The original defect zipped a cached *vector* positionally onto a different
    model's column order, so a value computed for one feature was consumed as
    another.  The pins below use deliberately distinctive values — each one
    identifyable by name alone — so a positional read could not pass by luck,
    and they assert the *stronger* property that no permuted series is ever
    returned: a mismatch answers ``None``.
    """
    import serve

    schema = ["ms_swing_low", "ms_swing_high", "ret_1d", "momentum_rsi",
              "usdidr_rate", "market_breadth"]
    window = "0900"
    row = pd.Series({
        "ms_swing_low": 1.0,
        "ms_swing_high": 2.0,
        "ret_1d": 3.0,
        "momentum_rsi": 4.0,
        "usdidr_rate": 15_500.0,
        "market_breadth": 6.0,
    })

    serve._cache_put("BBCA.JK", row, schema, window)
    got = serve._cache_get("BBCA.JK", schema, window)

    assert got is not None, "a row just written was not readable"
    assert list(got.index) == schema, (
        f"the cached row came back in a different order: {list(got.index)}"
    )
    for name in schema:
        assert got[name] == row[name], (
            f"column {name!r} round-tripped as {got[name]!r}, not {row[name]!r}"
        )

    # A positional read would make each name carry its neighbour's value; pin
    # that the values really are distinct, so the assertion above has teeth.
    assert len(set(row.to_numpy())) == len(schema)

    # A window change is a *different key*, hence a miss.
    assert serve._cache_get("BBCA.JK", schema, "1330") is None, (
        "a row cached for the 0900 window was served for the 1330 window"
    )

    # A schema change is a miss, both when a column is added...
    assert serve._cache_get("BBCA.JK", schema + ["volume_em"], window) is None, (
        "a row cached for a 6-column schema was served for a 7-column one"
    )
    # ...and when the same names arrive in another order.  This is the exact
    # shape of the original defect: same values, different column order.
    reversed_schema = list(reversed(schema))
    assert serve._cache_get("BBCA.JK", reversed_schema, window) is None, (
        "a reordered schema was a cache *hit* — the positional-zip defect"
    )

    # The fingerprint is what separates them, and it is order-sensitive by
    # design: XGBoost is fitted on a positionally-scaled matrix.
    assert serve._cache_schema_hash(schema) != serve._cache_schema_hash(reversed_schema)

    # A name-keyed write is never silently permuted on the way back in: the
    # single frame builder consumes the row and lands it in schema order.
    frame = serve._prediction_frame(got, schema)
    assert list(frame.columns) == schema
    for name in schema:
        assert frame.iloc[0][name] == row[name]

    # A NaN is a legitimate value and must survive as one — an all-NaN feature
    # is what F3 forbids, but a per-row NaN is what training itself carries.
    serve._cache_put("BBCA.JK", pd.Series({**row.to_dict(), "momentum_rsi": np.nan}),
                     schema, window)
    nan_row = serve._cache_get("BBCA.JK", schema, window)
    assert nan_row is not None
    assert np.isnan(nan_row["momentum_rsi"])
    assert nan_row["ret_1d"] == 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# F8 — degradation is visible, and the dead calibration symbol is gone
# ═══════════════════════════════════════════════════════════════════════════════

#: The name the deleted calibration block called.  It exists nowhere in
#: scikit-learn; the symbol the code wanted was ``CalibratedClassifierCV``.  It
#: may be *named in prose* — a tombstone comment explains why the block was
#: deleted — which is exactly why the assertion below walks the AST.
NEVER_EXISTED = "CalibratedCVClassifier"
CALIBRATION_MODULE = "sklearn.calibration"

SCHEMA = ["rsi"]


class _StubModel:
    """The two-method surface every predicting path uses."""

    def predict(self, frame):
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[0.4, 0.6]])


class _FakeAnalyzer:
    """``vaderSentiment``'s analyzer, as ``analyze_sentiment`` uses it."""

    COMPOUND = 0.4

    def polarity_scores(self, text):
        return {"compound": self.COMPOUND}


class _FakeResponse:
    """``requests.Response``, as ``news_sentiment.fetch_news`` uses it."""

    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _symbols_in_source(source: str) -> set[str]:
    """Every name a module's source binds or refers to, as text.

    Bare names, dotted attribute chains, ``ClassDef``/``FunctionDef`` names, and
    import aliases (both spellings).  Docstrings, comments and plain string
    literals contribute nothing — which is the whole point: recording *why* a
    symbol was deleted must not fail the test that asserts it is gone.  This is
    the same structural check ``tests/test_degradation_visible.py`` establishes.
    """
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            parts, cursor = [], node
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name):
                parts.append(cursor.id)
            found.add(".".join(reversed(parts)))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            found.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                found.add(alias.name)
                if alias.asname:
                    found.add(alias.asname)
    return found


def _calibration_offenders(source: str) -> list[str]:
    """Names in ``source`` that reach for the deleted calibration code.

    Two shapes, because ``from sklearn.calibration import X`` puts the module in
    an ``ImportFrom.module`` — a plain ``str`` that a symbol walk cannot see —
    while ``import sklearn.calibration`` and ``sklearn.calibration.X`` are
    aliases and attribute chains that it can.
    """
    found = [
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith(CALIBRATION_MODULE)
    ]
    found += [s for s in _symbols_in_source(source) if s.startswith(CALIBRATION_MODULE)]
    found += [
        s for s in _symbols_in_source(source)
        if NEVER_EXISTED in s or s.endswith("CalibratedClassifierCV")
    ]
    return sorted(set(found))


def _healthy_news(monkeypatch, cache_dir):
    """Put the *real* news seam in the state a successful production fetch leaves."""
    monkeypatch.setattr(news_sentiment, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "test-key")
    monkeypatch.setattr(news_sentiment, "VADER_AVAILABLE", True)
    monkeypatch.setattr(news_sentiment, "SentimentIntensityAnalyzer", _FakeAnalyzer,
                        raising=False)
    monkeypatch.setattr(news_sentiment, "_sentiment_analyzer", None)
    monkeypatch.setattr(
        news_sentiment, "requests",
        types.SimpleNamespace(get=lambda url, params=None, timeout=None: _FakeResponse(
            200, {"articles": [{"title": "Bank BCA profit rises"}]}
        )),
    )
    return news_sentiment


def _outage_news(monkeypatch, cache_dir):
    """The plain-venv outage: no ``NEWSAPI_KEY``, so the real seam reports it."""
    monkeypatch.setattr(news_sentiment, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(news_sentiment, "NEWSAPI_KEY", "")
    monkeypatch.setattr(news_sentiment, "VADER_AVAILABLE", False)
    monkeypatch.setattr(news_sentiment, "_sentiment_analyzer", None)
    return news_sentiment


def _serving_client(tmp_path, monkeypatch):
    """A ``TestClient`` over the real app, with the cache primed for a hit.

    ``_apply_sentiment_overlay`` is not stubbed, and neither is the fetcher it
    reads: the tests below drive the real
    ``features.fetchers.fetch_news_sentiment_status`` against the real
    ``news_sentiment`` module, so they exercise the seam production uses rather
    than a stub whose shape production never produces.
    """
    import serve
    from datetime import date as _date

    monkeypatch.setattr(serve, "PRED_CACHE_DB", str(tmp_path / "prediction_cache.db"))
    monkeypatch.setattr(
        serve, "load_best_model", lambda ticker: (_StubModel(), list(SCHEMA))
    )
    # Task 2.4: the serving path resolves fundamentals through
    # ``serve.get_fundamentals`` — the status seam, because the eight values
    # cannot say whether they are a measurement.  It resolves them *inside* the
    # cache-hit branch (Task 2.4 fix round 1), not before the cache is consulted:
    # a hit never builds features, so that branch is the only place its response
    # can learn the status, and resolving outside it would make every miss pay
    # for two ``.info`` calls.  This suite's network guard makes ``yf.Ticker``
    # *raise*, so without a stand-in here the request would legitimately report
    # ``degraded: ["fundamentals"]`` and this test's healthy direction would be
    # measuring the guard rather than the sentiment seam.  No values are needed:
    # ``SCHEMA`` is ``["rsi"]``, so no fundamental column is in this model's
    # contract.
    monkeypatch.setattr(serve, "get_fundamentals", lambda ticker: ({}, True, None))
    # ``/predict`` adds ``refresh_daily_cache`` as a background task.  Pinning
    # the daily-cache date to today makes that task return immediately instead of
    # reaching for Yahoo Finance and bi.go.id after the response.
    monkeypatch.setattr(serve, "_cache_date", str(_date.today()))
    # ``monkeypatch`` rather than ``serve.model_cache.clear()``: clearing the
    # module-level dict mutates shared state that no fixture restores, so an
    # empty cache would leak into every later test in the session.  Replacing the
    # object starts this test clean and restores the original dict — contents and
    # all — at teardown.
    monkeypatch.setattr(serve, "model_cache", {})
    serve._cache_put("BBCA.JK", pd.Series({"rsi": 0.5}), SCHEMA,
                     serve.CACHE_WINDOW_DEFAULT)
    return TestClient(serve.app)


def test_degraded_flag_present(tmp_path, monkeypatch):
    """F8: a healthy request says so; a sentiment outage says *which* part failed.

    Both directions are driven through the real status seam
    (``features.fetchers.fetch_news_sentiment_status``).  No test here stubs a
    ``raise``: the value-only fetcher returns ``0.0`` for a missing key, a dead
    endpoint, a 429 and an absent ``vaderSentiment`` alike and raises for none
    of them, so a raising stub would prove nothing about production — that dead
    ``except`` was the defect.

    The dead calibration symbol is asserted absent structurally, over the AST of
    both ``src/train.py`` and ``src/serve.py``, so a comment naming it costs
    nothing.  A raw substring scan would fail on the tombstones that explain the
    deletion.
    """
    # ── Structural: the dead calibration code is not executed anywhere ───────
    for source_path in (TRAIN_SOURCE, SERVE_SOURCE):
        text = source_path.read_text(encoding="utf-8")
        offenders = _calibration_offenders(text)
        assert offenders == [], (
            f"the dead calibration code is back as code in "
            f"{source_path.name}: {offenders}"
        )

    # The check has teeth in both directions, demonstrated on the same function
    # the assertions above call — not on a paraphrase of it.
    assert _calibration_offenders("# nothing from sklearn.calibration here\n") == []
    assert _calibration_offenders(f'DOC = "{NEVER_EXISTED} was deleted"\n') == []
    assert _calibration_offenders(
        f"final = {NEVER_EXISTED}(estimator=xgb)\n") != []
    assert _calibration_offenders(
        "from sklearn.calibration import CalibratedClassifierCV\n") != []

    # ── Direction 1: a healthy request is not degraded ───────────────────────
    _healthy_news(monkeypatch, tmp_path / "sentiment-healthy")
    client = _serving_client(tmp_path, monkeypatch)

    healthy = client.post("/predict", json={"ticker": "BBCA.JK"})
    assert healthy.status_code == 200, healthy.text
    healthy_body = healthy.json()
    assert healthy_body["degraded"] == [], healthy_body
    assert healthy_body["sentiment_score"] == _FakeAnalyzer.COMPOUND, healthy_body

    # ── Direction 2: a sentiment outage is named in `degraded` ───────────────
    # A *separate* cache directory: the healthy request above wrote a marked,
    # available reading for this ticker, and ``get_ticker_sentiment`` believes a
    # marked cache entry.  Sharing the directory would answer the outage from
    # the healthy run's cache and the seam under test would never be reached.
    _outage_news(monkeypatch, tmp_path / "sentiment-outage")
    degraded_client = _serving_client(tmp_path, monkeypatch)

    outage = degraded_client.post("/predict", json={"ticker": "BBCA.JK"})
    assert outage.status_code == 200, outage.text
    outage_body = outage.json()
    assert outage_body["degraded"] == ["sentiment"], outage_body
    # The prediction itself survives — sentiment is an overlay, never a blocker.
    assert outage_body["prediction"] in (0, 1), outage_body
    # ...and the safe default is not passed off as a measurement.
    assert outage_body["sentiment_score"] == 0.0, outage_body


def test_the_import_wrapper_hides_nothing_else():
    """The one wrapped import captured the one known starlette message."""
    assert set(_IMPORT_WARNINGS) <= {EXPECTED_IMPORT_WARNING}, _IMPORT_WARNINGS
    assert len(_IMPORT_WARNINGS) <= 1, _IMPORT_WARNINGS


# ═══════════════════════════════════════════════════════════════════════════════
# F11 — every third-party import is a declared dependency
# ═══════════════════════════════════════════════════════════════════════════════
#
# Task 0.3 found ``requests``, ``PyYAML``, ``prometheus_client`` and ``joblib``
# imported but undeclared; ``holidays`` and ``tzdata`` were added later.  The
# manifest test the plan referenced (``tests/test_manifest.py``) does not exist,
# so this is its real home: a real AST scan of the repository's ``*.py`` files.

#: Directories that are not this repository's source.  ``venv`` is the
#: interpreter's site-packages (the task's explicit exclusion); the rest are
#: caches, VCS metadata, the gitignored ``app/`` stray, local experiment stores
#: and the rescue plan's own prose tree.
SCAN_EXCLUDED_DIRS = frozenset({
    "venv", ".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".dvc", "app", ".superpowers", "mlruns", "data",
    "node_modules", "build", "dist", ".idea", ".vscode",
})

#: Top-level names this repository defines itself, so they are not distributions
#: and cannot be declared in a requirements file.
LOCAL_IMPORT_NAMES = frozenset({
    "config", "features", "ingest", "market_calendar", "model_store",
    "news_sentiment", "serve", "train", "backtest",
})

#: Import name -> the distribution that provides it.  Every entry is a *mapping*
#: where the two differ; an identity mapping is still written out so the table
#: is the one place a reader checks.  A distribution installs its import name(s)
#: and may install more than one, and a distribution name may differ from every
#: import name it provides — that is why this mapping exists rather than a
#: string comparison against ``requirements.txt``.
IMPORT_TO_DISTRIBUTION = {
    "aiohttp": "aiohttp",
    "fastapi": "fastapi",
    "fredapi": "fredapi",
    "holidays": "holidays",
    "joblib": "joblib",
    "lightgbm": "lightgbm",
    "matplotlib": "matplotlib",
    "mlflow": "mlflow",
    "numpy": "numpy",
    "optuna": "optuna",
    "pandas": "pandas",
    "prometheus_client": "prometheus-client",
    "prometheus_fastapi_instrumentator": "prometheus-fastapi-instrumentator",
    "pydantic": "pydantic",
    "pytest": "pytest",
    "pytrends": "pytrends",
    "requests": "requests",
    "shap": "shap",
    "sklearn": "scikit-learn",
    "ta": "ta",
    "uvicorn": "uvicorn",
    "vaderSentiment": "vaderSentiment",
    "xgboost": "xgboost",
    "yaml": "PyYAML",
    "yfinance": "yfinance",
}

#: The manifest files a scan may consult.  ``requirements.txt`` is the runtime
#: list the constraint names; ``-dev`` and ``-docker`` are consulted too, so a
#: test-only import (``pytest``) or the Docker-only analyser
#: (``vaderSentiment``) is declared rather than silently excused.
MANIFEST_FILES = ("requirements.txt", "requirements-dev.txt", "requirements-docker.txt")


def _normalise_distribution(name: str) -> str:
    """PEP 503 normalisation: ``PyYAML`` and ``pyyaml`` are one distribution."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_distributions(manifest: pathlib.Path) -> dict[str, str]:
    """``normalised distribution -> the name as written`` from one manifest.

    Comments, blank lines and pip's own option lines (``-r``, ``--hash``) are
    skipped; the requirement's name is everything before its first version
    specifier, environment marker, extra or whitespace.
    """
    declared: dict[str, str] = {}
    if not manifest.is_file():
        return declared
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0].strip()
        if name:
            declared.setdefault(_normalise_distribution(name), name)
    return declared


def _declared_everywhere() -> dict[str, tuple[str, str]]:
    """``normalised distribution -> (name as written, manifest that declared it)``."""
    declared: dict[str, tuple[str, str]] = {}
    for filename in MANIFEST_FILES:
        for normalised, written in _declared_distributions(_REPO_ROOT / filename).items():
            declared.setdefault(normalised, (written, filename))
    return declared


def _iter_repo_sources() -> list[pathlib.Path]:
    """Every ``*.py`` file this repository owns, sorted for a stable report."""
    found: list[pathlib.Path] = []
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SCAN_EXCLUDED_DIRS and not d.startswith(".")
        )
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                found.append(pathlib.Path(dirpath) / filename)
    return sorted(found)


def _imported_top_level_names(sources: list[pathlib.Path]) -> dict[str, set[str]]:
    """``top-level import name -> the files that import it``."""
    imported: dict[str, set[str]] = {}
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = str(path.relative_to(_REPO_ROOT))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.setdefault(alias.name.split(".")[0], set()).add(relative)
            elif isinstance(node, ast.ImportFrom):
                # ``node.level`` non-zero is a relative import: it names a
                # sibling module of this package, never a distribution.
                if node.level or not node.module:
                    continue
                imported.setdefault(node.module.split(".")[0], set()).add(relative)
    return imported


def test_all_imports_declared():
    """F11: the repository imports nothing a requirements file does not declare.

    The scan is structural (AST over every ``*.py`` the repository owns, with
    ``venv/`` and the other non-source trees excluded) rather than a grep, so a
    module named only in a comment or a docstring is not a dependency and a
    module imported inside a function *is*.

    Scope, printed to the test log: every third-party top-level name found, and
    the distribution each one maps to.  An import with no entry in
    :data:`IMPORT_TO_DISTRIBUTION` is a failure rather than a silent skip —
    otherwise adding a new dependency would quietly leave the scan unable to
    judge it, which is the "passes CI and proves nothing" shape this whole task
    exists to remove.

    The scan's *own* scope is pinned before anything is concluded from it:
    widening :data:`SCAN_EXCLUDED_DIRS` is a one-line edit to this file that
    would otherwise leave every assertion below green while the scan judged
    nothing.
    """
    sources = _iter_repo_sources()
    assert sources, "the scan found no source files — it is not judging anything"
    for known in ("src/serve.py", "src/features/ict.py",
                  "tests/test_audit_regressions.py"):
        assert _REPO_ROOT / known in sources, (
            f"{known} is not in the F11 scan scope — the scan's exclusions were "
            f"widened and it is no longer judging the repository it claims to"
        )

    imported = _imported_top_level_names(sources)
    stdlib = set(sys.stdlib_module_names)
    third_party = {
        name: files
        for name, files in imported.items()
        if name not in stdlib and name not in LOCAL_IMPORT_NAMES
    }
    assert third_party, "the scan found no third-party imports at all"

    declared = _declared_everywhere()
    assert declared, "no requirements manifest could be read"

    unmapped = sorted(name for name in third_party if name not in IMPORT_TO_DISTRIBUTION)
    assert not unmapped, (
        f"third-party import name(s) {unmapped} have no "
        f"IMPORT_TO_DISTRIBUTION entry, so this scan cannot judge whether they "
        f"are declared. Add the mapping (and the dependency, if it is missing): "
        f"{ {name: sorted(third_party[name]) for name in unmapped} }"
    )

    undeclared: dict[str, list[str]] = {}
    scope: dict[str, str] = {}
    for name in sorted(third_party):
        distribution = IMPORT_TO_DISTRIBUTION[name]
        entry = declared.get(_normalise_distribution(distribution))
        scope[name] = f"{distribution} ({entry[1]})" if entry else distribution
        if entry is None:
            undeclared[name] = sorted(third_party[name])

    print(
        "\nF11 third-party imports checked "
        f"({len(scope)} names in {len(sources)} files) — "
        f"import name -> declared distribution (manifest):\n  "
        + "\n  ".join(f"{name} -> {where}" for name, where in scope.items())
    )

    assert not undeclared, (
        "third-party module(s) are imported but no requirements file declares "
        f"the distribution that provides them: {undeclared}. The imports were "
        f"found in { {name: third_party[name] for name in undeclared} }"
    )

    # The four Task 0.3 named, plus the two added later — pinned against
    # ``requirements.txt`` itself, not against "some manifest declares it": the
    # constraint is that the *runtime* list carries them, and a declaration that
    # only exists in ``requirements-dev.txt`` would satisfy a looser pin while
    # leaving a plain ``pip install -r requirements.txt`` broken.
    runtime = _declared_distributions(_REPO_ROOT / "requirements.txt")
    assert runtime, "requirements.txt could not be read"
    for distribution in ("requests", "PyYAML", "prometheus-client", "joblib",
                         "holidays", "tzdata"):
        assert _normalise_distribution(distribution) in runtime, (
            f"{distribution} is no longer declared in requirements.txt"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# F12 — a batch failure is an error, not a signal
# ═══════════════════════════════════════════════════════════════════════════════

#: The fields that make a response *tradable*.  A failure response must carry
#: none of them: it used to answer ``prediction: 0`` (SELL) and
#: ``probability_up: 0.0`` under HTTP 200.
SIGNAL_SHAPED_FIELDS = ("prediction", "signal", "probability_up")


def _stub_batch_io(monkeypatch):
    """Make the batch path's I/O local, synchronous and inert.

    The price frame carries the full OHLCV surface, not just ``Close``: the batch
    path runs the real feature builder on a ticker it *can* build for, so that
    the unmapped-ticker case below fails where the audit found the defect — at
    model resolution — rather than for an unrelated reason inside the builder.
    """
    import serve

    async def inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    index = pd.bdate_range("2024-01-01", periods=60)
    price = np.arange(60.0) + 100.0
    price_frame = pd.DataFrame(
        {
            "Open": price,
            "High": price + 0.5,
            "Low": price - 0.5,
            "Close": price,
            "Volume": np.full(60, 1_000_000.0),
        },
        index=index,
    )
    monkeypatch.setattr(serve.asyncio, "to_thread", inline)
    monkeypatch.setattr(serve.yf, "download", lambda *a, **k: price_frame.copy())
    # Task 2.4: the serving path reads ``serve.get_fundamentals`` — the status
    # seam, because the eight values cannot say whether they are a measurement —
    # so the stub is on that seam rather than on the value-only fetcher, which
    # the serving path no longer calls.  The stand-in is exactly what the old
    # ``fetch_fundamentals`` stub produced (no values) plus the availability it
    # silently implied.
    monkeypatch.setattr(serve, "get_fundamentals", lambda ticker: ({}, True, None))
    # Task 2.4 fix round 1: ``/predict/batch`` builds a frame, so it now also
    # reports a macro outage (``pipe`` writes ``usdidr_return = 0.0`` with no
    # USD/IDR frame, and the response names that under ``"macro"``).  These tests
    # are about the *failure-response shape*, not about a cold macro cache, so
    # the day's healthy outcome is stated here rather than inferred from whatever
    # the surrounding session left on the module.
    from datetime import date as _date

    monkeypatch.setattr(serve, "_macro_outcomes", {str(_date.today()): None})
    monkeypatch.setattr(serve, "_usdidr_cache", pd.DataFrame(
        columns=["usdidr_rate", "usdidr_return"]))


def test_batch_failure_is_error(tmp_path, monkeypatch):
    """F12: a ticker that cannot be predicted is an error with no signal in it.

    Two failure modes are driven, because the defect was about the *shape of the
    failure response* rather than about one cause:

    * an **unmapped** ticker — ``load_best_model`` raises ``ModelNotFound``,
      which the handler re-raises as **exactly a 404** (``src/serve.py:1103``
      re-raises ``HTTPException``, so ``src/serve.py:511``'s status reaches the
      client unchanged; the pre-fix handler re-raised it too).  Pre-fix the
      *model substitution* meant a ticker with no model was quietly answered by
      another ticker's; the pins here are that it cannot be answered at all,
      that the status is 404 and not a tolerated 502, and that the body carries
      no tradable field;
    * a ticker whose **feature build fails** — the cause the audit measured, and
      the branch that used to answer ``prediction: 0`` (SELL),
      ``probability_up: 0.0`` and the error text stuffed into ``signal``, all
      under HTTP 200.  That one is **exactly a 502** (``src/serve.py:1113``),
      and the two statuses are asserted separately because they are separate
      causes: accepting either for either is weaker than this docstring.

    The final block constructs the deleted shape — the exact ``PredictResponse``
    the old handler returned — and shows it *does* carry the signal fields the
    live responses do not, so the absence assertions are not vacuous; the field
    list they loop over is itself pinned, so emptying it cannot disable them.
    """
    import serve
    from datetime import date as _date
    from model_store import ModelNotFound, resolve_model_folder

    # Anti-vacuity: the absence assertions below are loops over this constant, so
    # an emptied or shortened constant would silently disable them.
    assert len(SIGNAL_SHAPED_FIELDS) == 3, SIGNAL_SHAPED_FIELDS
    assert set(SIGNAL_SHAPED_FIELDS) == {"prediction", "signal", "probability_up"}, (
        SIGNAL_SHAPED_FIELDS
    )

    # ``/predict/batch`` schedules ``refresh_daily_cache``; pinning the date to
    # today makes it return immediately rather than reach for bi.go.id.
    monkeypatch.setattr(serve, "_cache_date", str(_date.today()))
    client = TestClient(serve.app, raise_server_exceptions=False)

    # ── Failure 1: a ticker the index does not map ───────────────────────────
    unmapped = "ZZZZ.JK"
    with pytest.raises(ModelNotFound):
        resolve_model_folder(unmapped)

    _stub_batch_io(monkeypatch)
    response = client.post("/predict/batch", json={"tickers": [unmapped]})

    # Exact, not tolerant: the handler is deterministic here, so 404 is the only
    # correct status for a client error and 502 belongs to Failure 2 alone.
    assert response.status_code == 404, response.text
    body = response.json()
    assert "detail" in body, body
    assert unmapped in body["detail"], body
    print(f"\nF12 unmapped ticker: HTTP {response.status_code} {response.text}")

    # ── Failure 2: the feature build fails ───────────────────────────────────
    def explode(*args, **kwargs):
        raise ValueError("feature build failed for BROKEN.JK")

    monkeypatch.setattr(serve, "build_inference_features", explode)
    broken = client.post("/predict/batch", json={"tickers": ["BROKEN.JK"]})

    # The other cause, and its own exact status — 502 is not an alternative to
    # the 404 above, it is what a server-side build failure answers.
    assert broken.status_code == 502, broken.text
    broken_body = broken.json()
    assert "detail" in broken_body, broken_body
    assert "BROKEN.JK" in broken_body["detail"], broken_body
    print(f"F12 failing feature build: HTTP {broken.status_code} {broken.text}")

    # Neither failure response carries a field a caller could trade on.  The
    # pre-fix shape is built explicitly right after, so this is a comparison and
    # not a string that happens to be absent.
    for field in SIGNAL_SHAPED_FIELDS:
        assert field not in response.text, (
            f"the failure response for {unmapped} carries a signal-shaped "
            f"field {field!r}: {response.text}"
        )
        assert field not in broken.text, (
            f"the failure response for BROKEN.JK carries a signal-shaped field "
            f"{field!r}: {broken.text}"
        )

    deleted_shape = serve.PredictResponse(
        ticker="BROKEN.JK",
        prediction=0,
        probability_up=0.0,
        signal="ERROR: feature build failed for BROKEN.JK",
    ).model_dump()
    deleted_text = json.dumps(deleted_shape)
    for field in SIGNAL_SHAPED_FIELDS:
        assert field in deleted_text, (
            f"the deleted response shape no longer contains {field!r}, so the "
            f"absence assertions above prove nothing: {deleted_text}"
        )

    # ── The control: the same endpoint still answers a predictable ticker ────
    monkeypatch.setattr(serve, "load_best_model",
                        lambda ticker: (_StubModel(), list(SCHEMA)))
    monkeypatch.setattr(
        serve, "build_inference_features",
        lambda df, ticker, fundamentals: pd.DataFrame([{"rsi": 0.5}]),
    )
    _healthy_news(monkeypatch, tmp_path / "sentiment-healthy")

    healthy = client.post("/predict/batch", json={"tickers": ["BBCA.JK"]})
    assert healthy.status_code == 200, healthy.text
    predictions = healthy.json()["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["prediction"] == 1
    assert predictions[0]["degraded"] == []
