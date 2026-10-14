"""Task 3.1 — the backtest's cost model is per side and models the IDX sell tax.

The audit found one flat ``cost_per_trade = 0.0015`` charged at entry, at exit and
at the final forced close.  IDX is not symmetric: brokerage is paid on both legs,
and a **0.1% final sales tax** is paid on the sell only, so a round trip costs
strictly more on the way out than on the way in.  Slippage — the difference
between the modelled price and the fill — was not modelled at all, and Phase 3
measures the strategy at three slippage levels.

These tests pin the arithmetic and the constants; the three call sites in
``backtest_ticker`` are covered by the module-level import check and by the
rescue's smoke run, which exercises the real path end to end.
"""
from __future__ import annotations

import warnings

import pytest

# ``src/backtest.py`` imports mlflow, which imports mlflow.tracing, which imports
# opentelemetry, whose runtime-context loader still calls the removed
# ``importlib.metadata`` dict interface.  On Python 3.11 that emits
# ``DeprecationWarning: SelectableGroups dict interface is deprecated. Use
# select.`` from opentelemetry/util/_importlib_metadata.py:32 — upstream, not this
# repository.  The suite runs with ``filterwarnings = ["error"]`` (Task 1.1's
# deliberate choice), so without a filter the import aborts collection.
#
# This one is a ``warnings.catch_warnings`` block rather than the ``pytestmark``
# that tests/test_feature_schema.py, tests/test_index_written.py and
# tests/test_index_rebuild.py use: ``pytestmark`` is only consulted once the
# module object is built, so it cannot cover a warning raised *while* the module
# is being imported (verified — the pytestmark-only form still errors with
# "Interrupted: 1 error during collection").  The block is scoped to that single
# message and category, is exited before any test runs, and therefore restores
# the suite's escalation; every other warning — including any this task's own
# code raises — still fails the run.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="SelectableGroups dict interface is deprecated",
        category=DeprecationWarning,
    )
    import backtest
    from backtest import DEFAULT_BUY_COST, DEFAULT_SELL_COST, costs_for

# A round-lot-ish notional in the backtest's own units (IDR).
VALUE = 1_000_000


# ── The two tests from the brief ─────────────────────────────────────────────

def test_sell_costs_more_than_buy():
    assert costs_for(VALUE, side="sell") > costs_for(VALUE, side="buy")


def test_slippage_is_applied_per_side():
    assert costs_for(VALUE, side="buy", slippage_bps=10) > costs_for(VALUE, side="buy")


# ── The rates themselves ─────────────────────────────────────────────────────

def test_default_rates_are_the_idx_rates():
    """Brokerage 0.15% both sides; the sell adds the 0.1% final sales tax."""
    assert DEFAULT_BUY_COST == 0.0015
    assert DEFAULT_SELL_COST == 0.0025


def test_flat_cost_per_trade_is_gone():
    """The single flat fee is replaced, not shadowed by the new constants."""
    assert not hasattr(backtest, "DEFAULT_COST_PER_TRADE")
    assert "DEFAULT_COST_PER_TRADE" not in backtest.backtest_ticker.__code__.co_consts
    assert "cost_per_trade" not in backtest.backtest_ticker.__code__.co_varnames


def test_cost_is_the_rate_times_value():
    assert costs_for(VALUE, side="buy") == DEFAULT_BUY_COST * VALUE
    assert costs_for(VALUE, side="sell") == DEFAULT_SELL_COST * VALUE


# ── The right side, not just "the more expensive one" ────────────────────────

def test_sell_is_charged_the_sell_rate_and_buy_the_buy_rate():
    """Pins the mapping side -> rate, so swapping the two is caught."""
    assert costs_for(VALUE, side="sell") - costs_for(VALUE, side="buy") == (
        DEFAULT_SELL_COST - DEFAULT_BUY_COST
    ) * VALUE


def test_the_sell_premium_is_the_one_tenth_percent_final_sales_tax():
    assert DEFAULT_SELL_COST - DEFAULT_BUY_COST == 0.001


# ── Slippage ─────────────────────────────────────────────────────────────────

def test_slippage_scales_linearly():
    """The slippage *term* is linear in bps — 10 bps costs twice 5, five times 2.

    Measured as the increment over the zero-slippage cost, not as the total:
    the total adds that term to a fixed rate, so the total itself cannot be
    proportional to the bps.
    """
    for side in ("buy", "sell"):
        base = costs_for(VALUE, side=side)
        slip2 = costs_for(VALUE, side=side, slippage_bps=2) - base
        slip5 = costs_for(VALUE, side=side, slippage_bps=5) - base
        slip10 = costs_for(VALUE, side=side, slippage_bps=10) - base
        # approx: bps/10_000 is not exactly representable in binary, so a
        # subtraction of two nearby floats leaves a ~1e-13 residue.
        assert slip10 == pytest.approx(2 * slip5)
        assert slip10 == pytest.approx(5 * slip2)
        assert slip2 == pytest.approx(0.0002 * VALUE)


def test_slippage_is_proportional_to_the_notional():
    """Twice the traded value pays twice the slippage, on either side."""
    for side in ("buy", "sell"):
        one = costs_for(VALUE, side=side, slippage_bps=10) - costs_for(VALUE, side=side)
        two = costs_for(2 * VALUE, side=side, slippage_bps=10) - costs_for(
            2 * VALUE, side=side
        )
        assert two == 2 * one


def test_slippage_is_added_to_the_rate_not_multiplied_by_it():
    """One basis point is one hundredth of a percent of the traded value."""
    assert costs_for(VALUE, side="buy", slippage_bps=100) == pytest.approx(
        costs_for(VALUE, side="buy") + 0.01 * VALUE
    )


def test_zero_and_no_slippage_agree():
    assert costs_for(VALUE, side="buy", slippage_bps=0) == costs_for(VALUE, side="buy")


def test_slippage_is_per_side_so_both_legs_of_a_round_trip_pay_it():
    """A round trip at 25 bps pays the slippage twice, not once."""
    round_trip_no_slip = costs_for(VALUE, side="buy") + costs_for(VALUE, side="sell")
    round_trip_slip = costs_for(VALUE, side="buy", slippage_bps=25) + costs_for(
        VALUE, side="sell", slippage_bps=25
    )
    assert round_trip_slip - round_trip_no_slip == pytest.approx(2 * 0.0025 * VALUE)
    assert DEFAULT_SELL_COST > DEFAULT_BUY_COST > 0
