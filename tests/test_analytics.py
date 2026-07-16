"""Analyst toolkit correctness — the new tools must reconcile with the Excel-validated
attribution engine, not drift from it."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

import config
from agent import tools
from engine import analytics


@pytest.fixture
def series(golden):
    dates = golden["_dates"]
    fr = golden["fund_returns"]
    timeline = [(dates[0], dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS["BAL"])))]
    return timeline, fr, dates, golden["_yields"]


def test_yearly_returns_compound_back_to_total(series):
    """The headline question: per-year returns must multiply back to the total return."""
    timeline, fr, dates, _ = series
    out = analytics.periodic_returns(timeline, fr, dates, "year")
    assert [p["period"] for p in out["periods"]] == ["2023", "2024", "2025", "2026"]

    compounded = np.prod([1 + p["return"] for p in out["periods"]]) - 1
    total = analytics.periodic_returns(timeline, fr, dates, "year")  # same source
    from engine import perf
    whole = np.prod(1 + np.array(perf.portfolio_returns(timeline, fr, dates))) - 1
    assert abs(compounded - whole) < 1e-9   # sub-periods reconcile to the whole


def test_quarterly_and_monthly_buckets(series):
    timeline, fr, dates, _ = series
    q = analytics.periodic_returns(timeline, fr, dates, "quarter")
    m = analytics.periodic_returns(timeline, fr, dates, "month")
    assert all("-Q" in p["period"] for p in q["periods"])
    assert len(m["periods"]) > len(q["periods"]) > 3


def test_volatility_trend_direction_is_evidence_backed(series):
    timeline, fr, dates, _ = series
    out = analytics.volatility_trend(timeline, fr, dates, window=63)
    assert out["direction"] in ("increasing", "decreasing", "broadly flat")
    # direction must agree with the sign of the recent-vs-prior change
    if out["direction"] == "increasing":
        assert out["change_pp"] > 0
    elif out["direction"] == "decreasing":
        assert out["change_pp"] < 0
    assert out["latest_ann_volatility"] > 0
    assert "summary" in out


def _synth(calm_n: int, wild_n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    r = np.concatenate([rng.normal(0, 0.002, calm_n), rng.normal(0, 0.02, wild_n)])
    dates = [date.fromordinal(date(2024, 1, 1).toordinal() + i) for i in range(len(r))]
    return [(dates[0], {"F": 1.0})], {"F": list(r)}, dates


def test_volatility_trend_detects_a_recent_increase():
    """Calm for a long stretch, turbulent recently -> 'increasing' RIGHT NOW."""
    timeline, fr, dates = _synth(calm_n=250, wild_n=150)
    out = analytics.volatility_trend(timeline, fr, dates, window=63)
    assert out["direction"] == "increasing"
    assert out["change_pp"] > 0


def test_volatility_trend_detects_a_recent_decrease():
    """Turbulent then calm recently -> 'decreasing' RIGHT NOW."""
    rng = np.random.default_rng(11)
    r = np.concatenate([rng.normal(0, 0.02, 250), rng.normal(0, 0.002, 150)])
    dates = [date.fromordinal(date(2024, 1, 1).toordinal() + i) for i in range(len(r))]
    out = analytics.volatility_trend([(dates[0], {"F": 1.0})], {"F": list(r)}, dates, window=63)
    assert out["direction"] == "decreasing"
    assert out["change_pp"] < 0


def test_whole_period_trend_is_reported_separately_from_right_now():
    """calm->wild over the full series: the OVERALL trend is up even if the last two
    windows are both in the turbulent regime (they are, so 'right now' reads flat)."""
    timeline, fr, dates = _synth(calm_n=200, wild_n=200)
    out = analytics.volatility_trend(timeline, fr, dates, window=63)
    assert out["overall_direction"] == "increasing"
    assert out["slope_pp_per_year"] > 0
    assert out["direction"] in ("increasing", "decreasing", "broadly flat")  # separate question


def test_attribution_range_matches_full_period_when_unbounded(series, golden):
    """A range covering everything must equal the whole-period attribution exactly."""
    timeline, fr, dates, yields = series
    ranged = analytics.attribution_range(timeline, fr, dates, yields, None, None)
    expected = golden["portfolios"]["BAL"]["expected"]
    assert abs(ranged["total_return"] - expected["total_return"]) < 1e-9
    assert abs(ranged["ann_vol"] - expected["ann_vol"]) < 1e-9


def test_fund_contributions_sum_to_arithmetic_total(series):
    timeline, fr, dates, _ = series
    out = analytics.fund_contribution(timeline, fr, dates)
    from engine import perf
    total_arith = sum(perf.portfolio_returns(timeline, fr, dates))
    assert abs(sum(f["return_contribution"] for f in out["funds"]) - total_arith) < 1e-9
    # only the funds actually held show up
    held = {i for i, w in zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS["BAL"]) if w}
    assert {f["isin"] for f in out["funds"]} == held


def test_drawdown_deepest_matches_max_drawdown_metric(series, golden):
    timeline, fr, dates, _ = series
    out = analytics.drawdown_periods(timeline, fr, dates, top=3)
    deepest = min(d["depth"] for d in out["drawdowns"])
    assert abs(deepest - golden["portfolios"]["BAL"]["expected"]["max_drawdown"]) < 1e-9
    d0 = out["drawdowns"][0]
    assert d0["start"] <= d0["trough"]


def test_compare_portfolios_ranks_by_sharpe(seeded_ctx):
    out = tools.execute_tool("compare_portfolios",
                             {"portfolios": ["BAL", "ADV", "ADV+"]}, seeded_ctx)
    sharpes = [p["sharpe"] for p in out["portfolios"]]
    assert sharpes == sorted(sharpes, reverse=True)
    assert out["best_sharpe"] == out["portfolios"][0]["portfolio"]


# --- the two questions the copilot previously could not answer ---------------
def test_tool_answers_yearly_return_question(seeded_ctx):
    out = tools.execute_tool("periodic_returns", {"portfolio": "BAL", "freq": "year"}, seeded_ctx)
    assert "error" not in out
    assert len(out["periods"]) >= 3
    assert "2024" in out["summary"]


def test_tool_answers_volatility_trend_question(seeded_ctx):
    out = tools.execute_tool("volatility_trend", {"portfolio": "BAL"}, seeded_ctx)
    assert "error" not in out
    assert out["direction"] in ("increasing", "decreasing", "broadly flat")


def test_bad_freq_is_rejected(seeded_ctx):
    out = tools.execute_tool("periodic_returns", {"portfolio": "BAL", "freq": "fortnight"},
                             seeded_ctx)
    assert "error" in out
