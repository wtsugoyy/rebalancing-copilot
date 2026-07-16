"""Benchmark module + the new analyst asks: monthly stats, VaR, rolling, drift,
allocations, holdings — plus the tool wiring for all of them."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

import config
from agent import tools
from engine import analytics, benchmark as bm
from store import history


# --- benchmark store ---------------------------------------------------------
def test_active_benchmark_defaults_then_saves(seeded_ctx):
    active = bm.get_active_benchmark(seeded_ctx.conn)
    assert active["name"] == config.DEFAULT_BENCHMARK
    bm.save_benchmark(seeded_ctx.conn, "6% p.a. Absolute Return Target",
                      {"annualized": 6.0, "1y": 6.0})
    active2 = bm.get_active_benchmark(seeded_ctx.conn)
    assert active2["name"] == "6% p.a. Absolute Return Target"
    assert active2["periods"]["annualized"] == 6.0


# --- monthly machinery ---------------------------------------------------------
def _steady(n=750, monthly=0.01):
    """Synthetic daily series compounding to ~1% per month, ~21 trading days."""
    daily = (1 + monthly) ** (1 / 21) - 1
    d0 = date(2024, 1, 1).toordinal()
    dates, k = [], 0
    while len(dates) < n:
        d = date.fromordinal(d0 + k); k += 1
        if d.weekday() < 5:
            dates.append(d)
    return dates, [daily] * n


def test_monthly_returns_compound_correctly():
    dates, daily = _steady()
    m = bm.monthly_returns(dates, daily)
    # full middle months should compound to ~(1+daily)^(~21..23)-1, near 1%
    mid = [r for _, r in m[1:-1]]
    assert all(0.007 < r < 0.013 for r in mid)


def test_trailing_periods_and_comparison():
    dates, daily = _steady()
    p = bm.trailing_period_returns(dates, daily)
    assert p["1m"] is not None and p["3m"] > p["1m"] and p["1y"] > p["6m"]
    cmp_ = bm.benchmark_comparison(dates, daily,
                                   {"name": "6%", "periods": {"annualized": 6.0, "1y": 6.0}})
    ann = next(x for x in cmp_["per_period"] if x["period"] == "annualized")
    assert ann["excess"] is not None
    assert cmp_["beta"] is None and "undefined" in cmp_["beta_note"]
    assert 0 <= cmp_["win_rate_vs_benchmark"] <= 1


# --- monthly stats / VaR / rolling ---------------------------------------------
@pytest.fixture
def bal(golden):
    dates = golden["_dates"]
    timeline = [(dates[0], dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS["BAL"])))]
    return timeline, golden["fund_returns"], dates


def test_monthly_stats_shape(bal):
    out = analytics.monthly_stats(*bal, bench_monthly=0.004)
    assert out["best_month"]["return"] >= out["worst_month"]["return"]
    assert 0 <= out["positive_months_pct"] <= 1
    assert out["var_95_mom"] <= out["var_99_mom"]      # 99% VaR is a worse loss
    assert 0 <= out["win_rate_vs_benchmark"] <= 1


def test_rolling_12_and_36(bal):
    r12 = analytics.rolling_period_returns(*bal, window_months=12)
    r36 = analytics.rolling_period_returns(*bal, window_months=36)
    assert r12["points"] and r36["points"]
    assert len(r12["points"]) > len(r36["points"])


def test_drift_zero_when_funds_identical():
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    fr = {"A": [0.01, 0.01, 0.01], "B": [0.01, 0.01, 0.01]}
    out = analytics.portfolio_drift([(dates[0], {"A": 0.5, "B": 0.5})], fr, dates)
    assert abs(out["total_drift_pct"]) < 1e-9


def test_drift_moves_toward_outperformer():
    dates = [date(2024, 1, 1), date(2024, 1, 2)]
    fr = {"A": [0.0, 0.10], "B": [0.0, 0.0]}
    out = analytics.portfolio_drift([(dates[0], {"A": 0.5, "B": 0.5})], fr, dates)
    a = next(r for r in out["funds"] if r["isin"] == "A")
    assert a["drift_pp"] > 0 and out["total_drift_pct"] > 0


# --- allocations / holdings -----------------------------------------------------
def test_allocation_breakdown_sums_to_total():
    w = dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS["BAL"]))
    out = analytics.allocation_breakdown(w)
    assert abs(sum(out["asset_allocation"].values()) - 1.0) < 1e-9
    assert abs(sum(out["geographic_allocation"].values()) - 1.0) < 1e-9
    assert "approximate" in out["note"].lower() or "Approximate" in out["note"]


def test_top_holdings_lookthrough_partial_coverage():
    w = dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS["BAL"]))
    hold = tools._load_holdings()
    assert hold, "sample/top5_holdings.json must ship with the repo"
    out = analytics.top_holdings(w, hold)
    assert 0 < out["lookthrough_coverage"] < 1        # partial by design
    assert out["funds_without_data"]                   # honesty: gaps are named
    assert len(out["top_holdings"]) <= 5


# --- tool wiring (the 10 asks route through execute_tool) ------------------------
@pytest.mark.parametrize("tool,args", [
    ("target_vs_realized", {"portfolio": "BAL"}),
    ("composition_change", {"portfolio": "BAL"}),
    ("benchmark_metrics", {"portfolio": "BAL"}),
    ("monthly_stats", {"portfolio": "BAL"}),
    ("rolling_returns", {"portfolio": "BAL", "window_months": 12}),
    ("portfolio_drift", {"portfolio": "BAL"}),
    ("allocation_breakdown", {"portfolio": "BAL"}),
    ("top_holdings", {"portfolio": "BAL"}),
])
def test_new_tools_execute(seeded_ctx, tool, args):
    out = tools.execute_tool(tool, args, seeded_ctx)
    assert "error" not in out, out
    assert "summary" in out


def test_optimize_tool_advisory(seeded_ctx):
    out = tools.execute_tool("optimize_portfolio", {"sigma_target": 0.035}, seeded_ctx)
    assert "error" not in out, out
    assert out["advisory"] is True
    assert abs(sum(out["weights"].values()) - 1.0) < 1e-6
    assert out["expected_vol"] <= 0.035 + 1e-6


def test_optimize_tool_rejects_bad_bounds(seeded_ctx):
    out = tools.execute_tool("optimize_portfolio",
                             {"sigma_target": 0.035, "bounds": {"NOT_AN_ISIN": [0, 10]}},
                             seeded_ctx)
    assert "error" in out
