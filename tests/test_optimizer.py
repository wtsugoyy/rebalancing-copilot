"""Rebalancing-engine regression gate.

`solutions_golden.json` holds synthetic daily returns and three solver blocks (vol
targets 3.5/4.5/5.5%) recorded from this engine — see
`scripts/make_synthetic_fixtures.py`. These tests pin the optimizer to its own
recorded output: they catch drift from a refactor, they do not independently prove
the solver is right. Structural properties below (bounds honoured, hard vol cap,
determinism, monotonicity in the target) are real assertions and do not depend on
the recorded numbers.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

import config
from engine import optimizer
from engine.validate import OptimizationInfeasibleError

FIX = Path(__file__).parent / "fixtures" / "solutions_golden.json"


@pytest.fixture(scope="module")
def sol():
    g = json.loads(FIX.read_text(encoding="utf-8"))
    g["_dates"] = [date.fromisoformat(d) for d in g["dates"]]
    return g


def test_value1_value2_match_baseline(sol):
    """Value 1 = trading-day CAGR, Value 2 = std*sqrt252 — all 12 funds to 4dp."""
    mu, vol, _ = optimizer.fund_stats(sol["fund_returns"], sol["isins"])
    for k, f in enumerate(sol["blocks"]["block_3.5"]["funds"]):
        assert abs(round(float(mu[k]), 4) - f["value1"]) < 5e-5, f["isin"]
        assert abs(round(float(vol[k]), 4) - f["value2"]) < 5e-5, f["isin"]


@pytest.mark.parametrize("block", ["block_3.5", "block_4.5", "block_5.5"])
def test_reproduces_recorded_blocks(sol, block):
    """Continuous optimum still hits the recorded headline ER (4dp) at the vol cap,
    and whole-percent rounding still lands on the recorded allocation."""
    blk = sol["blocks"][block]
    bounds = {f["isin"]: (f["min"], f["max"]) for f in blk["funds"]}
    r = optimizer.optimize(sol["fund_returns"], sol["_dates"], blk["vol_target"],
                           bounds, sol["isins"])
    assert abs(r.expected_return - blk["expected_return"]) < 5e-4     # headline ER
    assert r.expected_vol <= blk["vol_target"] + 1e-6                 # hard cap held
    theirs = {f["isin"]: (f["allocation"] or 0.0) for f in blk["funds"]}
    for isin in sol["isins"]:
        assert abs(r.weights_rounded[isin] - theirs[isin]) < 5e-3, (block, isin)


def test_deterministic(sol):
    blk = sol["blocks"]["block_3.5"]
    bounds = {f["isin"]: (f["min"], f["max"]) for f in blk["funds"]}
    a = optimizer.optimize(sol["fund_returns"], sol["_dates"], 0.035, bounds, sol["isins"])
    b = optimizer.optimize(sol["fund_returns"], sol["_dates"], 0.035, bounds, sol["isins"])
    assert a.weights == b.weights


def test_bounds_respected(sol):
    blk = sol["blocks"]["block_3.5"]
    bounds = {f["isin"]: (f["min"], f["max"]) for f in blk["funds"]}
    r = optimizer.optimize(sol["fund_returns"], sol["_dates"], 0.035, bounds, sol["isins"])
    for f in blk["funds"]:
        w = r.weights[f["isin"]]
        assert f["min"] / 100 - 1e-9 <= w <= f["max"] / 100 + 1e-9
    assert abs(sum(r.weights.values()) - 1.0) < 1e-6
    assert abs(sum(r.weights_rounded.values()) - 1.0) < 1e-6


def test_unreachably_low_target_is_infeasible(sol):
    """Even 100% in the calmest cash fund has ~0.15% vol; ask for far below any
    feasible mix by forcing equity exposure."""
    bounds = {i: (0.0, 100.0) for i in sol["isins"]}
    bounds["ZZ0000000012"] = (50.0, 100.0)   # force 50% into the gold fund (~26% vol)
    with pytest.raises(OptimizationInfeasibleError) as exc:
        optimizer.optimize(sol["fund_returns"], sol["_dates"], 0.01, bounds, sol["isins"])
    assert "achievable" in str(exc.value).lower()


def test_impossible_bounds_rejected(sol):
    bounds = {i: (0.0, 5.0) for i in sol["isins"]}   # max sums to 60% < 100%
    with pytest.raises(OptimizationInfeasibleError):
        optimizer.optimize(sol["fund_returns"], sol["_dates"], 0.05, bounds, sol["isins"])


def test_higher_target_never_lowers_return(sol):
    blk = sol["blocks"]["block_3.5"]
    bounds = {f["isin"]: (f["min"], f["max"]) for f in blk["funds"]}
    rets = [optimizer.optimize(sol["fund_returns"], sol["_dates"], t, bounds,
                               sol["isins"]).expected_return
            for t in (0.02, 0.035, 0.055)]
    assert rets[0] <= rets[1] + 1e-9 <= rets[2] + 2e-9
