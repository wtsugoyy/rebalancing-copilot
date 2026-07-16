"""Attribution regression gate: the engine must reproduce its recorded baseline to
<=1e-9 across every portfolio.

The baseline in `golden.json` was produced by running this engine over synthetic
inputs (`scripts/make_synthetic_fixtures.py`), so it pins behaviour rather than
proving correctness — any drift in a metric definition fails here loudly, which is
the point. The tolerance is 1e-9 because these are pure deterministic computations:
anything looser would let real numerical drift through unnoticed."""
import numpy as np
import pytest

from engine import perf, rf

TOL = 1e-9
METRIC_KEYS = ["total_return", "ann_return", "ann_vol", "max_drawdown", "sharpe", "sortino"]


def _capped_yields(golden):
    """Yields inside the lookup window. The fixture's cutoff deliberately stops short
    of the last NAV date so the RF_FALLBACK path for later dates stays under test.
    See scripts/make_synthetic_fixtures.py."""
    from datetime import date
    cutoff = date.fromisoformat(golden["rf_lookup_cutoff"])
    return [(d, y) for d, y in golden["_yields"] if d <= cutoff]


def _series(golden, code):
    dates = golden["_dates"]
    fund_returns = golden["fund_returns"]
    weights = golden["portfolios"][code]["weights"]
    seg = [(dates[0], {k: (v or 0.0) for k, v in weights.items()})]
    port = perf.portfolio_returns(seg, fund_returns, dates)
    rf_period = rf.rf_period_series(dates, _capped_yields(golden))
    excess = perf.excess_returns(port, rf_period)
    return dates, port, rf_period, excess


@pytest.mark.parametrize("code", ["SC", "SC+", "BAL", "BAL+", "ADV", "ADV+"])
def test_portfolio_returns_match_baseline(golden, code):
    _, port, _, _ = _series(golden, code)
    expected = golden["portfolios"][code]["portfolio_return_expected"]
    assert np.allclose(port, expected, atol=TOL, rtol=0), \
        f"{code}: portfolio return series diverges from the baseline"


def test_rf_period_matches_baseline(golden):
    rf_period = rf.rf_period_series(golden["_dates"], _capped_yields(golden))
    assert np.allclose(rf_period, golden["rf_period_expected"], atol=TOL, rtol=0), \
        "rf_period series diverges from the baseline"


@pytest.mark.parametrize("code", ["SC", "SC+", "BAL", "BAL+", "ADV", "ADV+"])
def test_excess_matches_baseline(golden, code):
    _, _, _, excess = _series(golden, code)
    expected = golden["portfolios"][code]["excess_expected"]
    assert np.allclose(excess, expected, atol=TOL, rtol=0), \
        f"{code}: excess series diverges from the baseline"


@pytest.mark.parametrize("code", ["SC", "SC+", "BAL", "BAL+", "ADV", "ADV+"])
def test_metrics_match_baseline(golden, code):
    dates, port, _, excess = _series(golden, code)
    avg_rf = rf.average_rf(dates[0], dates[-1], golden["_yields"])
    m = perf.compute_metrics(port, excess, dates, avg_rf).to_dict()
    expected = golden["portfolios"][code]["expected"]

    # avg_rf is portfolio-independent
    assert abs(m["avg_rf"] - golden["avg_rf_expected"]) <= TOL, f"{code}: avg_rf"

    for key in METRIC_KEYS:
        exp = expected[key]
        got = m[key]
        assert abs(got - exp) <= TOL * (1 + abs(exp)), \
            f"{code}.{key}: got {got!r} expected {exp!r} (diff {abs(got-exp):.2e})"
