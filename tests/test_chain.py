"""TC-2: chaining must reduce exactly to the static case for a single snapshot, and
must switch weights at the correct boundary for multiple segments."""
from datetime import date

import numpy as np

from engine import perf

FR = {
    "ZZ0000000001": [0.01, -0.02, 0.03, 0.00, 0.05],
    "ZZ0000000002": [0.00, 0.01, -0.01, 0.02, -0.03],
}
DATES = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]


def test_single_snapshot_equals_static():
    w = {"ZZ0000000001": 0.6, "ZZ0000000002": 0.4}
    chained = perf.portfolio_returns([(DATES[0], w)], FR, DATES)
    static = [0.6 * FR["ZZ0000000001"][t] + 0.4 * FR["ZZ0000000002"][t] for t in range(5)]
    assert np.allclose(chained, static, atol=1e-15)


def test_multi_segment_switches_at_boundary():
    w1 = {"ZZ0000000001": 1.0, "ZZ0000000002": 0.0}
    w2 = {"ZZ0000000001": 0.0, "ZZ0000000002": 1.0}
    # rebalance takes effect on 2024-01-04 (index 2)
    chained = perf.portfolio_returns([(DATES[0], w1), (DATES[2], w2)], FR, DATES)
    expected = [
        FR["ZZ0000000001"][0], FR["ZZ0000000001"][1],   # w1 active
        FR["ZZ0000000002"][2], FR["ZZ0000000002"][3], FR["ZZ0000000002"][4],  # w2 active
    ]
    assert np.allclose(chained, expected, atol=1e-15)


def test_dates_before_first_segment_are_excluded():
    w = {"ZZ0000000001": 1.0}
    # segment effective only from index 2 onward
    out = perf.portfolio_returns([(DATES[2], w)], FR, DATES)
    assert len(out) == 3  # only indices 2,3,4 attributed
