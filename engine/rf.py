"""Risk-free: per-period rf series (Returns!K) and average rf (Sheet3 row 20).

Exact port of the Excel logic:

  rf_period[0] = 0                                            (Excel hardcodes row-2 = 0)
  rf_period[i] = XLOOKUP(date_i, yields, mode=1)/100 * (date_i - date_{i-1}).days / 365
               fallback RF_FALLBACK/100-equivalent on lookup miss (still * daycount/365)

  avg_rf = mean(yield_pct for yield dates within [start, end]) / 100   (AVERAGEIFS/100)

`XLOOKUP(..., match_mode=1)` = exact match or the next LARGER key.
"""
from __future__ import annotations

from datetime import date

import config


def _lookup_yield_pct(target: date, yields: list[tuple[date, float]]) -> float | None:
    """XLOOKUP match_mode=1: exact date, else the smallest date strictly greater.

    `yields` must be sorted ascending by date. Returns yield in percent, or None if
    the target is beyond the last yield date (=> caller applies fallback).
    """
    best = None
    for d, y in yields:
        if d == target:
            return y
        if d > target:
            if best is None or d < best[0]:
                best = (d, y)
    return best[1] if best is not None else None


def rf_period_series(dates: list[date], yields: list[tuple[date, float]]) -> list[float]:
    """Per-observation risk-free accrual aligned to `dates` (same length)."""
    ys = sorted(yields, key=lambda t: t[0])
    out = [0.0]  # first observation hardcoded to 0 (Excel row-2)
    for i in range(1, len(dates)):
        daycount = (dates[i] - dates[i - 1]).days
        y = _lookup_yield_pct(dates[i], ys)
        rate = (y / 100.0) if y is not None else config.RF_FALLBACK
        out.append(rate * daycount / 365.0)
    return out


def average_rf(period_start: date, period_end: date,
               yields: list[tuple[date, float]]) -> float:
    """Mean of yield% within [start, end], divided by 100 (Sheet3 avg_rf)."""
    vals = [y for d, y in yields if period_start <= d <= period_end]
    if not vals:
        return config.RF_FALLBACK
    return (sum(vals) / len(vals)) / 100.0
