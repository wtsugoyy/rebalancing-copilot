"""Performance attribution — the exact port of the Excel model.

This is the correctness spine of the whole tool. Every formula here is transcribed
verbatim from `fund_returns_workbook.xlsx` (Sheet3) and gated by the golden
test at <=1e-9. The 365-calendar-day return annualisation vs. the hardcoded sqrt(252)
volatility annualisation is a deliberate reproduction of the workbook, NOT a bug.

Pure functions, no UI imports.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date

import numpy as np

import config
from engine.validate import InsufficientHistoryError


@dataclass(frozen=True)
class Metrics:
    total_return: float
    ann_return: float
    ann_vol: float
    max_drawdown: float
    sharpe: float
    sortino: float
    avg_rf: float
    n_obs: int
    period_start: str
    period_end: str

    def to_dict(self) -> dict:
        return asdict(self)


WeightMap = dict[str, float]
Segment = tuple[date, WeightMap]  # (effective_date, {isin: weight})


def portfolio_returns(
    weight_timeline: list[Segment],
    fund_returns: dict[str, list[float]],
    dates: list[date],
) -> list[float]:
    """Chained, piecewise-constant-weight portfolio daily returns.

    For each date t, the active weight vector is the latest segment whose
    effective_date <= t. r_port,t = sum_i w_i * r_i,t. Dates earlier than the first
    segment's effective_date are excluded (no active allocation yet).

    A single segment with effective_date <= dates[0] reduces exactly to the Excel
    static-weight computation (regression-tested, TC-2).
    """
    if not weight_timeline:
        raise InsufficientHistoryError("No weight snapshots supplied for attribution.")
    segments = sorted(weight_timeline, key=lambda s: s[0])

    out: list[float] = []
    for t, d in enumerate(dates):
        active = None
        for eff_date, wmap in segments:
            if eff_date <= d:
                active = wmap
            else:
                break
        if active is None:
            continue  # date precedes first rebalance; not attributed
        r = 0.0
        for isin, w in active.items():
            if w:
                val = fund_returns[isin][t]
                if val:  # Excel SUM treats a blank fund-return cell as 0
                    r += w * val
        out.append(r)
    return out


def excess_returns(port_returns: list[float], rf_period: list[float]) -> list[float]:
    """excess_t = r_port,t - rf_period,t, with the first observation forced to 0
    (Excel hardcodes Returns!L2 = 0)."""
    if not port_returns:
        return []
    out = [0.0]
    for i in range(1, len(port_returns)):
        out.append(port_returns[i] - rf_period[i])
    return out


def _sample_std(x: np.ndarray) -> float:
    """Excel STDEV: sample standard deviation (ddof=1)."""
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=config.STDEV_DDOF))


def _max_drawdown(returns: np.ndarray) -> float:
    nav = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(nav)
    return float(np.min(nav / peak) - 1.0)


def compute_metrics(
    port_returns: list[float],
    excess: list[float],
    dates: list[date],
    avg_rf: float,
) -> Metrics:
    """The pure Sheet3 metric set (benchmark items excluded, plan R6)."""
    if len(port_returns) < config.MIN_OBS:
        raise InsufficientHistoryError(
            f"Only {len(port_returns)} aligned observations; need >= {config.MIN_OBS}.",
            n_obs=len(port_returns),
            min_obs=config.MIN_OBS,
        )
    r = np.asarray(port_returns, dtype=float)
    e = np.asarray(excess, dtype=float)
    sqrt_t = np.sqrt(config.VOL_ANNUALISATION)

    growth = float(np.prod(1.0 + r))
    total_return = growth - 1.0

    elapsed_days = (dates[-1] - dates[0]).days
    ann_return = growth ** (365.0 / elapsed_days) - 1.0

    ann_vol = _sample_std(r) * sqrt_t
    max_dd = _max_drawdown(r)

    sharpe = (ann_return - avg_rf) / (_sample_std(e) * sqrt_t)

    downside = e[e < 0]
    denom_sortino = _sample_std(downside) * sqrt_t
    if not np.isfinite(denom_sortino) or denom_sortino == 0:
        denom_sortino = _sample_std(r) * sqrt_t  # Excel IFERROR fallback
    sortino = (ann_return - avg_rf) / denom_sortino

    return Metrics(
        total_return=total_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        avg_rf=avg_rf,
        n_obs=len(port_returns),
        period_start=dates[0].isoformat(),
        period_end=dates[-1].isoformat(),
    )
