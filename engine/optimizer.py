"""Rebalancing engine — mean-variance optimisation with a hard volatility constraint.

Mean-variance optimiser with a hard volatility cap. The conventions below mirror a
spreadsheet Solver model this was ported from:

  Value 1 (per-fund expected return) = (prod(1+r_daily))^(252 / n_obs) - 1
  Value 2 (per-fund volatility)      = std(r_daily, ddof=1) * sqrt(252)
  Portfolio expected return          = w · Value1        (linear blend)
  Portfolio volatility               = sqrt(w' Σ w),  Σ = cov(daily, ddof=1) * 252

  maximise   w · mu
  subject to sqrt(w' Σ w) <= sigma_target      (HARD volatility cap)
             sum(w) = 1
             min_i <= w_i <= max_i             (per-fund exposure bounds)

With the vol cap binding, maximising return is maximising risk-adjusted return at the
chosen risk budget (the engine's stated basis). Solved with SLSQP from a deterministic
multi-start set; same inputs always produce the same output. The original model's
allocations are feasible points, so ours must score >= theirs (calibration-tested).

Pure functions, no UI imports. The LLM never runs this math itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.optimize import minimize

import config
from engine.validate import OptimizationInfeasibleError, PriceDataError

_EPS = 1e-9


@dataclass(frozen=True)
class OptimizerResult:
    weights: dict[str, float]            # continuous optimum
    weights_rounded: dict[str, float]    # whole-percent implementable version
    expected_return: float
    expected_vol: float
    sharpe: float
    rounded_return: float
    rounded_vol: float
    sigma_target: float
    vol_cap_binding: bool
    fund_table: list[dict] = field(default_factory=list)  # per-fund V1/V2/bounds/alloc
    n_obs: int = 0
    window: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def fund_stats(fund_returns: dict[str, list[float]], isins: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mu, vol, Sigma) using the engine's exact conventions (Value 1 / Value 2).
    Blank return cells are 0, matching the spreadsheet SUM convention."""
    R = np.array([[v if v is not None else 0.0 for v in fund_returns[i]]
                  for i in isins], dtype=float)
    n = R.shape[1]
    if n < 30:
        raise PriceDataError(f"Need >= 30 observations to optimise; have {n}.")
    growth = np.prod(1.0 + R, axis=1)
    mu = growth ** (config.OPT_ANNUALISATION / n) - 1.0          # Value 1
    vol = R.std(axis=1, ddof=1) * np.sqrt(config.OPT_ANNUALISATION)  # Value 2
    Sigma = np.cov(R, ddof=1) * config.OPT_ANNUALISATION
    return mu, vol, Sigma


def _port_vol(w: np.ndarray, Sigma: np.ndarray) -> float:
    return float(np.sqrt(max(w @ Sigma @ w, 0.0)))


def _solve(mu, Sigma, lb, ub, sigma_target, x0) -> tuple[np.ndarray, bool]:
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: sigma_target**2 - w @ Sigma @ w},
    ]
    res = minimize(lambda w: -(w @ mu), x0, method="SLSQP",
                   bounds=list(zip(lb, ub)), constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    ok = (res.success and abs(np.sum(res.x) - 1.0) < 1e-6
          and _port_vol(res.x, Sigma) <= sigma_target + 1e-6)
    return res.x, ok


def _extreme_vol(mu, Sigma, lb, ub, minimise_vol: bool) -> tuple[float, np.ndarray]:
    """Feasible min- or max-volatility point (diagnostics + a guaranteed-feasible start)."""
    n = len(mu)
    x0 = np.clip(np.full(n, 1.0 / n), lb, ub)
    x0 = x0 / x0.sum()
    sign = 1.0 if minimise_vol else -1.0
    res = minimize(lambda w: sign * (w @ Sigma @ w), x0, method="SLSQP",
                   bounds=list(zip(lb, ub)),
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})
    return _port_vol(res.x, Sigma), res.x


def _round_weights(w: np.ndarray, lb, ub) -> np.ndarray:
    """Whole-percent implementable weights (largest-remainder, bounds respected) —
    mirrors the whole-percent allocations the original model published."""
    pct = w * 100.0
    base = np.floor(pct)
    base = np.clip(base, lb * 100, ub * 100)
    short = int(round(100 - base.sum()))
    order = np.argsort(-(pct - np.floor(pct)))
    for k in range(abs(short)):
        i = order[k % len(order)]
        base[i] += np.sign(short)
    return np.clip(base, lb * 100, ub * 100) / 100.0


def optimize(
    fund_returns: dict[str, list[float]],
    dates: list[date],
    sigma_target: float,
    bounds_pct: dict[str, tuple[float, float]] | None = None,
    isins: list[str] | None = None,
    rf: float = 0.0,
) -> OptimizerResult:
    """Run the rebalancing engine. bounds_pct in percent (e.g. (0, 45))."""
    isins = isins or list(fund_returns)
    if not 0 < sigma_target < 1:
        raise PriceDataError(f"sigma_target must be a decimal like 0.035; got {sigma_target}")
    bp = {**{i: config.DEFAULT_FUND_BOUNDS.get(i, (0.0, 100.0)) for i in isins},
          **(bounds_pct or {})}
    lb = np.array([bp[i][0] / 100.0 for i in isins])
    ub = np.array([bp[i][1] / 100.0 for i in isins])
    if (lb > ub).any():
        bad = [i for k, i in enumerate(isins) if lb[k] > ub[k]]
        raise OptimizationInfeasibleError(f"min > max for {bad}", offenders=bad)
    if lb.sum() > 1.0 + _EPS or ub.sum() < 1.0 - _EPS:
        raise OptimizationInfeasibleError(
            f"Bounds cannot sum to 100%: sum(min)={lb.sum():.2%}, sum(max)={ub.sum():.2%}.")

    mu, vol, Sigma = fund_stats(fund_returns, isins)

    vol_min, w_minvol = _extreme_vol(mu, Sigma, lb, ub, minimise_vol=True)
    if vol_min > sigma_target + 1e-6:
        vol_max, _ = _extreme_vol(mu, Sigma, lb, ub, minimise_vol=False)
        raise OptimizationInfeasibleError(
            f"Volatility target {sigma_target:.2%} is unreachable: the lowest achievable "
            f"volatility under these bounds is {vol_min:.2%} "
            f"(achievable range {vol_min:.2%}–{vol_max:.2%}).",
            achievable_min=vol_min, achievable_max=vol_max)

    # deterministic multi-start: the min-vol point (always feasible), equal-weight,
    # inverse-vol tilt, and a return-greedy tilt
    n = len(isins)
    starts = [w_minvol]
    eq = np.clip(np.full(n, 1.0 / n), lb, ub); starts.append(eq / eq.sum())
    inv = np.clip(1.0 / (vol + 1e-6), None, None); inv = np.clip(inv / inv.sum(), lb, ub)
    starts.append(inv / inv.sum())
    greedy = np.clip((mu - mu.min()) + 1e-6, None, None); greedy = np.clip(greedy / greedy.sum(), lb, ub)
    starts.append(greedy / greedy.sum())

    best, best_ret = None, -np.inf
    for x0 in starts:
        w, ok = _solve(mu, Sigma, lb, ub, sigma_target, x0)
        if ok and w @ mu > best_ret:
            best, best_ret = w, float(w @ mu)
    if best is None:
        raise OptimizationInfeasibleError(
            "Solver could not find a feasible allocation from any starting point.")

    w = np.clip(best, lb, ub); w = w / w.sum()
    pv = _port_vol(w, Sigma)
    wr = _round_weights(w, lb, ub)
    rv, rr = _port_vol(wr, Sigma), float(wr @ mu)

    table = [{
        "isin": i, "fund": config.FUND_NAMES.get(i, i),
        "min_pct": bp[i][0], "max_pct": bp[i][1],
        "value1_expected_return": float(mu[k]), "value2_volatility": float(vol[k]),
        "weight": float(w[k]), "weight_rounded": float(wr[k]),
    } for k, i in enumerate(isins)]

    return OptimizerResult(
        weights={i: float(w[k]) for k, i in enumerate(isins)},
        weights_rounded={i: float(wr[k]) for k, i in enumerate(isins)},
        expected_return=float(w @ mu), expected_vol=pv,
        sharpe=(float(w @ mu) - rf) / pv if pv > 0 else float("nan"),
        rounded_return=rr, rounded_vol=rv,
        sigma_target=sigma_target,
        vol_cap_binding=bool(pv >= sigma_target - 5e-4),
        fund_table=table, n_obs=len(dates),
        window=f"{dates[0].isoformat()}..{dates[-1].isoformat()}" if dates else "",
    )
