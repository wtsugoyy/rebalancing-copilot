"""Analyst toolkit — deterministic time-series analysis over the portfolio return series.

Everything here is pure, deterministic, and reuses `engine.perf` so it reconciles with
the attribution model. These are the functions behind the tools that let
the copilot answer real analyst questions:

    "yearly return for each of the 3 years"      -> periodic_returns(freq="year")
    "is volatility increasing or decreasing?"    -> volatility_trend()
    "compare 2024 vs 2025"                       -> attribution_range()
    "which fund drove the return?"               -> fund_contribution()
    "worst drawdown and when?"                   -> drawdown_periods()

No UI imports. The LLM never computes any of this — it only selects it.
"""
from __future__ import annotations

from datetime import date

import numpy as np

import config
from engine import perf, rf as rf_mod
from engine.validate import InsufficientHistoryError, PriceDataError

Segment = perf.Segment


# --------------------------------------------------------------------- helpers
def _series(weight_timeline: list[Segment], fund_returns: dict[str, list[float]],
            dates: list[date]) -> list[float]:
    return perf.portfolio_returns(weight_timeline, fund_returns, dates)


def _slice(dates: list[date], values: list[float],
           start: date | None, end: date | None) -> tuple[list[date], list[float]]:
    d, v = [], []
    for dt, val in zip(dates, values):
        if start and dt < start:
            continue
        if end and dt > end:
            continue
        d.append(dt); v.append(val)
    return d, v


def _compound(returns: list[float]) -> float:
    return float(np.prod(1.0 + np.asarray(returns, dtype=float)) - 1.0)


def _ann_vol(returns: list[float]) -> float:
    a = np.asarray(returns, dtype=float)
    if a.size < 2:
        return float("nan")
    return float(np.std(a, ddof=config.STDEV_DDOF) * np.sqrt(config.VOL_ANNUALISATION))


# ------------------------------------------------------------------- periodic
def periodic_returns(weight_timeline, fund_returns, dates, freq: str = "year") -> dict:
    """Return (and volatility) for each calendar year / quarter / month.

    Answers: "what is the yearly return for each of the 3 years?"
    Each bucket compounds its own daily returns — sub-period returns multiply back to
    the total-period return (guaranteed by construction, asserted in tests).
    """
    freq = freq.lower()
    if freq not in ("year", "quarter", "month"):
        raise PriceDataError(f"freq must be year|quarter|month, got {freq!r}")

    r = _series(weight_timeline, fund_returns, dates)

    def key(d: date) -> str:
        if freq == "year":
            return str(d.year)
        if freq == "quarter":
            return f"{d.year}-Q{(d.month - 1) // 3 + 1}"
        return f"{d.year}-{d.month:02d}"

    buckets: dict[str, list[float]] = {}
    bdates: dict[str, list[date]] = {}
    for d, v in zip(dates, r):
        buckets.setdefault(key(d), []).append(v)
        bdates.setdefault(key(d), []).append(d)

    periods = []
    for k in sorted(buckets):
        vals = buckets[k]
        periods.append({
            "period": k,
            "return": _compound(vals),
            "ann_volatility": _ann_vol(vals),
            "n_obs": len(vals),
            "start": bdates[k][0].isoformat(),
            "end": bdates[k][-1].isoformat(),
        })

    return {
        "freq": freq,
        "periods": periods,
        "summary": "; ".join(f"{p['period']}: {p['return'] * 100:.2f}%" for p in periods),
    }


# -------------------------------------------------------------------- rolling
def rolling_volatility(weight_timeline, fund_returns, dates, window: int = 63) -> dict:
    """Annualised volatility over a rolling window (63 trading days ~= 1 quarter)."""
    r = np.asarray(_series(weight_timeline, fund_returns, dates), dtype=float)
    if r.size < window + 1:
        raise InsufficientHistoryError(
            f"Need > {window} observations for a {window}-day rolling window; have {r.size}.")
    sqrt_t = np.sqrt(config.VOL_ANNUALISATION)
    out = []
    for i in range(window, r.size + 1):
        w = r[i - window:i]
        out.append({"date": dates[i - 1].isoformat(),
                    "ann_volatility": float(np.std(w, ddof=config.STDEV_DDOF) * sqrt_t)})
    return {"window": window, "points": out}


def volatility_trend(weight_timeline, fund_returns, dates, window: int = 63) -> dict:
    """Is volatility rising or falling? Compares the latest rolling-vol reading with the
    prior window, and fits a least-squares slope over the rolling series.

    Answers: "what is the volatility trend right now — increasing or decreasing?"
    """
    roll = rolling_volatility(weight_timeline, fund_returns, dates, window)["points"]
    vals = np.array([p["ann_volatility"] for p in roll], dtype=float)

    latest = float(vals[-1])
    # compare the most recent window's average with the one before it
    recent = float(np.mean(vals[-window:])) if vals.size >= window else float(np.mean(vals))
    prior = (float(np.mean(vals[-2 * window:-window]))
             if vals.size >= 2 * window else float(np.mean(vals[:max(1, vals.size // 2)])))
    change_pp = (recent - prior) * 100.0

    # least-squares slope over the whole rolling series (vol-points per year)
    x = np.arange(vals.size, dtype=float)
    slope_per_obs = float(np.polyfit(x, vals, 1)[0]) if vals.size > 1 else 0.0
    slope_pp_per_year = slope_per_obs * config.VOL_ANNUALISATION * 100.0

    def _label(x: float, tol: float) -> str:
        if x > tol:
            return "increasing"
        if x < -tol:
            return "decreasing"
        return "broadly flat"

    # Two DIFFERENT questions, answered separately (conflating them misleads):
    #   direction         -> "is vol rising right now?"  (recent window vs the one before)
    #   overall_direction -> "has vol risen across the whole period?" (least-squares slope)
    direction = _label(change_pp, 0.25)
    overall_direction = _label(slope_pp_per_year, 0.25)

    return {
        "window": window,
        "direction": direction,                    # recent / "right now"
        "overall_direction": overall_direction,    # whole-period trend
        "latest_ann_volatility": latest,
        "recent_avg": recent,
        "prior_avg": prior,
        "change_pp": change_pp,
        "slope_pp_per_year": slope_pp_per_year,
        "first_date": roll[0]["date"],
        "last_date": roll[-1]["date"],
        "summary": (
            f"Rolling {window}-day annualised volatility is {direction} right now "
            f"(latest {latest * 100:.2f}%; recent-window average {recent * 100:.2f}% vs "
            f"prior {prior * 100:.2f}%, {change_pp:+.2f} pp). Over the whole period the "
            f"trend is {overall_direction} ({slope_pp_per_year:+.2f} pp/year). Measured "
            f"{roll[0]['date']}..{roll[-1]['date']}."),
    }


def rolling_return(weight_timeline, fund_returns, dates, window: int = 63) -> dict:
    """Compounded return over a rolling window (momentum view)."""
    r = _series(weight_timeline, fund_returns, dates)
    if len(r) < window + 1:
        raise InsufficientHistoryError(f"Need > {window} observations; have {len(r)}.")
    pts = [{"date": dates[i - 1].isoformat(), "return": _compound(r[i - window:i])}
           for i in range(window, len(r) + 1)]
    return {"window": window, "points": pts}


# --------------------------------------------------------------- range metrics
def attribution_range(weight_timeline, fund_returns, dates, yields,
                      start: date | None = None, end: date | None = None) -> dict:
    """Full attribution metrics over an arbitrary date window (e.g. one year).

    Answers: "compare 2024 vs 2025", "how did it do in H1?"
    """
    r_all = _series(weight_timeline, fund_returns, dates)
    d, r = _slice(dates, r_all, start, end)
    if len(r) < config.MIN_OBS:
        raise InsufficientHistoryError(
            f"Only {len(r)} observations in {start}..{end}; need >= {config.MIN_OBS}.")
    rf_period = rf_mod.rf_period_series(d, yields)
    excess = perf.excess_returns(r, rf_period)
    avg_rf = rf_mod.average_rf(d[0], d[-1], yields)
    m = perf.compute_metrics(r, excess, d, avg_rf).to_dict()
    m["summary"] = (
        f"{d[0].isoformat()}..{d[-1].isoformat()} ({m['n_obs']} obs): total return "
        f"{m['total_return'] * 100:.2f}%, annualised return {m['ann_return'] * 100:.2f}%, "
        f"annualised volatility {m['ann_vol'] * 100:.2f}%, max drawdown "
        f"{m['max_drawdown'] * 100:.2f}%, Sharpe {m['sharpe']:.4f}, "
        f"Sortino {m['sortino']:.4f}.")
    return m


# ------------------------------------------------------------- decomposition
def fund_contribution(weight_timeline, fund_returns, dates) -> dict:
    """Per-fund contribution to the portfolio's total return and to its variance.

    Answers: "which fund drove the returns / the risk?"
    Return contribution uses the additive daily decomposition r_p = sum_i w_i * r_i,
    summed over time (so contributions add up to the arithmetic total).
    """
    segments = sorted(weight_timeline, key=lambda s: s[0])
    isins = sorted({i for _, w in segments for i in w})

    contrib_ts: dict[str, list[float]] = {i: [] for i in isins}
    port: list[float] = []
    for t, d in enumerate(dates):
        active = None
        for eff, w in segments:
            if eff <= d:
                active = w
            else:
                break
        if active is None:
            continue
        tot = 0.0
        for i in isins:
            w = active.get(i, 0.0)
            c = w * (fund_returns[i][t] or 0.0) if w else 0.0
            contrib_ts[i].append(c)
            tot += c
        port.append(tot)

    p = np.asarray(port, dtype=float)
    port_var = float(np.var(p, ddof=config.STDEV_DDOF)) if p.size > 1 else 0.0

    rows = []
    for i in isins:
        c = np.asarray(contrib_ts[i], dtype=float)
        if not np.any(c):
            continue
        # risk share via covariance of the fund's contribution with the portfolio
        cov = float(np.cov(c, p, ddof=config.STDEV_DDOF)[0, 1]) if p.size > 1 else 0.0
        rows.append({
            "isin": i,
            "return_contribution": float(np.sum(c)),          # additive, sums to total
            "risk_contribution_share": (cov / port_var) if port_var else 0.0,
        })
    rows.sort(key=lambda r: r["return_contribution"], reverse=True)
    return {
        "funds": rows,
        "summary": "; ".join(
            f"{r['isin']}: return contribution {r['return_contribution'] * 100:.2f}pp, "
            f"risk share {r['risk_contribution_share'] * 100:.1f}%" for r in rows),
    }


def drawdown_periods(weight_timeline, fund_returns, dates, top: int = 3) -> dict:
    """The deepest peak-to-trough drawdowns, with their dates and recovery status.

    Answers: "what was the worst drawdown and when?"
    """
    r = np.asarray(_series(weight_timeline, fund_returns, dates), dtype=float)
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0

    episodes, in_dd, start_i, trough_i = [], False, 0, 0
    for i in range(len(dd)):
        if not in_dd and dd[i] < 0:
            in_dd, start_i, trough_i = True, i, i
        elif in_dd:
            if dd[i] < dd[trough_i]:
                trough_i = i
            if dd[i] >= -1e-12:  # recovered to a new peak
                episodes.append((start_i, trough_i, i, float(dd[trough_i])))
                in_dd = False
    if in_dd:
        episodes.append((start_i, trough_i, None, float(dd[trough_i])))

    episodes.sort(key=lambda e: e[3])
    out = []
    for s, tr, rec, depth in episodes[:top]:
        out.append({
            "depth": depth,
            "start": dates[s].isoformat(),
            "trough": dates[tr].isoformat(),
            "recovered": dates[rec].isoformat() if rec is not None else None,
            "days_to_trough": (dates[tr] - dates[s]).days,
            "recovery_days": ((dates[rec] - dates[tr]).days if rec is not None else None),
        })
    return {
        "drawdowns": out,
        "summary": "; ".join(
            f"{o['depth'] * 100:.2f}% ({o['start']} → trough {o['trough']}"
            + (f", recovered {o['recovered']}" if o["recovered"] else ", not yet recovered")
            + ")" for o in out) or "no drawdowns",
    }


def correlation_matrix(fund_returns: dict[str, list[float]],
                       isins: list[str] | None = None) -> dict:
    """Pairwise correlation of the fund daily-return series."""
    keys = isins or sorted(fund_returns)
    m = np.array([fund_returns[k] for k in keys], dtype=float)
    with np.errstate(invalid="ignore"):
        c = np.corrcoef(m)
    pairs = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            v = float(c[a, b])
            if np.isfinite(v):
                pairs.append({"a": keys[a], "b": keys[b], "corr": v})
    pairs.sort(key=lambda p: abs(p["corr"]), reverse=True)
    return {"funds": keys, "top_pairs": pairs[:10]}


# ---------------------------------------------------------------- monthly stats
def monthly_stats(weight_timeline, fund_returns, dates,
                  bench_monthly: float | None = None,
                  year: int | None = None) -> dict:
    """Best/worst month, positive-month %, win rate vs a monthly benchmark rate,
    and historical month-over-month VaR (95/99). Optionally scoped to one calendar
    year (e.g. 'best month in 2026')."""
    from engine.benchmark import monthly_returns as _mr
    r = _series(weight_timeline, fund_returns, dates)
    m = _mr(dates, r)
    if year is not None:
        available = sorted({k[:4] for k, _ in m})
        m = [x for x in m if x[0].startswith(str(year))]
        if not m:
            raise InsufficientHistoryError(
                f"No data for {year}. Years available: {', '.join(available)}.",
                years_available=available)
    if len(m) < 1 or (year is None and len(m) < 3):
        raise InsufficientHistoryError(f"Only {len(m)} full/partial months; need >= 3.")
    vals = np.array([x for _, x in m], dtype=float)
    best_i, worst_i = int(np.argmax(vals)), int(np.argmin(vals))
    out = {
        "n_months": len(m),
        "best_month": {"month": m[best_i][0], "return": float(vals[best_i])},
        "worst_month": {"month": m[worst_i][0], "return": float(vals[worst_i])},
        "positive_months_pct": float(np.mean(vals > 0)),
        "avg_month": float(vals.mean()),
        "var_95_mom": float(-np.percentile(vals, 5)),   # loss magnitude, 95% 1-month VaR
        "var_99_mom": float(-np.percentile(vals, 1)),
    }
    if bench_monthly is not None:
        out["win_rate_vs_benchmark"] = float(np.mean(vals > bench_monthly))
        out["benchmark_monthly_rate"] = bench_monthly
    scope = f" in {year}" if year is not None else ""
    out["scope_year"] = year
    out["summary"] = (
        f"{len(m)} months{scope}: best {m[best_i][0]} {vals[best_i]*100:+.2f}%, worst "
        f"{m[worst_i][0]} {vals[worst_i]*100:+.2f}%, positive months "
        f"{out['positive_months_pct']*100:.0f}%, 95% MoM VaR {out['var_95_mom']*100:.2f}% "
        f"(historical).")
    return out


def rolling_period_returns(weight_timeline, fund_returns, dates,
                           window_months: int = 12) -> dict:
    """Rolling N-month compounded returns (12 or 36 typical)."""
    from engine.benchmark import monthly_returns as _mr
    r = _series(weight_timeline, fund_returns, dates)
    m = _mr(dates, r)
    if len(m) < window_months:
        raise InsufficientHistoryError(
            f"Need >= {window_months} months for rolling {window_months}m; have {len(m)}.")
    pts = []
    for i in range(window_months, len(m) + 1):
        chunk = [x for _, x in m[i - window_months:i]]
        pts.append({"month": m[i - 1][0],
                    "return": float(np.prod([1 + x for x in chunk]) - 1)})
    vals = [p["return"] for p in pts]
    return {"window_months": window_months, "points": pts,
            "latest": pts[-1], "min": float(min(vals)), "max": float(max(vals)),
            "summary": (f"Rolling {window_months}m return: latest {pts[-1]['return']*100:.2f}% "
                        f"({pts[-1]['month']}), range {min(vals)*100:.2f}%..{max(vals)*100:.2f}% "
                        f"across {len(pts)} windows.")}


# ------------------------------------------------------------------------ drift
def portfolio_drift(weight_timeline, fund_returns, dates) -> dict:
    """How far the portfolio has drifted from its target weights since the last
    rebalance, because funds moved by different amounts (buy-and-hold drift)."""
    segments = sorted(weight_timeline, key=lambda s: s[0])
    eff_date, target = segments[-1]
    idx = [t for t, d in enumerate(dates) if d >= eff_date]
    if not idx:
        raise InsufficientHistoryError("No price data on/after the last rebalance date.")
    growth = {}
    for isin, w in target.items():
        if w:
            rs = [fund_returns[isin][t] or 0.0 for t in idx]
            growth[isin] = float(np.prod(1.0 + np.array(rs)))
    total = sum(target[i] * growth[i] for i in growth)
    drifted = {i: target[i] * growth[i] / total for i in growth}
    rows = [{"isin": i, "fund": None, "target": float(target[i]),
             "drifted": float(drifted[i]),
             "drift_pp": float((drifted[i] - target[i]) * 100)} for i in growth]
    rows.sort(key=lambda x: -abs(x["drift_pp"]))
    total_drift = 0.5 * sum(abs(r["drifted"] - r["target"]) for r in rows)
    return {"effective_date": eff_date.isoformat(), "as_of": dates[-1].isoformat(),
            "funds": rows, "total_drift_pct": float(total_drift * 100),
            "summary": (f"Since the {eff_date.isoformat()} rebalance the portfolio has "
                        f"drifted {total_drift*100:.2f}% from target (largest: "
                        + ", ".join(f"{r['isin']} {r['drift_pp']:+.2f}pp" for r in rows[:3])
                        + ").")}


# ------------------------------------------------------- allocation breakdowns
def allocation_breakdown(weights: dict[str, float]) -> dict:
    """Weights grouped by asset class and geography (mandate-derived, approximate)."""
    import config as _c
    by_class: dict[str, float] = {}
    by_geo: dict[str, float] = {}
    for isin, w in weights.items():
        if not w:
            continue
        meta = _c.FUND_META.get(isin, {})
        by_class[meta.get("asset_class", "Unclassified")] = \
            by_class.get(meta.get("asset_class", "Unclassified"), 0.0) + w
        by_geo[meta.get("geography", "Unclassified")] = \
            by_geo.get(meta.get("geography", "Unclassified"), 0.0) + w
    return {
        "asset_allocation": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        "geographic_allocation": dict(sorted(by_geo.items(), key=lambda kv: -kv[1])),
        "note": ("Derived from each fund's mandate/name (approximate). Sector-level "
                 "allocation needs factsheet look-through data, which is not loaded."),
    }


def top_holdings(weights: dict[str, float], holdings_db: dict) -> dict:
    """Look-through top holdings: fund weight x holding weight, where factsheet data
    exists. Coverage is partial and stated explicitly."""
    import config as _c
    agg: dict[str, float] = {}
    covered = 0.0
    missing = []
    for isin, w in weights.items():
        if not w:
            continue
        h = holdings_db.get(isin)
        if not h:
            missing.append(f"{_c.FUND_NAMES.get(isin, isin)} ({w*100:.0f}%)")
            continue
        covered += w
        for item in h:
            agg[item["holding"]] = agg.get(item["holding"], 0.0) + w * item["weight"]
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:5]
    return {
        "top_holdings": [{"holding": k, "portfolio_weight": float(v)} for k, v in top],
        "lookthrough_coverage": float(covered),
        "funds_without_data": missing,
        "note": (f"Look-through covers {covered*100:.0f}% of the portfolio (factsheet "
                 "top-5 data exists for 3 of 12 funds). Holdings for the remaining funds "
                 "require their factsheets to be processed."),
    }


def compare_portfolios(codes: list[str], attribution_fn) -> dict:
    """Side-by-side metrics for several portfolios (attribution_fn supplied by the caller)."""
    rows = []
    for c in codes:
        m = attribution_fn(c)
        rows.append({"portfolio": c, "total_return": m["total_return"],
                     "ann_return": m["ann_return"], "ann_vol": m["ann_vol"],
                     "sharpe": m["sharpe"], "sortino": m["sortino"],
                     "max_drawdown": m["max_drawdown"]})
    rows.sort(key=lambda r: r["sharpe"], reverse=True)
    return {
        "portfolios": rows,
        "best_sharpe": rows[0]["portfolio"] if rows else None,
        "summary": "; ".join(
            f"{r['portfolio']}: ret {r['total_return'] * 100:.2f}%, vol "
            f"{r['ann_vol'] * 100:.2f}%, Sharpe {r['sharpe']:.4f}" for r in rows),
    }
