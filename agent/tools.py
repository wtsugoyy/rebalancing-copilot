"""Typed tool surface the LLM invokes.

Every tool wraps deterministic engine/store code, validates its arguments with
pydantic before touching anything, and returns JSON-serializable primitives. The LLM
never computes — it only selects a tool and narrates the returned numbers. No UI imports.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field, ValidationError

import config
from engine import analytics, perf, rf, sandbox
from engine.validate import CopilotError, NotFoundError
from store import history


@dataclass
class AppData:
    """The currently-loaded session data (from the uploaded files)."""
    dates: list[date]
    fund_returns: dict[str, list[float]]
    yields: list[tuple[date, float]]
    isins: list[str]


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    data: AppData | None  # None until the user uploads price + yield files


# --- argument schemas -------------------------------------------------------
_PortfolioLiteral = tuple(config.PORTFOLIO_CODES)


class PortfolioArg(BaseModel):
    portfolio: str = Field(description="Portfolio code: one of " + ", ".join(config.PORTFOLIO_CODES))

    def check(self):
        if self.portfolio not in config.PORTFOLIO_CODES:
            raise ValueError(f"portfolio must be one of {config.PORTFOLIO_CODES}")
        return self


class HistoryArg(PortfolioArg):
    limit: int = Field(default=10, ge=1, le=100)


class VolTargetArg(PortfolioArg):
    target: float = Field(description="Target annualised volatility as a decimal, e.g. 0.10 for 10%",
                          gt=0, lt=1)


class PeriodicArg(PortfolioArg):
    freq: str = Field(default="year",
                      description="Bucket size: 'year', 'quarter' or 'month'")

    def check(self):
        super().check()
        if self.freq.lower() not in ("year", "quarter", "month"):
            raise ValueError("freq must be one of: year, quarter, month")
        return self


class TrendArg(PortfolioArg):
    window: int = Field(default=63, ge=10, le=252,
                        description="Rolling window in trading days (63 ~= one quarter)")


class RangeArg(PortfolioArg):
    start: str | None = Field(default=None, description="Start date YYYY-MM-DD (inclusive)")
    end: str | None = Field(default=None, description="End date YYYY-MM-DD (inclusive)")

    def check(self):
        super().check()
        for v in (self.start, self.end):
            if v:
                try:
                    date.fromisoformat(v)
                except ValueError as exc:
                    raise ValueError(f"date must be YYYY-MM-DD, got {v!r}") from exc
        return self


class DrawdownArg(PortfolioArg):
    top: int = Field(default=3, ge=1, le=10)


class ComparePortfoliosArg(BaseModel):
    portfolios: list[str] = Field(description="Portfolio codes to compare, e.g. ['BAL','ADV']")

    def check(self):
        bad = [p for p in self.portfolios if p not in config.PORTFOLIO_CODES]
        if bad:
            raise ValueError(f"unknown portfolio(s) {bad}; must be in {config.PORTFOLIO_CODES}")
        if not self.portfolios:
            raise ValueError("supply at least one portfolio")
        return self


class AnalysisArg(PortfolioArg):
    code: str = Field(description="Python (pandas/numpy) assigning the answer to `result`. "
                                  "No imports, no I/O.", min_length=1, max_length=4000)


class OptimizeArg(BaseModel):
    sigma_target: float = Field(gt=0, lt=1,
                                description="Hard annualised volatility cap as a decimal, "
                                            "e.g. 0.035 for 3.5%")
    bounds: dict[str, tuple[float, float]] | None = Field(
        default=None, description="Optional per-ISIN (min%, max%) overrides, e.g. "
                                  '{"ZZ0000000004": [0, 45]}')

    def check(self):
        for isin in (self.bounds or {}):
            if isin not in config.FUND_UNIVERSE:
                raise ValueError(f"unknown ISIN in bounds: {isin}")
        return self


class MonthlyArg(PortfolioArg):
    year: int | None = Field(default=None, ge=1990, le=2100,
                             description="Optional calendar year to scope to, e.g. 2026")


class RollingArg(PortfolioArg):
    window_months: int = Field(default=12, description="12 or 36 typically")

    def check(self):
        super().check()
        if self.window_months not in (3, 6, 12, 24, 36):
            raise ValueError("window_months must be one of 3, 6, 12, 24, 36")
        return self


# --- helpers ----------------------------------------------------------------
def _require_data(ctx: ToolContext) -> AppData:
    if ctx.data is None:
        raise NotFoundError("No price/yield data loaded yet. Upload the NAV and yield "
                            "files first, then ask again.")
    return ctx.data


def _attribution(ctx: ToolContext, portfolio: str) -> dict:
    data = _require_data(ctx)
    timeline = history.chained_timeline(ctx.conn, portfolio)
    if not timeline:
        raise NotFoundError(f"No committed allocation for '{portfolio}' to attribute.",
                            portfolio=portfolio)
    port = perf.portfolio_returns(timeline, data.fund_returns, data.dates)
    # attribution is only defined over dates the timeline covers
    covered_dates = [d for d in data.dates if d >= timeline[0][0]]
    rf_period = rf.rf_period_series(covered_dates, data.yields)
    excess = perf.excess_returns(port, rf_period)
    avg_rf = rf.average_rf(covered_dates[0], covered_dates[-1], data.yields)
    return perf.compute_metrics(port, excess, covered_dates, avg_rf).to_dict()


# --- tool implementations (LLM-facing) --------------------------------------
def list_portfolios() -> dict:
    """List the available portfolio codes and their full names."""
    return {"portfolios": [{"code": c, "name": config.PORTFOLIO_NAMES[c]}
                           for c in config.PORTFOLIO_CODES]}


def get_current_portfolio(ctx: ToolContext, portfolio: str) -> dict:
    """Get the current live weights and effective date for a portfolio."""
    snap = history.get_current(ctx.conn, portfolio)
    return {"portfolio": portfolio, "effective_date": snap.effective_date,
            "weights": {k: v for k, v in snap.weights.items() if v},
            "source": snap.source}


def get_history(ctx: ToolContext, portfolio: str, limit: int = 10) -> dict:
    """List recent rebalance snapshots (newest first) for a portfolio."""
    snaps = history.list_history(ctx.conn, portfolio, limit)
    return {"portfolio": portfolio, "snapshots": [
        {"effective_date": s.effective_date, "source": s.source,
         "sharpe": (s.metrics or {}).get("sharpe")} for s in snaps]}


def attribution(ctx: ToolContext, portfolio: str) -> dict:
    """Compute realized performance attribution (total/annualised return, volatility,
    max drawdown, Sharpe, Sortino) for a portfolio over the loaded price history."""
    m = _attribution(ctx, portfolio)
    # A pre-labelled string so a small model can quote it verbatim without mislabelling
    # which figure is which (e.g. confusing total vs annualised return).
    m["summary"] = (
        f"{portfolio} over {m['period_start']}..{m['period_end']} ({m['n_obs']} obs): "
        f"total return {m['total_return']*100:.2f}%, annualised return {m['ann_return']*100:.2f}%, "
        f"annualised volatility {m['ann_vol']*100:.2f}%, max drawdown {m['max_drawdown']*100:.2f}%, "
        f"Sharpe {m['sharpe']:.4f}, Sortino {m['sortino']:.4f}.")
    return {"portfolio": portfolio, **m}


def vol_vs_target(ctx: ToolContext, portfolio: str, target: float) -> dict:
    """How far is a portfolio's realized annualised volatility from a target (decimal)."""
    m = _attribution(ctx, portfolio)
    vol = m["ann_vol"]
    return {"portfolio": portfolio, "ann_vol": vol, "target": target,
            "gap_pp": (vol - target) * 100.0}


# --- analyst toolkit (deterministic time-series analysis) --------------------
def _timeline_and_data(ctx: ToolContext, portfolio: str):
    data = _require_data(ctx)
    timeline = history.chained_timeline(ctx.conn, portfolio)
    if not timeline:
        raise NotFoundError(f"No committed allocation for '{portfolio}'.", portfolio=portfolio)
    covered = [d for d in data.dates if d >= timeline[0][0]]
    return data, timeline, covered


def periodic_returns(ctx: ToolContext, portfolio: str, freq: str = "year") -> dict:
    """Return and volatility for EACH calendar year / quarter / month in the period.
    Use this for 'what was the return in each of the 3 years?' or 'yearly returns'."""
    data, timeline, covered = _timeline_and_data(ctx, portfolio)
    out = analytics.periodic_returns(timeline, data.fund_returns, covered, freq)
    return {"portfolio": portfolio, **out}


def volatility_trend(ctx: ToolContext, portfolio: str, window: int = 63) -> dict:
    """Is volatility INCREASING or DECREASING? Rolling annualised volatility, its
    direction, recent-vs-prior comparison and trend slope. Use this for any question
    about a trend, or whether risk is rising/falling."""
    data, timeline, covered = _timeline_and_data(ctx, portfolio)
    out = analytics.volatility_trend(timeline, data.fund_returns, covered, window)
    return {"portfolio": portfolio, **out}


def attribution_period(ctx: ToolContext, portfolio: str,
                       start: str | None = None, end: str | None = None) -> dict:
    """Full attribution metrics over a SPECIFIC date range (YYYY-MM-DD), e.g. one year.
    Use this to compare sub-periods such as 2024 vs 2025."""
    data, timeline, covered = _timeline_and_data(ctx, portfolio)
    s = date.fromisoformat(start) if start else None
    e = date.fromisoformat(end) if end else None
    out = analytics.attribution_range(timeline, data.fund_returns, covered, data.yields, s, e)
    return {"portfolio": portfolio, "start": start, "end": end, **out}


def fund_contribution(ctx: ToolContext, portfolio: str) -> dict:
    """Which funds drove the portfolio's return and its risk (contribution breakdown)."""
    data, timeline, covered = _timeline_and_data(ctx, portfolio)
    out = analytics.fund_contribution(timeline, data.fund_returns, covered)
    return {"portfolio": portfolio, **out}


def drawdown_periods(ctx: ToolContext, portfolio: str, top: int = 3) -> dict:
    """The deepest drawdowns with their start/trough/recovery dates."""
    data, timeline, covered = _timeline_and_data(ctx, portfolio)
    out = analytics.drawdown_periods(timeline, data.fund_returns, covered, top)
    return {"portfolio": portfolio, **out}


def compare_portfolios(ctx: ToolContext, portfolios: list[str]) -> dict:
    """Side-by-side metrics for several portfolios, ranked by Sharpe."""
    out = analytics.compare_portfolios(portfolios, lambda c: _attribution(ctx, c))
    return out


def correlations(ctx: ToolContext) -> dict:
    """Pairwise correlation between the funds' daily returns."""
    data = _require_data(ctx)
    return analytics.correlation_matrix(data.fund_returns, data.isins)


def run_analysis(ctx: ToolContext, portfolio: str, code: str) -> dict:
    """ADVISORY ad-hoc analysis: run pandas/numpy code over the real return data when no
    other tool fits. Assign the answer to `result`. Available: `df` (DataFrame indexed by
    date, one column per fund ISIN plus 'portfolio'), `returns` (portfolio daily return
    Series), `pd`, `np`. No imports, no file/network access.
    Example: result = returns.resample('YE').apply(lambda s: (1+s).prod()-1)"""
    data, timeline, covered = _timeline_and_data(ctx, portfolio)
    port = perf.portfolio_returns(timeline, data.fund_returns, covered)
    fr = {k: [v[i] for i, d in enumerate(data.dates) if d >= timeline[0][0]]
          for k, v in data.fund_returns.items()}
    return {"portfolio": portfolio,
            **sandbox.run_analysis(code, covered, port, fr)}


# --- rebalancing engine + benchmark + composition tools ----------------------
def optimize_portfolio(ctx: ToolContext, sigma_target: float,
                       bounds: dict | None = None) -> dict:
    """Run the rebalancing engine (mean-variance, hard vol cap) on the loaded data."""
    from engine import optimizer as opt
    data = _require_data(ctx)
    bp = None
    if bounds:
        bp = {k: (float(v[0]), float(v[1])) for k, v in bounds.items()}
    res = opt.optimize(data.fund_returns, data.dates, sigma_target, bp,
                       isins=list(config.FUND_UNIVERSE)).to_dict()
    res["advisory"] = True
    nz = {i: w for i, w in res["weights_rounded"].items() if w > 0.0005}
    res["summary"] = (
        f"Optimised at vol target {sigma_target*100:.2f}%: expected return "
        f"{res['expected_return']*100:.2f}% at vol {res['expected_vol']*100:.2f}% "
        f"(Sharpe {res['sharpe']:.2f}). Implementable (rounded) allocation: "
        + ", ".join(f"{config.FUND_NAMES.get(i, i)} {w*100:.0f}%" for i, w in
                    sorted(nz.items(), key=lambda kv: -kv[1]))
        + ". Advisory — committing a live allocation stays a human decision.")
    return res


def target_vs_realized(ctx: ToolContext, portfolio: str) -> dict:
    """'Fixed vs Current': the committed sigma target (and expected return, if recorded)
    versus the REALIZED volatility and return."""
    snap = history.get_current(ctx.conn, portfolio)
    m = _attribution(ctx, portfolio)
    out = {
        "portfolio": portfolio,
        "fixed_sigma_target": snap.sigma_target,
        "realized_ann_vol": m["ann_vol"],
        "vol_gap_pp": ((m["ann_vol"] - snap.sigma_target) * 100
                       if snap.sigma_target else None),
        "realized_ann_return": m["ann_return"],
        "realized_total_return": m["total_return"],
        "effective_date": snap.effective_date,
    }
    tgt = f"{snap.sigma_target*100:.2f}%" if snap.sigma_target else "not recorded"
    gap = f" (gap {out['vol_gap_pp']:+.2f}pp)" if out["vol_gap_pp"] is not None else ""
    out["summary"] = (
        f"{portfolio}: fixed vol target {tgt} vs realized vol {m['ann_vol']*100:.2f}%{gap}; "
        f"realized annualised return {m['ann_return']*100:.2f}% "
        f"(total {m['total_return']*100:.2f}%).")
    return out


def composition_change(ctx: ToolContext, portfolio: str) -> dict:
    """'Composition before vs after': the previous snapshot's weights vs the current
    ones, with per-fund deltas and turnover."""
    cur = history.get_current(ctx.conn, portfolio)
    try:
        prev = history.get_previous(ctx.conn, portfolio)
    except CopilotError:
        return {"portfolio": portfolio,
                "current": {k: v for k, v in cur.weights.items() if v},
                "summary": f"{portfolio} has only one recorded allocation "
                           f"(effective {cur.effective_date}) — nothing to compare yet."}
    rows = []
    for isin in config.FUND_UNIVERSE:
        b, a = prev.weights.get(isin, 0.0), cur.weights.get(isin, 0.0)
        if b or a:
            rows.append({"isin": isin, "fund": config.FUND_NAMES.get(isin, isin),
                         "before": b, "after": a, "delta_pp": (a - b) * 100})
    turnover = 0.5 * sum(abs(r["after"] - r["before"]) for r in rows)
    moved = sorted((r for r in rows if abs(r["delta_pp"]) > 0.05),
                   key=lambda r: -abs(r["delta_pp"]))
    return {"portfolio": portfolio, "before_date": prev.effective_date,
            "after_date": cur.effective_date, "funds": rows,
            "turnover": turnover,
            "summary": (f"{portfolio} rebalance {prev.effective_date} → "
                        f"{cur.effective_date}: turnover {turnover*100:.1f}%. Moves: "
                        + ("; ".join(f"{r['fund']} {r['before']*100:.0f}%→{r['after']*100:.0f}%"
                                     for r in moved[:5]) or "none"))}


def benchmark_metrics(ctx: ToolContext, portfolio: str) -> dict:
    """Portfolio vs the ACTIVE benchmark: per-period returns, tracking error,
    information ratio, win rate; beta and YTM notes."""
    from engine import benchmark as bm
    data, timeline, covered = _timeline_and_data(ctx, portfolio)
    port = perf.portfolio_returns(timeline, data.fund_returns, covered)
    bench = bm.get_active_benchmark(ctx.conn)
    out = bm.benchmark_comparison(covered, port, bench)
    out["portfolio"] = portfolio
    out["ytm_note"] = ("YTM cannot be derived from NAV returns; it requires the funds' "
                       "bond-holdings data (not loaded).")
    ann = next((p for p in out["per_period"] if p["period"] == "annualized"), None)
    out["summary"] = (
        f"{portfolio} vs '{out['benchmark']}': annualised "
        f"{(ann['portfolio'] or 0)*100:.2f}% vs benchmark {(ann['benchmark'] or 0)*100:.2f}% "
        f"(excess {(ann['excess'] or 0)*100:+.2f}pp); TE "
        f"{(out['tracking_error_ann'] or 0)*100:.2f}%, IR "
        f"{out['information_ratio'] if out['information_ratio'] is not None else 'n/a'}, "
        f"monthly win rate {(out['win_rate_vs_benchmark'] or 0)*100:.0f}%. "
        f"Beta: n/a vs a flat-rate benchmark.")
    return out


def monthly_stats(ctx: ToolContext, portfolio: str, year: int | None = None) -> dict:
    """Best/worst month, positive-month %, win rate vs benchmark, VaR (MoM); optionally
    scoped to a single calendar year (e.g. best/worst month IN 2026)."""
    from engine import benchmark as bm
    data, timeline, covered = _timeline_and_data(ctx, portfolio)
    bench = bm.get_active_benchmark(ctx.conn)
    bench_ann = (bench["periods"].get("annualized") or 0.0) / 100.0
    bench_monthly = (1 + bench_ann) ** (1 / 12) - 1
    out = analytics.monthly_stats(timeline, data.fund_returns, covered, bench_monthly,
                                  year=year)
    out["portfolio"] = portfolio
    out["benchmark"] = bench["name"]
    return out


def rolling_returns(ctx: ToolContext, portfolio: str, window_months: int = 12) -> dict:
    """Rolling 12- or 36-month compounded returns."""
    data, timeline, covered = _timeline_and_data(ctx, portfolio)
    out = analytics.rolling_period_returns(timeline, data.fund_returns, covered,
                                           window_months)
    out["portfolio"] = portfolio
    return out


def portfolio_drift(ctx: ToolContext, portfolio: str) -> dict:
    """Buy-and-hold drift away from target weights since the last rebalance."""
    data, timeline, covered = _timeline_and_data(ctx, portfolio)
    out = analytics.portfolio_drift(timeline, data.fund_returns, covered)
    for r in out["funds"]:
        r["fund"] = config.FUND_NAMES.get(r["isin"], r["isin"])
    out["portfolio"] = portfolio
    return out


def allocation_breakdown(ctx: ToolContext, portfolio: str) -> dict:
    """Asset-class and geographic breakdown of the current allocation (approximate,
    mandate-derived). Sector data requires factsheets and is flagged unavailable."""
    snap = history.get_current(ctx.conn, portfolio)
    out = analytics.allocation_breakdown(snap.weights)
    out["portfolio"] = portfolio
    out["effective_date"] = snap.effective_date
    aa = out["asset_allocation"]
    out["summary"] = (f"{portfolio} allocation by asset class: "
                      + ", ".join(f"{k} {v*100:.0f}%" for k, v in aa.items())
                      + ". Geography: "
                      + ", ".join(f"{k} {v*100:.0f}%" for k, v
                                  in out["geographic_allocation"].items())
                      + ". (Mandate-derived, approximate; sector needs factsheet data.)")
    return out


def _load_holdings() -> dict:
    import json as _json
    import os as _os
    p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                      "sample", "top5_holdings.json")
    if _os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return _json.load(fh)
    return {}


def top_holdings(ctx: ToolContext, portfolio: str) -> dict:
    """Look-through top-5 holdings from factsheet data (partial coverage, stated)."""
    snap = history.get_current(ctx.conn, portfolio)
    out = analytics.top_holdings(snap.weights, _load_holdings())
    out["portfolio"] = portfolio
    tops = out["top_holdings"]
    out["summary"] = ((f"{portfolio} look-through top holdings: "
                       + ", ".join(f"{t['holding']} {t['portfolio_weight']*100:.2f}%"
                                   for t in tops) + ". ") if tops else
                      f"No look-through holdings computable for {portfolio}. ") + out["note"]
    return out


# name -> (callable, needs_ctx, arg_model)
REGISTRY = {
    "list_portfolios": (list_portfolios, False, None),
    "get_current_portfolio": (get_current_portfolio, True, PortfolioArg),
    "get_history": (get_history, True, HistoryArg),
    "attribution": (attribution, True, PortfolioArg),
    "vol_vs_target": (vol_vs_target, True, VolTargetArg),
    # analyst toolkit
    "periodic_returns": (periodic_returns, True, PeriodicArg),
    "volatility_trend": (volatility_trend, True, TrendArg),
    "attribution_period": (attribution_period, True, RangeArg),
    "fund_contribution": (fund_contribution, True, PortfolioArg),
    "drawdown_periods": (drawdown_periods, True, DrawdownArg),
    "compare_portfolios": (compare_portfolios, True, ComparePortfoliosArg),
    "correlations": (correlations, True, None),
    # fenced sandbox (advisory)
    "run_analysis": (run_analysis, True, AnalysisArg),
    # rebalancing engine + benchmark suite
    "optimize_portfolio": (optimize_portfolio, True, OptimizeArg),
    "target_vs_realized": (target_vs_realized, True, PortfolioArg),
    "composition_change": (composition_change, True, PortfolioArg),
    "benchmark_metrics": (benchmark_metrics, True, PortfolioArg),
    "monthly_stats": (monthly_stats, True, MonthlyArg),
    "rolling_returns": (rolling_returns, True, RollingArg),
    "portfolio_drift": (portfolio_drift, True, PortfolioArg),
    "allocation_breakdown": (allocation_breakdown, True, PortfolioArg),
    "top_holdings": (top_holdings, True, PortfolioArg),
}


def execute_tool(name: str, args: dict, ctx: ToolContext) -> dict:
    """Validate args (pydantic) then run the tool. Returns a JSON-serializable dict.

    Any failure is returned as a structured {"error": ...} object (never raised past
    this boundary) so the model can self-correct."""
    from engine.obs import get_logger
    get_logger().info("tool_call name=%s args=%s", name, args)
    entry = REGISTRY.get(name)
    if entry is None:
        return {"error": f"Unknown tool '{name}'.", "code": "UNKNOWN_TOOL"}
    func, needs_ctx, model = entry
    try:
        validated = {}
        if model is not None:
            m = model(**(args or {}))
            if hasattr(m, "check"):
                m.check()
            validated = m.model_dump()
        if needs_ctx:
            return func(ctx, **validated)
        return func(**validated)
    except ValidationError as exc:
        return {"error": f"Invalid arguments for {name}: {exc.errors()}",
                "code": "VALIDATION_ERROR"}
    except ValueError as exc:
        return {"error": str(exc), "code": "VALIDATION_ERROR"}
    except CopilotError as exc:
        return exc.to_dict()
    except Exception as exc:  # noqa: BLE001 - never leak a raw stack past the boundary
        return {"error": f"{type(exc).__name__}: {exc}", "code": "INTERNAL_ERROR"}
