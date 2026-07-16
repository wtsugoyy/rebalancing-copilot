"""Benchmark-relative analytics (plan: Benchmark tab + copilot asks 1/3/5).

The benchmark model mirrors the engine's Benchmark tab: a named preset with
user-editable PER-PERIOD returns in percent (annualized, 1m, 3m, 6m, ytd, 1y, 3y,
5y, all). The active benchmark is whatever was last saved to the store.

Honesty notes baked into outputs:
* Beta vs a flat-rate benchmark is undefined (a constant-rate series has zero
  variance). We return null with the reason instead of a fake number.
* YTM cannot be derived from NAV returns; the tool says so rather than guessing.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date

import numpy as np

import config

MONTHS_PER = {"1m": 1, "3m": 3, "6m": 6, "1y": 12, "3y": 36, "5y": 60}


# --- store --------------------------------------------------------------------
def save_benchmark(conn: sqlite3.Connection, name: str, periods: dict[str, float]) -> int:
    cur = conn.execute("INSERT INTO benchmarks(name, periods_json) VALUES (?,?)",
                       (name, json.dumps(periods)))
    conn.commit()
    return int(cur.lastrowid)


def get_active_benchmark(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT name, periods_json FROM benchmarks ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return {"name": config.DEFAULT_BENCHMARK,
                "periods": dict(config.BENCHMARK_PRESETS[config.DEFAULT_BENCHMARK]),
                "source": "default preset (none saved yet)"}
    return {"name": row[0], "periods": json.loads(row[1]), "source": "saved"}


# --- portfolio period returns ---------------------------------------------------
def monthly_returns(dates: list[date], daily: list[float]) -> list[tuple[str, float]]:
    """Compound daily returns into calendar months -> [('2024-01', r), ...]."""
    buckets: dict[str, float] = {}
    order: list[str] = []
    for d, r in zip(dates, daily):
        k = f"{d.year}-{d.month:02d}"
        if k not in buckets:
            buckets[k] = 1.0
            order.append(k)
        buckets[k] *= (1.0 + r)
    return [(k, buckets[k] - 1.0) for k in order]


def trailing_period_returns(dates: list[date], daily: list[float]) -> dict[str, float | None]:
    """Portfolio returns over the benchmark tab's periods, ending at the last date."""
    m = monthly_returns(dates, daily)
    out: dict[str, float | None] = {}
    for key, n in MONTHS_PER.items():
        if len(m) >= n:
            out[key] = float(np.prod([1 + r for _, r in m[-n:]]) - 1)
        else:
            out[key] = None
    last_year = dates[-1].year
    ytd = [r for k, r in m if k.startswith(str(last_year))]
    out["ytd"] = float(np.prod([1 + r for r in ytd]) - 1) if ytd else None
    out["all"] = float(np.prod([1 + r for _, r in m]) - 1)
    el_days = max((dates[-1] - dates[0]).days, 1)
    out["annualized"] = float((1 + out["all"]) ** (365.0 / el_days) - 1)
    return out


# --- relative metrics -----------------------------------------------------------
def benchmark_comparison(dates: list[date], daily: list[float],
                         bench: dict) -> dict:
    """Portfolio vs the active benchmark: per-period excess, TE, IR, win rate, beta note."""
    periods = bench["periods"]
    port = trailing_period_returns(dates, daily)

    per_period = []
    for key in config.BENCHMARK_PERIODS:
        b = periods.get(key)
        p = port.get(key)
        per_period.append({
            "period": key,
            "portfolio": p,
            "benchmark": (b / 100.0) if b is not None else None,
            "excess": (p - b / 100.0) if (p is not None and b is not None) else None,
        })

    m = monthly_returns(dates, daily)
    mr = np.array([r for _, r in m], dtype=float)
    bench_ann = (periods.get("annualized") or 0.0) / 100.0
    bench_monthly = (1 + bench_ann) ** (1 / 12) - 1

    excess_m = mr - bench_monthly
    te = float(np.std(excess_m, ddof=1) * np.sqrt(12)) if mr.size > 1 else None
    port_ann = port["annualized"]
    ir = ((port_ann - bench_ann) / te) if (te and te > 0 and port_ann is not None) else None
    win_rate = float(np.mean(mr > bench_monthly)) if mr.size else None

    return {
        "benchmark": bench["name"],
        "benchmark_source": bench.get("source", ""),
        "per_period": per_period,
        "tracking_error_ann": te,
        "information_ratio": ir,
        "win_rate_vs_benchmark": win_rate,
        "n_months": int(mr.size),
        "beta": None,
        "beta_note": ("Beta is undefined against a flat-rate benchmark (constant return "
                      "series has zero variance). Supply a market index return series to "
                      "compute a meaningful beta."),
    }
