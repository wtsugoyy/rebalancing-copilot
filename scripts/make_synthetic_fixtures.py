"""Generate every demo and test fixture in this repo from a seeded RNG.

Run from the repo root:

    python scripts/make_synthetic_fixtures.py

Writes:
    tests/fixtures/golden.json            attribution regression baseline
    tests/fixtures/solutions_golden.json  optimizer regression baseline
    sample/nav_sample.csv                 demo NAV series (upload in the UI)
    sample/yield_sample.csv               demo risk-free yield series
    sample/top5_holdings.json             demo look-through holdings

WHAT THE GOLDEN FILES ARE, AND ARE NOT
--------------------------------------
The expected values below are produced by running THIS repo's engine over the
synthetic inputs and recording what it returned. They are *regression* baselines:
they fail loudly if a refactor changes a number, which is what you want from a CI
gate. They are NOT an independent oracle -- they cannot tell you the engine is
correct, because the engine computed them. Verifying the metric definitions
themselves against an external reference implementation is a separate exercise and
is not something this repository can do on its own.

Regenerating is therefore a deliberate act: if a diff to `engine/` changes these
numbers, that is the test doing its job. Re-run this script only when you have
decided the new behaviour is right.

The universe (ISINs, names, risk parameters) lives in `config.py`. The `ZZ` ISIN
prefix is reserved for user-assigned codes and is never issued to a real security.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from engine import optimizer, perf, rf  # noqa: E402

SEED = 20260717
TRADING_DAYS = 252

# Loadings on a single common market factor, per fund. Cash barely moves with the
# market; equity and commodity-linked equity move with it a lot. This gives the
# covariance matrix a realistic structure so `correlations` and the optimizer have
# something non-trivial to chew on.
FACTOR_LOADING = {
    "ZZ0000000001": 0.02, "ZZ0000000002": 0.02, "ZZ0000000003": 0.01,
    "ZZ0000000004": 0.03, "ZZ0000000005": 0.15, "ZZ0000000006": 0.18,
    "ZZ0000000007": 0.01, "ZZ0000000008": 0.30, "ZZ0000000009": 0.85,
    "ZZ0000000010": 0.45, "ZZ0000000011": 0.80, "ZZ0000000012": 0.55,
}


def business_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_returns(rng: np.random.Generator, dates: list[date]) -> dict[str, list[float]]:
    """Daily simple returns via a one-factor model + idiosyncratic noise."""
    n = len(dates)
    market = rng.standard_normal(n)
    out: dict[str, list[float]] = {}
    for isin in config.FUND_UNIVERSE:
        drift, vol = config.SYNTHETIC_FUND_PARAMS[isin]
        beta = FACTOR_LOADING[isin]
        d_vol = vol / np.sqrt(TRADING_DAYS)
        d_mu = drift / TRADING_DAYS
        # Split variance between the common factor and idiosyncratic noise so the
        # realised total vol still lands on the target.
        idio = np.sqrt(max(1.0 - beta ** 2, 0.0))
        shock = beta * market + idio * rng.standard_normal(n)
        r = d_mu + d_vol * shock
        out[isin] = [float(x) for x in r]
    return out


def make_yields(rng: np.random.Generator, start: date, end: date) -> list[dict]:
    """A slow random walk in [2.8, 4.2] percent on every calendar day."""
    days = (end - start).days + 1
    level, out = 3.30, []
    for i in range(days):
        level += float(rng.normal(0, 0.006))
        level = min(max(level, 2.80), 4.20)
        out.append({"date": (start + timedelta(days=i)).isoformat(),
                    "yield_pct": round(level, 3)})
    return out


def build_golden() -> dict:
    rng = np.random.default_rng(SEED)
    # ~3.25 years, spanning calendar years 2023-2026. The span is deliberate: the
    # analytics tests bucket by year and roll a 36-month window, so a shorter
    # series would leave those paths untested.
    dates = business_days(date(2023, 1, 2), 820)
    fund_returns = make_returns(rng, dates)

    # Yields start before the first NAV date (the lookup needs a key >= the target)
    # and run past the last, so the full series covers the whole window.
    yields = make_yields(rng, dates[0] - timedelta(days=60), dates[-1] + timedelta(days=30))
    y_pairs = [(date.fromisoformat(y["date"]), y["yield_pct"]) for y in yields]

    # Cut the lookup range short of the last NAV date on purpose: dates past the
    # cutoff must fall back to RF_FALLBACK, which keeps that branch under test.
    cutoff = dates[int(len(dates) * 0.9)]
    capped = [(d, y) for d, y in y_pairs if d <= cutoff]

    rf_period = rf.rf_period_series(dates, capped)
    avg_rf = rf.average_rf(dates[0], dates[-1], y_pairs)

    portfolios = {}
    for code in config.PORTFOLIO_CODES:
        wmap = dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS[code]))
        seg = [(dates[0], wmap)]
        port = perf.portfolio_returns(seg, fund_returns, dates)
        excess = perf.excess_returns(port, rf_period)
        metrics = perf.compute_metrics(port, excess, dates, avg_rf).to_dict()
        portfolios[code] = {
            "weights": wmap,
            "portfolio_return_expected": [float(x) for x in port],
            "excess_expected": [float(x) for x in excess],
            "expected": {k: float(metrics[k]) for k in
                         ("total_return", "ann_return", "ann_vol",
                          "max_drawdown", "sharpe", "sortino")},
        }

    return {
        "isins": list(config.FUND_UNIVERSE),
        "dates": [d.isoformat() for d in dates],
        "fund_returns": fund_returns,
        "rf_period_expected": [float(x) for x in rf_period],
        "yields": yields,
        "rf_lookup_cutoff": cutoff.isoformat(),
        "avg_rf_expected": float(avg_rf),
        "portfolios": portfolios,
        "meta": {
            "n_rows": len(dates),
            "source": "synthetic (scripts/make_synthetic_fixtures.py)",
            "seed": SEED,
            "kind": "regression baseline, not an independent oracle",
        },
    }


def build_solutions() -> dict:
    rng = np.random.default_rng(SEED + 1)
    # Its own window, independent of golden.json — the optimizer fixture only needs
    # enough observations to estimate a stable covariance matrix.
    dates = business_days(date(2022, 9, 1), 900)
    fund_returns = make_returns(rng, dates)

    mu, vol, _ = optimizer.fund_stats(fund_returns, config.FUND_UNIVERSE)
    bounds = dict(config.DEFAULT_FUND_BOUNDS)

    blocks = {}
    for target in (0.035, 0.045, 0.055):
        r = optimizer.optimize(fund_returns, dates, target, bounds, config.FUND_UNIVERSE)
        blocks[f"block_{target * 100:.1f}"] = {
            "vol_target": target,
            "expected_return": round(float(r.expected_return), 4),
            "funds": [
                {
                    "isin": isin,
                    "min": bounds[isin][0],
                    "max": bounds[isin][1],
                    "value1": round(float(mu[k]), 4),
                    "value2": round(float(vol[k]), 4),
                    "allocation": round(float(r.weights_rounded[isin]), 6),
                }
                for k, isin in enumerate(config.FUND_UNIVERSE)
            ],
        }

    return {
        "isins": list(config.FUND_UNIVERSE),
        "fund_names": dict(config.FUND_NAMES),
        "dates": [d.isoformat() for d in dates],
        "fund_returns": fund_returns,
        "blocks": blocks,
        "meta": {
            "n_rows": len(dates),
            "source": "synthetic (scripts/make_synthetic_fixtures.py)",
            "seed": SEED + 1,
            "kind": "regression baseline, not an independent oracle",
        },
    }


def write_samples(golden: dict) -> None:
    """Demo uploads: NAV is the cumulative index implied by the golden returns."""
    sample = ROOT / "sample"
    sample.mkdir(exist_ok=True)

    dates, isins = golden["dates"], golden["isins"]
    nav = {i: np.cumprod(1.0 + np.asarray(golden["fund_returns"][i])) for i in isins}
    # Rebase so every fund starts at exactly 1.0 on day one.
    for i in isins:
        nav[i] = nav[i] / nav[i][0]

    lines = ["Date," + ",".join(isins)]
    for k, d in enumerate(dates):
        lines.append(d + "," + ",".join(f"{nav[i][k]:.10f}" for i in isins))
    (sample / "nav_sample.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ylines = ["Date,Yield"] + [f"{y['date']},{y['yield_pct']}" for y in golden["yields"]]
    (sample / "yield_sample.csv").write_text("\n".join(ylines) + "\n", encoding="utf-8")

    # Look-through holdings for the three funds that have them. Invented issuers;
    # the point is to exercise the partial-coverage path (3 of 12 funds), because
    # the tools must say so rather than quietly imply full coverage.
    holdings = {
        "ZZ0000000012": [
            {"holding": "Northwind Mining Corp", "weight": 0.0861},
            {"holding": "Silverpeak Resources Ltd", "weight": 0.0752},
            {"holding": "Ironbark Gold Ltd", "weight": 0.0644},
            {"holding": "Meridian Metals Inc", "weight": 0.0588},
            {"holding": "Calder Bay Minerals", "weight": 0.0501},
        ],
        "ZZ0000000010": [
            {"holding": "Bullion Trust ETF", "weight": 0.2140},
            {"holding": "Northwind Mining Corp", "weight": 0.0925},
            {"holding": "Harbour Gold Holdings", "weight": 0.0710},
            {"holding": "Silverpeak Resources Ltd", "weight": 0.0663},
            {"holding": "Redstone Exploration", "weight": 0.0402},
        ],
        "ZZ0000000009": [
            {"holding": "Pacific Semiconductor Co", "weight": 0.0712},
            {"holding": "Eastgate Financial Group", "weight": 0.0655},
            {"holding": "Lantern Telecom Bhd", "weight": 0.0530},
            {"holding": "Kestrel Industries", "weight": 0.0487},
            {"holding": "Blue Harbour Logistics", "weight": 0.0421},
        ],
    }
    (sample / "top5_holdings.json").write_text(
        json.dumps(holdings, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    fixtures = ROOT / "tests" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)

    golden = build_golden()
    (fixtures / "golden.json").write_text(json.dumps(golden), encoding="utf-8")
    print(f"golden.json           {golden['meta']['n_rows']} rows x {len(golden['isins'])} funds")
    for code, p in golden["portfolios"].items():
        e = p["expected"]
        print(f"  {code:<5} total={e['total_return']:+.4f} vol={e['ann_vol']:.4f} "
              f"sharpe={e['sharpe']:+.3f}")

    sol = build_solutions()
    (fixtures / "solutions_golden.json").write_text(json.dumps(sol), encoding="utf-8")
    print(f"solutions_golden.json {sol['meta']['n_rows']} rows")
    for name, b in sol["blocks"].items():
        held = sum(1 for f in b["funds"] if f["allocation"] > 0)
        print(f"  {name:<10} target_vol={b['vol_target']:.3f} "
              f"ER={b['expected_return']:+.4f} funds_held={held}")

    write_samples(golden)
    print("sample/nav_sample.csv, sample/yield_sample.csv, sample/top5_holdings.json")


if __name__ == "__main__":
    main()
