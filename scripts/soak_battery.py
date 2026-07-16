"""Live-model soak battery — run inside the app container against REAL data files.

Usage:  python scripts/soak_battery.py /tmp/nav.csv /tmp/yield.csv [runs]

Each question has DETERMINISTIC pass criteria: required tool(s) called, required
token(s) in the answer, and zero refusal markers. The battery runs N times (default 2)
to check consistency of tool selection at temperature 0. Exit code 0 = all pass.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent import graph
from agent.tools import AppData, ToolContext
from engine import ingest
from store import history

REFUSAL_MARKERS = ("not yet occurred", "has not occurred", "in the future",
                   "cannot provide results", "please clarify", "hypothetical scenario",
                   "does not exist")

# (question, selected_portfolio, required_tools_any, required_answer_tokens_any)
BATTERY = [
    ("what is the portfolio's Best Month / Worst Month in 2026", "SC",
     {"monthly_stats"}, ("2026",)),
    ("Who is the firm and which portfolios do you cover?", "SC",
     set(), ("Malaysian", "BAL")),
    ("How far is the portfolio's volatility from its committed target?", "BAL",
     {"target_vs_realized", "vol_vs_target"}, ("%",)),
    # accept any natural phrasing of contribution ("contributed"/"contributor"/…)
    ("Which funds drove the portfolio's returns?", "BAL",
     {"fund_contribution"}, ("contribut", "driver", "drove")),
    ("What was the return for each year?", "BAL",
     {"periodic_returns"}, ("2024",)),
    ("Is volatility increasing or decreasing right now?", "ADV+",
     {"volatility_trend"}, ("increas", "decreas", "flat")),
    ("How does the portfolio compare against the benchmark?", "BAL",
     {"benchmark_metrics"}, ("benchmark",)),
    ("What are the top 5 holdings?", "BAL",
     {"top_holdings"}, ("GOLD", "coverage", "holding")),
    ("How much has the portfolio drifted from its target weights?", "ADV",
     {"portfolio_drift"}, ("drift",)),
    ("Run the engine at a 4% volatility cap - what allocation results?", "BAL",
     {"optimize_portfolio"}, ("%",)),
]


def build_ctx(nav_path: str, yield_path: str) -> ToolContext:
    b = ingest.load_nav(nav_path)
    y = ingest.load_yields(yield_path)
    conn = history.connect(":memory:")
    history.init_db(conn)
    for c in config.PORTFOLIO_CODES:
        history.save_snapshot(conn, c, b.dates[0].isoformat(),
                              dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS[c])),
                              sigma_target=0.035, source="seed")
    return ToolContext(conn=conn, data=AppData(b.dates, b.fund_returns, y, b.isins))


def run_battery(ctx: ToolContext, run_id: int) -> tuple[int, list[list[str]]]:
    failures = 0
    tool_log: list[list[str]] = []
    for i, (q, sel, need_tools, need_tokens) in enumerate(BATTERY, 1):
        out = graph.chat(q, ctx, selected_portfolio=sel)
        ans = (out.get("answer") or "")
        tools_used = [t["tool"] for t in out.get("tool_results", [])]
        tool_log.append(tools_used)

        problems = []
        low = ans.lower()
        hit_markers = [m for m in REFUSAL_MARKERS if m in low]
        if hit_markers:
            problems.append(f"refusal markers {hit_markers}")
        if need_tools and not (set(tools_used) & need_tools):
            problems.append(f"expected one of {sorted(need_tools)}, got {tools_used}")
        if need_tokens and not any(tok.lower() in low for tok in need_tokens):
            problems.append(f"answer lacks all of {need_tokens}")
        if not ans.strip():
            problems.append("empty answer")

        status = "PASS" if not problems else "FAIL"
        if problems:
            failures += 1
        print(f"[run{run_id} {i:02d}] {status} tools={tools_used} :: {q[:58]}")
        for p in problems:
            print(f"          !! {p}")
            print(f"          answer: {ans[:200]!r}")
    return failures, tool_log


def main():
    nav, yld = sys.argv[1], sys.argv[2]
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ctx = build_ctx(nav, yld)
    total = 0
    logs = []
    for r in range(1, runs + 1):
        f, tl = run_battery(ctx, r)
        total += f
        logs.append(tl)
    if runs > 1:
        drift = sum(1 for a, b in zip(logs[0], logs[-1]) if set(a) != set(b))
        print(f"\ntool-selection consistency: {len(BATTERY)-drift}/{len(BATTERY)} "
              f"identical across runs")
    print(f"\nTOTAL FAILURES: {total} across {runs} run(s) of {len(BATTERY)} questions")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
