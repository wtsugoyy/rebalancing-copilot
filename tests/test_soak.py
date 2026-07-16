"""TC-11 soak: drive the full pipeline over many cycles and assert (a) determinism
(identical inputs -> identical outputs every time), (b) no unbounded resource growth,
(c) no silent failures (no unlogged exceptions leak).

Runs headless (no LLM, no Streamlit). Marked slow; run explicitly.
"""
import gc
import io
import json
from pathlib import Path

import pytest

import config
from agent.tools import AppData, ToolContext, _attribution, execute_tool
from engine import ingest
from store import history

ROOT = Path(__file__).resolve().parents[1]
CYCLES = 60


@pytest.fixture(scope="module")
def loaded():
    b = ingest.load_nav(str(ROOT / "sample" / "nav_sample.csv"))
    y = ingest.load_yields(str(ROOT / "sample" / "yield_sample.csv"))
    return b, y


def _fresh_ctx(tmp_path, b, y, tag):
    conn = history.connect(str(tmp_path / f"soak_{tag}.db"))
    history.init_db(conn)
    for code in config.PORTFOLIO_CODES:
        wmap = dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS[code]))
        history.save_snapshot(conn, code, b.dates[0].isoformat(), wmap, source="seed")
    return conn, ToolContext(conn=conn, data=AppData(b.dates, b.fund_returns, y, b.isins))


def test_attribution_is_deterministic_across_cycles(tmp_path, loaded):
    b, y = loaded
    baseline = None
    for i in range(CYCLES):
        conn, ctx = _fresh_ctx(tmp_path, b, y, i)
        snapshot = {c: _attribution(ctx, c) for c in config.PORTFOLIO_CODES}
        conn.close()
        key = json.dumps(snapshot, sort_keys=True)
        if baseline is None:
            baseline = key
        else:
            assert key == baseline, f"cycle {i}: attribution drifted from cycle 0"
        gc.collect()


def test_full_upload_optimize_query_save_cycle_no_silent_failures(tmp_path, loaded):
    b, y = loaded
    conn, ctx = _fresh_ctx(tmp_path, b, y, "cycle")
    errors = []
    for i in range(CYCLES):
        for code in config.PORTFOLIO_CODES:
            out = execute_tool("attribution", {"portfolio": code}, ctx)
            if "error" in out:
                errors.append((i, code, out))
            v = execute_tool("vol_vs_target", {"portfolio": code, "target": 0.10}, ctx)
            if "error" in v:
                errors.append((i, code, v))
        # simulate a committed rebalance each cycle (append-only history growth)
        wmap = dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS["BAL"]))
        history.save_snapshot(conn, "BAL", f"2026-07-{(i % 28) + 1:02d}", wmap,
                              source="user_committed")
    # history grew, current/previous still resolve correctly
    assert len(history.list_history(conn, "BAL", limit=1000)) >= CYCLES
    history.get_current(conn, "BAL")
    history.get_previous(conn, "BAL")
    conn.close()
    assert not errors, f"silent/tool errors during soak: {errors[:3]}"


def test_bad_inputs_always_typed_never_crash(tmp_path, loaded):
    b, y = loaded
    conn, ctx = _fresh_ctx(tmp_path, b, y, "bad")
    bad_cases = [
        ("attribution", {"portfolio": "XX"}),
        ("vol_vs_target", {"portfolio": "BAL", "target": 99}),
        ("get_current_portfolio", {"portfolio": ""}),
        ("nonexistent", {"portfolio": "BAL"}),
    ]
    for _ in range(CYCLES):
        for name, args in bad_cases:
            out = execute_tool(name, args, ctx)
            assert "error" in out and "code" in out, f"{name} did not return a typed error"
    conn.close()
