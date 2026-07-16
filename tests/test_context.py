"""Regression tests for the context-blindness failure (management screenshot):
the model refused 'best/worst month in 2026' claiming 2026 was in the future, and
asked for a portfolio code the dashboard already knew. These pin the fixes."""
from __future__ import annotations

import pytest

import config
from agent import graph, tools
from agent.memory import LocalBackend, Memory


from agent.llm import LLMReply


class _FakeProvider:
    """Records the messages the graph builds, so we can assert on the system prompt."""
    name = "fake"
    model = "fake-model"

    def __init__(self): self.calls = []

    def chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        return LLMReply(content="ok")

    def assistant_message(self, reply): return {"role": "assistant", "content": reply.content}
    def tool_result_message(self, call, result_json):
        return {"role": "tool", "tool_name": call.name, "content": result_json}
    def health(self): return True, "fake ready"


@pytest.fixture
def mem(tmp_path):
    return Memory(local=LocalBackend(str(tmp_path / "m.db")), remote=None)


def _system_prompt(monkeypatch, seeded_ctx, mem, question="hello", **chat_kw) -> str:
    fake = _FakeProvider()
    monkeypatch.setattr(graph, "build_provider", lambda *a, **k: fake)
    graph.chat(question, seeded_ctx, memory=mem, **chat_kw)
    return fake.calls[0]["messages"][0]["content"]


def test_data_window_and_years_injected(monkeypatch, seeded_ctx, mem):
    sys = _system_prompt(monkeypatch, seeded_ctx, mem)
    d0, d1 = seeded_ctx.data.dates[0], seeded_ctx.data.dates[-1]
    assert d0.isoformat() in sys and d1.isoformat() in sys
    years = sorted({d.year for d in seeded_ctx.data.dates})
    for y in years:
        assert str(y) in sys
    assert "NOT in the future" in sys


def test_selected_portfolio_injected(monkeypatch, seeded_ctx, mem):
    sys = _system_prompt(monkeypatch, seeded_ctx, mem, selected_portfolio="SC")
    assert "currently selected in the dashboard: SC" in sys
    assert "Secure Cash" in sys


def test_copilot_identity_present(monkeypatch, seeded_ctx, mem):
    """The model must be told what it is looking at. Without the dataset preamble it
    falls back on its training prior and refuses questions the data can answer."""
    sys = _system_prompt(monkeypatch, seeded_ctx, mem)
    assert "ABOUT THE DATASET" in sys
    assert "12 unit-trust funds" in sys
    assert "risk ladder" in sys
    assert "NEVER refuse a question about a period that lies" in sys


# --- prompt ORDERING is load-bearing for tool calling --------------------------
# Measured on the remote gateway (`local-main`, 22 tools, same questions):
#   rules -> context  =>  0/6 native tool calls (answers become ungrounded prose JSON)
#   context -> rules  =>  6/6
# The "call a tool" rule must sit closest to the user turn. Never append after it.
def test_rules_come_last_in_system_prompt(monkeypatch, seeded_ctx, mem):
    sys = _system_prompt(monkeypatch, seeded_ctx, mem, selected_portfolio="BAL")
    assert sys.index("DATASET CONTEXT") < sys.index("RULES (absolute)"), \
        "context must precede the rules or gateway tool-calling collapses"
    # pin the invariant, not the wording: SYSTEM_PROMPT must be the tail of the prompt
    assert sys.rstrip().endswith(graph.SYSTEM_PROMPT.rstrip()), \
        "nothing may be appended after SYSTEM_PROMPT — the tool rules must come last"


# --- sandbox must not out-compete the dedicated tools -------------------------
# The gateway model (`local-main`) reached for run_analysis instead of the specific
# validated tools: it answered "return for each year" via the ADVISORY sandbox, and
# burned 4 sandbox calls on "volatility trend" before returning "...". Rule 9 routes it.
def test_system_prompt_routes_away_from_the_sandbox(monkeypatch, seeded_ctx, mem):
    sys = _system_prompt(monkeypatch, seeded_ctx, mem)
    assert "TOOL CHOICE" in sys
    assert "periodic_returns (NOT run_analysis)" in sys
    assert "volatility_trend (NOT run_analysis)" in sys
    assert "ONLY if no tool above can answer" in sys


def test_sandbox_schema_is_explicitly_last_resort():
    from agent.agent import TOOL_SCHEMAS
    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    # ordering is load-bearing: the sandbox must sit last in the tool array
    assert names[-1] == "run_analysis"
    desc = [t for t in TOOL_SCHEMAS
            if t["function"]["name"] == "run_analysis"][0]["function"]["description"]
    assert "LAST RESORT ONLY" in desc
    assert "DO NOT use it for" in desc
    for tool in ("periodic_returns", "volatility_trend", "monthly_stats",
                 "fund_contribution", "portfolio_drift", "benchmark_metrics"):
        assert tool in desc, f"{tool} must be named as a preferred alternative"


def test_memories_also_precede_the_rules(monkeypatch, seeded_ctx, mem):
    mem.remember("William prefers Sharpe over total return.")
    # ask something the FTS index will actually match, so a memory is recalled
    sys = _system_prompt(monkeypatch, seeded_ctx, mem,
                         question="which metric does William prefer comparing portfolios")
    assert "Remembered context" in sys
    assert sys.index("Remembered context") < sys.index("RULES (absolute)")
    assert sys.rstrip().endswith(graph.SYSTEM_PROMPT.rstrip())


def test_no_data_context_degrades_cleanly(monkeypatch, seeded_ctx, mem):
    from agent.tools import ToolContext
    empty = ToolContext(conn=seeded_ctx.conn, data=None)
    fake = _FakeProvider()
    monkeypatch.setattr(graph, "build_provider", lambda *a, **k: fake)
    graph.chat("hello", empty, memory=mem)
    sys = fake.calls[0]["messages"][0]["content"]
    assert "No price data loaded yet" in sys


# --- year-scoped monthly stats (the exact failing ask) --------------------------
def test_monthly_stats_scoped_to_2026(seeded_ctx):
    out = tools.execute_tool("monthly_stats", {"portfolio": "SC", "year": 2026}, seeded_ctx)
    assert "error" not in out, out
    assert out["best_month"]["month"].startswith("2026")
    assert out["worst_month"]["month"].startswith("2026")
    assert "2026" in out["summary"]


def test_monthly_stats_unavailable_year_lists_options(seeded_ctx):
    out = tools.execute_tool("monthly_stats", {"portfolio": "SC", "year": 2019}, seeded_ctx)
    assert "error" in out
    assert "2023" in out["error"] or "Years available" in out["error"]


def test_monthly_stats_unscoped_still_whole_period(seeded_ctx):
    out = tools.execute_tool("monthly_stats", {"portfolio": "SC"}, seeded_ctx)
    assert "error" not in out
    assert out["n_months"] > 12
