"""TC-7: bad LLM-supplied arguments never reach the engine; the tool-call loop
executes deterministic tools and never fabricates numbers."""
import json
from datetime import date

import pytest

import config
from agent import agent, tools
from agent.tools import AppData, ToolContext
from store import history


@pytest.fixture
def ctx(tmp_path, golden):
    conn = history.connect(str(tmp_path / "t.db"))
    history.init_db(conn)
    dates = golden["_dates"]
    # seed BAL current weights at the first date
    wmap = dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS["BAL"]))
    history.save_snapshot(conn, "BAL", dates[0].isoformat(), wmap, source="seed")
    data = AppData(dates=dates, fund_returns=golden["fund_returns"],
                   yields=golden["_yields"], isins=golden["isins"])
    yield ToolContext(conn=conn, data=data)
    conn.close()


def test_bad_portfolio_arg_returns_error_not_raise(ctx):
    out = tools.execute_tool("attribution", {"portfolio": "NOPE"}, ctx)
    assert "error" in out and out["code"] == "VALIDATION_ERROR"


def test_unknown_tool_returns_error(ctx):
    out = tools.execute_tool("delete_everything", {"portfolio": "BAL"}, ctx)
    assert out["code"] == "UNKNOWN_TOOL"


def test_attribution_tool_runs_and_returns_metrics(ctx):
    out = tools.execute_tool("attribution", {"portfolio": "BAL"}, ctx)
    assert "sharpe" in out and "ann_vol" in out and "error" not in out


def test_vol_vs_target_gap(ctx):
    out = tools.execute_tool("vol_vs_target", {"portfolio": "BAL", "target": 0.10}, ctx)
    assert "gap_pp" in out and out["error"] if "error" in out else True
    assert abs(out["gap_pp"] - (out["ann_vol"] - 0.10) * 100) < 1e-9


def test_out_of_range_target_rejected(ctx):
    out = tools.execute_tool("vol_vs_target", {"portfolio": "BAL", "target": 5.0}, ctx)
    assert out["code"] == "VALIDATION_ERROR"


def test_no_data_loaded_surfaces_typed_error(tmp_path, golden):
    conn = history.connect(str(tmp_path / "n.db"))
    history.init_db(conn)
    wmap = dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS["BAL"]))
    history.save_snapshot(conn, "BAL", "2023-07-03", wmap, source="seed")
    ctx = ToolContext(conn=conn, data=None)
    out = tools.execute_tool("attribution", {"portfolio": "BAL"}, ctx)
    assert out["code"] == "NOT_FOUND"
    conn.close()


class _FakeClient:
    """Mimics ollama.Client.chat: first turn asks for a tool, second turn answers."""
    def __init__(self):
        self.turn = 0

    def chat(self, model, messages, tools, options):
        self.turn += 1
        if self.turn == 1:
            return {"message": {"content": "", "tool_calls": [
                {"function": {"name": "attribution", "arguments": {"portfolio": "BAL"}}}]}}
        # verify the tool result was fed back
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert tool_msgs, "tool result was not appended before the final turn"
        payload = json.loads(tool_msgs[-1]["content"])
        return {"message": {"content": f"BAL Sharpe is {payload['sharpe']:.4f}.",
                            "tool_calls": None}}


def test_chat_loop_executes_tool_and_reports_real_number(ctx):
    reply, transcript = agent.chat("How did BAL do?", ctx, _client=_FakeClient())
    assert "Sharpe" in reply
    # the number in the reply must equal the deterministic tool output
    truth = tools.execute_tool("attribution", {"portfolio": "BAL"}, ctx)["sharpe"]
    assert f"{truth:.4f}" in reply


def test_chat_loop_caps_iterations():
    class _Loopy:
        def chat(self, model, messages, tools, options):
            return {"message": {"content": "", "tool_calls": [
                {"function": {"name": "list_portfolios", "arguments": {}}}]}}
    conn = history.connect(":memory:")
    history.init_db(conn)
    reply, _ = agent.chat("hi", ToolContext(conn=conn, data=None), _client=_Loopy())
    assert "couldn't complete" in reply.lower()
    conn.close()
