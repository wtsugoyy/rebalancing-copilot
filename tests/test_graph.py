"""LangGraph harness tests.

The load-bearing guarantee: the model may only SELECT tools; every number comes from
deterministic engine code. These tests use a fake Ollama client so they run offline,
fast, and without a GPU — they assert the wiring, not the model's prose.
"""
from __future__ import annotations

import types

import pytest

import config
from agent import graph
from agent.memory import LocalBackend, Memory


from agent.llm import LLMReply, ToolCall


def _FakeToolCall(name, args):
    """Backend-neutral tool call (args already parsed, as every provider guarantees)."""
    return ToolCall(id=f"call_{name}", name=name, arguments=args)


def _FakeMsg(content="", tool_calls=None, thinking=""):
    return LLMReply(content=content, thinking=thinking,
                    tool_calls=list(tool_calls or []),
                    raw_tool_calls=list(tool_calls or []))


class _FakeProvider:
    """Scripted LLM provider. Turn 1: request a tool. Turn 2: answer from the result.

    Tests inject this via graph.chat(provider=...) — the same seam the real Ollama and
    gateway providers plug into, so these tests exercise the abstraction itself.
    """
    name = "fake"
    model = "fake-model"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []          # each entry mirrors the old kwargs shape

    def chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        return self.script.pop(0)

    def assistant_message(self, reply):
        msg = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}}
                                 for c in reply.tool_calls]
        return msg

    def tool_result_message(self, call, result_json):
        return {"role": "tool", "tool_name": call.name, "content": result_json}

    def health(self):
        return True, "fake ready"


@pytest.fixture
def mem(tmp_path):
    return Memory(local=LocalBackend(str(tmp_path / "m.db")), remote=None)


def _patch(monkeypatch, script):
    """Return a scripted provider; tests pass it through graph.chat(provider=...)."""
    fake = _FakeProvider(script)
    monkeypatch.setattr(graph, "build_provider", lambda *a, **k: fake)
    return fake


def test_model_calls_tool_and_grounds_answer(monkeypatch, golden, seeded_ctx, mem):
    fake = _patch(monkeypatch, [
        _FakeMsg(tool_calls=[_FakeToolCall("attribution", {"portfolio": "BAL"})],
                 thinking="I must call a tool."),
        _FakeMsg(content="BAL Sharpe is whatever the tool said."),
    ])
    out = graph.chat("What is BAL's Sharpe?", seeded_ctx, memory=mem)

    # a deterministic tool actually ran, and its result is real engine output —
    # compare against the fixture rather than a literal, so regenerating the
    # fixtures does not silently strand this assertion on a stale number.
    assert [t["tool"] for t in out["tool_results"]] == ["attribution"]
    result = out["tool_results"][0]["result"]
    expected_sharpe = golden["portfolios"]["BAL"]["expected"]["sharpe"]
    assert abs(result["sharpe"] - expected_sharpe) < 1e-3
    assert "summary" in result                          # labelled, un-mislabellable
    assert out["thinking"]                              # reasoning captured


def test_tool_result_is_fed_back_to_model(monkeypatch, seeded_ctx, mem):
    fake = _patch(monkeypatch, [
        _FakeMsg(tool_calls=[_FakeToolCall("attribution", {"portfolio": "ADV"})]),
        _FakeMsg(content="done"),
    ])
    graph.chat("ADV metrics?", seeded_ctx, memory=mem)
    second_turn_msgs = fake.calls[1]["messages"]
    tool_msgs = [m for m in second_turn_msgs if m["role"] == "tool"]
    assert tool_msgs, "tool output must be fed back to the model"
    assert "sharpe" in tool_msgs[0]["content"]


def test_tool_iteration_is_capped(monkeypatch, seeded_ctx, mem):
    # model stubbornly keeps requesting tools; the graph must stop
    script = [_FakeMsg(tool_calls=[_FakeToolCall("attribution", {"portfolio": "BAL"})])
              for _ in range(config.MAX_TOOL_ITERS + 3)]
    _patch(monkeypatch, script)
    out = graph.chat("loop", seeded_ctx, memory=mem)
    assert len(out["tool_results"]) <= config.MAX_TOOL_ITERS


def test_bad_tool_args_return_structured_error_not_crash(monkeypatch, seeded_ctx, mem):
    _patch(monkeypatch, [
        _FakeMsg(tool_calls=[_FakeToolCall("attribution", {"portfolio": "NOT_A_PORTFOLIO"})]),
        _FakeMsg(content="I could not do that."),
    ])
    out = graph.chat("bogus", seeded_ctx, memory=mem)
    assert "error" in out["tool_results"][0]["result"]


def test_no_tool_call_means_no_fabricated_evidence(monkeypatch, seeded_ctx, mem):
    _patch(monkeypatch, [_FakeMsg(content="Hello, I am the copilot.")])
    out = graph.chat("hi", seeded_ctx, memory=mem)
    assert out["tool_results"] == []
    assert out["answer"].startswith("Hello")


def test_memory_recall_is_injected_into_system_prompt(monkeypatch, seeded_ctx, mem):
    mem.remember("William prefers Sharpe over total return.")
    fake = _patch(monkeypatch, [_FakeMsg(content="noted")])
    graph.chat("which metric does William prefer", seeded_ctx, memory=mem)
    system = fake.calls[0]["messages"][0]["content"]
    assert "Remembered context" in system
    assert "Sharpe over total return" in system


def test_system_prompt_forbids_computation(monkeypatch, seeded_ctx, mem):
    fake = _patch(monkeypatch, [_FakeMsg(content="ok")])
    graph.chat("hi", seeded_ctx, memory=mem)
    system = fake.calls[0]["messages"][0]["content"]
    assert "NEVER compute" in system
    assert "must come from a tool result" in system


def test_agent_survives_dead_memory_and_tracing(monkeypatch, seeded_ctx):
    class _BoomMemory:
        def recall(self, *_a, **_kw): raise RuntimeError("memory down")
        def remember(self, *_a, **_kw): raise RuntimeError("memory down")

    _patch(monkeypatch, [_FakeMsg(content="still works")])
    out = graph.chat("hi", seeded_ctx, memory=_BoomMemory())
    assert out["answer"] == "still works"   # observability/memory never break the product
