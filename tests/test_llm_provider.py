"""LLM provider abstraction — pins the API differences verified against the live
remote gateway, so a backend swap can never silently break grounding.

Probed facts these tests encode (do not "simplify" them away):
  * gateway returns tool-call `arguments` as a JSON STRING; Ollama returns a dict
  * gateway tool results need `tool_call_id`; Ollama uses `tool_name`
  * gateway exposes `reasoning` (empty in practice); Ollama uses `thinking`
"""
from __future__ import annotations

import json

import pytest

import config
from agent import llm
from agent.llm import GatewayProvider, OllamaProvider, ToolCall, _parse_args, build_provider


# --- argument parsing (the #1 swap hazard) ----------------------------------
def test_parse_args_accepts_dict_from_ollama():
    assert _parse_args({"portfolio": "BAL"}) == {"portfolio": "BAL"}


def test_parse_args_parses_json_string_from_gateway():
    assert _parse_args('{"portfolio": "BAL"}') == {"portfolio": "BAL"}


def test_parse_args_never_raises_on_garbage():
    """Bad args must reach pydantic as a dict so the model gets a typed error to
    self-correct from — never crash the turn."""
    out = _parse_args("not json at all")
    assert isinstance(out, dict) and "_unparseable_arguments" in out


def test_parse_args_handles_empty():
    assert _parse_args("") == {}
    assert _parse_args(None) == {}


# --- message contracts differ per backend -----------------------------------
def test_ollama_tool_result_uses_tool_name():
    p = OllamaProvider()
    msg = p.tool_result_message(ToolCall(id="x", name="attribution", arguments={}), "{}")
    assert msg["role"] == "tool" and msg["tool_name"] == "attribution"
    assert "tool_call_id" not in msg


def test_gateway_tool_result_uses_tool_call_id():
    p = GatewayProvider(base="https://x", bearer="b", model="local-main")
    call = ToolCall(id="chatcmpl-tool-abc", name="attribution", arguments={})
    msg = p.tool_result_message(call, '{"sharpe": 1.32}')
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "chatcmpl-tool-abc"   # OpenAI contract
    assert "tool_name" not in msg


def test_gateway_echoes_native_tool_calls_in_assistant_message():
    """The gateway requires the assistant turn to carry the original tool_calls
    (with ids) before the tool results — echo the native form verbatim."""
    p = GatewayProvider(base="https://x", bearer="b")
    native = [{"id": "chatcmpl-tool-1", "type": "function",
               "function": {"name": "attribution", "arguments": '{"portfolio":"BAL"}'}}]
    reply = llm.LLMReply(content="", raw_tool_calls=native,
                         tool_calls=[ToolCall("chatcmpl-tool-1", "attribution",
                                              {"portfolio": "BAL"})])
    msg = p.assistant_message(reply)
    assert msg["tool_calls"] == native


# --- gateway response parsing ------------------------------------------------
class _FakeResp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


def test_gateway_chat_parses_tool_call_and_reasoning(monkeypatch):
    payload = {"choices": [{"message": {
        "content": None,
        "reasoning": "some reasoning",
        "tool_calls": [{"id": "chatcmpl-tool-9", "type": "function",
                        "function": {"name": "attribution",
                                     "arguments": '{"portfolio": "ADV+"}'}}],
    }}]}
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(payload))
    p = GatewayProvider(base="https://x", bearer="b", model="local-main")
    reply = p.chat([{"role": "user", "content": "hi"}], [])
    assert len(reply.tool_calls) == 1
    c = reply.tool_calls[0]
    assert c.name == "attribution"
    assert c.arguments == {"portfolio": "ADV+"}      # parsed from the JSON string
    assert c.id == "chatcmpl-tool-9"
    assert reply.thinking == "some reasoning"


def test_gateway_chat_handles_plain_answer(monkeypatch):
    payload = {"choices": [{"message": {"content": "BAL Sharpe is 1.3209.",
                                        "tool_calls": None}}]}
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(payload))
    p = GatewayProvider(base="https://x", bearer="b")
    reply = p.chat([], [])
    assert reply.tool_calls == []
    assert "1.3209" in reply.content


def test_gateway_sends_bearer_and_model(monkeypatch):
    seen = {}

    def _post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, payload=json, headers=headers)
        return _FakeResp({"choices": [{"message": {"content": "ok"}}]})

    import httpx
    monkeypatch.setattr(httpx, "post", _post)
    GatewayProvider(base="https://gw", bearer="secret-token",
                    model="local-main").chat([{"role": "user", "content": "x"}], [])
    assert seen["url"] == "https://gw/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer secret-token"
    assert seen["payload"]["model"] == "local-main"
    assert seen["payload"]["temperature"] == config.LLM_TEMPERATURE


# --- health ------------------------------------------------------------------
def test_gateway_health_without_bearer_is_explicit():
    ok, msg = GatewayProvider(base="https://x", bearer="").health()
    assert ok is False and "bearer" in msg.lower()


def test_gateway_health_flags_missing_model(monkeypatch):
    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"data": [{"id": "local-main"}]}
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _R())
    ok, msg = GatewayProvider(base="https://x", bearer="b", model="not-served").health()
    assert ok is False and "not offered" in msg


def test_gateway_health_ok(monkeypatch):
    class _R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"data": [{"id": "local-main"}]}
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _R())
    ok, msg = GatewayProvider(base="https://x", bearer="b", model="local-main").health()
    assert ok is True and "local-main" in msg


def test_gateway_health_401_is_actionable(monkeypatch):
    class _R:
        status_code = 401
        def raise_for_status(self): raise AssertionError("should short-circuit on 401")
        def json(self): return {}
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _R())
    ok, msg = GatewayProvider(base="https://x", bearer="bad").health()
    assert ok is False and "401" in msg


# --- factory -----------------------------------------------------------------
def test_build_provider_defaults_local(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    assert isinstance(build_provider(), OllamaProvider)


def test_build_provider_gateway(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gateway")
    assert isinstance(build_provider(), GatewayProvider)


def test_build_provider_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_provider("openai-cloud")
