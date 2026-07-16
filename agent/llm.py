"""LLM transport abstraction — the ONLY place that knows which backend serves the model.

Two providers, one interface, so the agent graph is backend-agnostic:

* `OllamaProvider`  — local native Ollama (offline; GPU on this machine).
* `GatewayProvider` — a remote OpenAI-compatible LLM gateway, an
  OpenAI-compatible endpoint at `{gateway_base}/v1` authed with a **tenant bearer**.

Everything above this module (tool schemas, the 22 deterministic tools, the engine, the
context injection) is unchanged by the switch. The invariant still holds: the model may
only *select* a tool; every number comes from deterministic Python.

Verified differences between the two backends (probed against the live gateway, not
assumed — see docs/PROJECT_TRANSCRIPT.md):

| | Ollama | Gateway (OpenAI-compatible) |
|---|---|---|
| tool-call arguments | `dict` | **JSON string** -> must parse |
| tool result message | `{role:tool, tool_name}` | `{role:tool, tool_call_id}` |
| reasoning trace | `message.thinking` (populated) | `message.reasoning` (**empty in practice**) |
| auth | none | `Authorization: Bearer <tenant bearer>` |

Never log or echo the bearer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import config
from engine.obs import get_logger

_log = get_logger()


@dataclass
class ToolCall:
    """Backend-neutral tool call. `arguments` is ALWAYS a parsed dict here."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMReply:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_tool_calls: Any = None   # backend-native form, for echoing back in the transcript


class LLMProvider(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply: ...
    def tool_result_message(self, call: ToolCall, result_json: str) -> dict: ...
    def assistant_message(self, reply: LLMReply) -> dict: ...
    def health(self) -> tuple[bool, str]: ...


def _parse_args(raw: Any) -> dict:
    """Tool arguments arrive as a dict (Ollama) or a JSON string (OpenAI/gateway)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
            return parsed if isinstance(parsed, dict) else {"_value": parsed}
        except json.JSONDecodeError:
            # Never crash the turn: hand it to pydantic, which returns a typed error the
            # model can self-correct from.
            return {"_unparseable_arguments": raw}
    return {}


# ----------------------------------------------------------------- local ollama
class OllamaProvider:
    name = "ollama"

    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = host or config.OLLAMA_HOST
        self.model = model or config.OLLAMA_MODEL

    def _client(self):
        import ollama
        return ollama.Client(host=self.host)

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        r = self._client().chat(model=self.model, messages=messages, tools=tools,
                                think=config.LLM_THINK,
                                options={"temperature": config.LLM_TEMPERATURE})
        m = r.message
        calls = [ToolCall(id=getattr(tc, "id", "") or f"call_{i}",
                          name=tc.function.name,
                          arguments=_parse_args(tc.function.arguments))
                 for i, tc in enumerate(m.tool_calls or [])]
        return LLMReply(content=m.content or "", thinking=m.thinking or "",
                        tool_calls=calls, raw_tool_calls=m.tool_calls)

    def assistant_message(self, reply: LLMReply) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}}
                                 for c in reply.tool_calls]
        return msg

    def tool_result_message(self, call: ToolCall, result_json: str) -> dict:
        return {"role": "tool", "tool_name": call.name, "content": result_json}

    def health(self) -> tuple[bool, str]:
        try:
            models = [m.get("model", m.get("name", ""))
                      for m in self._client().list().get("models", [])]
            if not any(str(m).split(":")[0] == self.model.split(":")[0] for m in models):
                return False, (f"Ollama is up but model '{self.model}' is not pulled. "
                               f"Run: ollama pull {self.model}")
            return True, f"Local Ollama ready ({self.model})."
        except Exception as exc:  # noqa: BLE001
            return False, f"Local Ollama unreachable at {self.host}: {exc}"


# --------------------------------------------------- remote gateway (OpenAI API)
class GatewayProvider:
    """Remote OpenAI-compatible gateway.

    NOTE: using this provider sends prompts + tool results (which contain fund return
    data) to the remote gateway. That is a deliberate, user-approved departure from
    the project's offline-only default. See config.LLM_PROVIDER.
    """
    name = "gateway"

    def __init__(self, base: str | None = None, bearer: str | None = None,
                 model: str | None = None, timeout: float | None = None):
        self.base = (base or config.GATEWAY_BASE).rstrip("/")
        self.bearer = bearer or config.GATEWAY_BEARER
        self.model = model or config.GATEWAY_MODEL
        self.timeout = timeout or config.GATEWAY_TIMEOUT_S

    def _post(self, path: str, payload: dict) -> dict:
        import httpx
        r = httpx.post(f"{self.base}{path}", json=payload,
                       headers={"Authorization": f"Bearer {self.bearer}",
                                "Content-Type": "application/json"},
                       timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        payload = {"model": self.model, "messages": messages, "tools": tools,
                   "temperature": config.LLM_TEMPERATURE}
        d = self._post("/v1/chat/completions", payload)
        m = d["choices"][0]["message"]
        calls = [ToolCall(id=tc.get("id") or f"call_{i}",
                          name=tc["function"]["name"],
                          arguments=_parse_args(tc["function"].get("arguments")))
                 for i, tc in enumerate(m.get("tool_calls") or [])]
        # The gateway exposes a `reasoning` field, but it comes back empty in practice
        # (verified against the live endpoint). Kept so the UI lights up automatically
        # if the backend ever starts populating it.
        return LLMReply(content=m.get("content") or "",
                        thinking=(m.get("reasoning") or ""),
                        tool_calls=calls, raw_tool_calls=m.get("tool_calls"))

    def assistant_message(self, reply: LLMReply) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
        if reply.raw_tool_calls:
            msg["tool_calls"] = reply.raw_tool_calls   # echo native form incl. ids
        return msg

    def tool_result_message(self, call: ToolCall, result_json: str) -> dict:
        return {"role": "tool", "tool_call_id": call.id, "content": result_json}

    def health(self) -> tuple[bool, str]:
        import httpx
        if not self.bearer:
            return False, ("Gateway bearer not set. Put the tenant bearer in "
                           "GATEWAY_BEARER (.env).")
        try:
            r = httpx.get(f"{self.base}/v1/models",
                          headers={"Authorization": f"Bearer {self.bearer}"},
                          timeout=15.0)
            if r.status_code == 401:
                return False, "Gateway rejected the bearer (401). Ask the operator to re-provision."
            if r.status_code == 403:
                return False, "Gateway forbade the tenant (403). Check project_key/bearer."
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", [])]
            if self.model not in ids:
                return False, (f"Gateway is up but model '{self.model}' is not offered to "
                               f"this tenant. Available: {ids}")
            return True, f"Remote gateway ready ({self.model} @ {self.base})."
        except Exception as exc:  # noqa: BLE001
            return False, f"Gateway unreachable at {self.base}: {exc}"


def build_provider(name: str | None = None) -> LLMProvider:
    """Factory. `LLM_PROVIDER=gateway` routes to the remote model; default local."""
    choice = (name or config.LLM_PROVIDER or "ollama").strip().lower()
    if choice == "gateway":
        return GatewayProvider()
    if choice == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown LLM_PROVIDER {choice!r}; expected 'ollama' or 'gateway'.")
