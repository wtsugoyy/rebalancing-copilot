"""Langfuse tracing (self-hosted, telemetry disabled, nothing leaves the host).

Observability must never be able to break the product. Every function here degrades
to a no-op if Langfuse is unavailable, misconfigured, or the SDK isn't installed.
"""
from __future__ import annotations

from contextlib import contextmanager

import config
from engine.obs import get_logger

_log = get_logger()
_client = None
_checked = False


def _get_client():
    global _client, _checked
    if _checked:
        return _client
    _checked = True
    if not config.LANGFUSE_ENABLED:
        return None
    try:
        from langfuse import Langfuse  # type: ignore
        _client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
        _log.info("langfuse tracing enabled host=%s", config.LANGFUSE_HOST)
    except Exception as exc:  # noqa: BLE001
        _log.warning("langfuse unavailable, tracing disabled (non-fatal): %s", exc)
        _client = None
    return _client


def health() -> tuple[bool, str]:
    c = _get_client()
    if c is None:
        return False, "Langfuse disabled or unavailable (tracing off)."
    try:
        c.auth_check()
        return True, f"Langfuse tracing active ({config.LANGFUSE_HOST})."
    except Exception as exc:  # noqa: BLE001
        return False, f"Langfuse unreachable: {exc}"


class _NullSpan:
    def update(self, **_kw): pass
    def end(self, **_kw): pass
    def score(self, **_kw): pass


@contextmanager
def trace(name: str, user_input: str | None = None, metadata: dict | None = None):
    """Trace one agent run. Yields an object with .span(name) and .update()."""
    client = _get_client()
    if client is None:
        yield _NullTrace()
        return
    t = None
    try:
        t = client.trace(name=name, input=user_input, metadata=metadata or {})
        yield _LangfuseTrace(t)
    except Exception as exc:  # noqa: BLE001
        _log.warning("langfuse trace failed (non-fatal): %s", exc)
        yield _NullTrace()
    finally:
        try:
            if t is not None:
                client.flush()
        except Exception:  # noqa: BLE001
            pass


class _NullTrace:
    def span(self, *_a, **_kw): return _NullSpan()
    def generation(self, *_a, **_kw): return _NullSpan()
    def update(self, **_kw): pass


class _LangfuseTrace:
    def __init__(self, t):
        self._t = t

    def span(self, name: str, **kw):
        try:
            return self._t.span(name=name, **kw)
        except Exception:  # noqa: BLE001
            return _NullSpan()

    def generation(self, name: str, **kw):
        try:
            return self._t.generation(name=name, **kw)
        except Exception:  # noqa: BLE001
            return _NullSpan()

    def update(self, **kw):
        try:
            self._t.update(**kw)
        except Exception:  # noqa: BLE001
            pass
