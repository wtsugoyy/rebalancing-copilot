"""Agent memory — remembers analyst preferences/context across turns and sessions.

Two backends behind one interface:

* `SupermemoryBackend` — self-hosted Supermemory (local embeddings, encrypted local
  storage, no data egress). Writes work; **recall is broken in supermemory-server
  0.0.3**: documents reach `status=done` and the memory agent extracts memories, but
  `/v3/search` and `/v4/search` always return `total: 0`. Verified with and without
  containerTag and with threshold 0. We therefore still WRITE to it (so the demo is
  genuinely wired to Supermemory) but do not rely on it for recall.
* `LocalBackend` — SQLite FTS5 keyword recall. Deterministic, offline, zero deps.
  This is what actually answers recall today.

`Memory` dual-writes and reads from whichever backend returns results, preferring
Supermemory if it ever starts returning hits (i.e. after an upstream fix/upgrade).

Nothing here ever touches the numeric engine. Memory stores *context*, never figures.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass

import httpx

import config
from engine.obs import get_logger

_log = get_logger()


@dataclass(frozen=True)
class MemoryHit:
    text: str
    source: str


# --------------------------------------------------------------------------- local
_LOCAL_DDL = """
CREATE TABLE IF NOT EXISTS memories(
  id INTEGER PRIMARY KEY,
  container TEXT NOT NULL,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
  USING fts5(text, content='memories', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


class LocalBackend:
    """SQLite FTS5 recall. Always available; the reliable path."""

    name = "local"

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_LOCAL_DDL)
        self._conn.commit()

    def add(self, text: str, container: str) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO memories(container, text) VALUES (?,?)",
                               (container, text))
            self._conn.commit()

    def search(self, query: str, container: str, limit: int = 5) -> list[MemoryHit]:
        # FTS5 needs a sanitised query: keep word chars, OR the terms together.
        terms = [t for t in re.findall(r"[A-Za-z0-9+']{3,}", query)]
        if not terms:
            return []
        expr = " OR ".join(terms)
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT m.text FROM memories_fts f JOIN memories m ON m.id = f.rowid "
                    "WHERE memories_fts MATCH ? AND m.container = ? "
                    "ORDER BY rank LIMIT ?", (expr, container, limit)).fetchall()
            except sqlite3.OperationalError:
                return []
        return [MemoryHit(text=r[0], source="local") for r in rows]


# ---------------------------------------------------------------------- supermemory
class SupermemoryBackend:
    """Self-hosted Supermemory. Write path verified; recall broken upstream (0.0.3)."""

    name = "supermemory"

    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def available(self) -> bool:
        try:
            r = httpx.post(f"{self.base_url}/v3/search", headers=self._headers(),
                           json={"q": "ping"}, timeout=self.timeout)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def add(self, text: str, container: str) -> None:
        try:
            httpx.post(f"{self.base_url}/v3/documents", headers=self._headers(),
                       json={"content": text, "containerTag": container},
                       timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            _log.warning("supermemory add failed (non-fatal): %s", exc)

    def search(self, query: str, container: str, limit: int = 5) -> list[MemoryHit]:
        try:
            r = httpx.post(f"{self.base_url}/v3/search", headers=self._headers(),
                           json={"q": query, "containerTag": container, "limit": limit},
                           timeout=self.timeout)
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            _log.warning("supermemory search failed (non-fatal): %s", exc)
            return []
        hits = []
        for item in data.get("results", []):
            txt = item.get("memory") or ""
            if not txt:
                chunks = item.get("chunks") or []
                txt = chunks[0].get("content", "") if chunks else ""
            if txt:
                hits.append(MemoryHit(text=txt, source="supermemory"))
        return hits


# --------------------------------------------------------------------------- facade
class Memory:
    def __init__(self, local: LocalBackend, remote: SupermemoryBackend | None = None,
                 container: str = "copilot"):
        self.local = local
        self.remote = remote
        self.container = container

    def remember(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.local.add(text, self.container)
        if self.remote:
            self.remote.add(text, self.container)  # dual-write; failures are non-fatal

    def recall(self, query: str, limit: int = 3) -> list[MemoryHit]:
        if self.remote:
            hits = self.remote.search(query, self.container, limit)
            if hits:                       # prefer supermemory once it works upstream
                return hits
        return self.local.search(query, self.container, limit)


def build_memory() -> Memory:
    local = LocalBackend(config.MEMORY_DB_PATH)
    remote = None
    if config.SUPERMEMORY_URL:
        remote = SupermemoryBackend(config.SUPERMEMORY_URL, config.SUPERMEMORY_API_KEY)
    return Memory(local=local, remote=remote)
