"""Append-only snapshot history (SQLite, WAL mode).

Single-writer by design: all writes go through save_snapshot() in one transaction. Deterministic current/previous
ordering. No UPDATE/DELETE path — corrections are new rows.
Pure data-access; no UI imports.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import config
from engine.validate import NotFoundError

_SCHEMA = Path(__file__).with_name("schema.sql")

# Serialize all writes: SQLite has a single global write lock, and
# Streamlit reruns can touch a cached connection from different pool threads.
_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Snapshot:
    id: int
    portfolio: str
    effective_date: str
    source: str
    created_at: str
    weights: dict[str, float]
    metrics: dict | None
    sigma_target: float | None = None
    trading_days: int | None = None
    note: str | None = None


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or config.DB_PATH
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: Streamlit reruns access the cached connection from a
    # thread pool. Safe here because writes are serialized via _WRITE_LOCK + single
    # atomic transactions, and WAL lets readers not block.
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA.read_text(encoding="utf-8"))
    conn.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_snapshot(
    conn: sqlite3.Connection,
    portfolio: str,
    effective_date: str | date,
    weights: dict[str, float],
    *,
    source: str = "user_committed",
    metrics: dict | None = None,
    sigma_target: float | None = None,
    trading_days: int | None = None,
    note: str | None = None,
) -> int:
    """Atomically insert a snapshot + its weights + optional metrics. Returns id."""
    eff = effective_date.isoformat() if isinstance(effective_date, date) else str(effective_date)
    with _WRITE_LOCK, conn:  # serialized + single atomic transaction
        cur = conn.execute(
            "INSERT INTO snapshots (portfolio, effective_date, sigma_target, trading_days, "
            "source, note, created_at) VALUES (?,?,?,?,?,?,?)",
            (portfolio, eff, sigma_target, trading_days, source, note, _now_iso()),
        )
        sid = cur.lastrowid
        conn.executemany(
            "INSERT INTO weights (snapshot_id, isin, weight) VALUES (?,?,?)",
            [(sid, isin, float(w)) for isin, w in weights.items()],
        )
        if metrics is not None:
            conn.execute(
                "INSERT INTO metrics (snapshot_id, total_return, ann_return, ann_vol, "
                "max_drawdown, sharpe, sortino, avg_rf) VALUES (?,?,?,?,?,?,?,?)",
                (sid, metrics.get("total_return"), metrics.get("ann_return"),
                 metrics.get("ann_vol"), metrics.get("max_drawdown"),
                 metrics.get("sharpe"), metrics.get("sortino"), metrics.get("avg_rf")),
            )
    return sid


def _hydrate(conn: sqlite3.Connection, row: sqlite3.Row) -> Snapshot:
    weights = {r["isin"]: r["weight"] for r in conn.execute(
        "SELECT isin, weight FROM weights WHERE snapshot_id=?", (row["id"],))}
    mrow = conn.execute(
        "SELECT total_return, ann_return, ann_vol, max_drawdown, sharpe, sortino, avg_rf "
        "FROM metrics WHERE snapshot_id=?", (row["id"],)).fetchone()
    metrics = dict(mrow) if mrow else None
    return Snapshot(
        id=row["id"], portfolio=row["portfolio"], effective_date=row["effective_date"],
        source=row["source"], created_at=row["created_at"], weights=weights,
        metrics=metrics, sigma_target=row["sigma_target"],
        trading_days=row["trading_days"], note=row["note"],
    )


# committed rows (exclude candidate previews) ordered newest-first
_ORDER = "ORDER BY date(effective_date) DESC, created_at DESC"
# Sources that count as a live/committed allocation. 'engine' = committed from the
# Rebalancing Engine tab. 'candidate_preview' is deliberately excluded — previews
# must never become "current".
_COMMITTED = "source IN ('user_committed','seed','engine')"


def get_current(conn: sqlite3.Connection, portfolio: str) -> Snapshot:
    row = conn.execute(
        f"SELECT * FROM snapshots WHERE portfolio=? AND {_COMMITTED} {_ORDER} LIMIT 1",
        (portfolio,)).fetchone()
    if row is None:
        raise NotFoundError(f"No committed snapshot for portfolio '{portfolio}'.",
                            portfolio=portfolio)
    return _hydrate(conn, row)


def get_previous(conn: sqlite3.Connection, portfolio: str) -> Snapshot:
    rows = conn.execute(
        f"SELECT * FROM snapshots WHERE portfolio=? AND {_COMMITTED} {_ORDER} LIMIT 2",
        (portfolio,)).fetchall()
    if len(rows) < 2:
        raise NotFoundError(f"No previous snapshot for portfolio '{portfolio}'.",
                            portfolio=portfolio)
    return _hydrate(conn, rows[1])


def list_history(conn: sqlite3.Connection, portfolio: str, limit: int = 20) -> list[Snapshot]:
    rows = conn.execute(
        f"SELECT * FROM snapshots WHERE portfolio=? AND {_COMMITTED} {_ORDER} LIMIT ?",
        (portfolio, limit)).fetchall()
    return [_hydrate(conn, r) for r in rows]


def chained_timeline(conn: sqlite3.Connection, portfolio: str) -> list[tuple[date, dict[str, float]]]:
    """Ordered (effective_date, weights) segments for perf.portfolio_returns."""
    rows = conn.execute(
        f"SELECT * FROM snapshots WHERE portfolio=? AND {_COMMITTED} "
        "ORDER BY date(effective_date) ASC, created_at ASC", (portfolio,)).fetchall()
    out = []
    for r in rows:
        snap = _hydrate(conn, r)
        out.append((date.fromisoformat(snap.effective_date), snap.weights))
    return out
