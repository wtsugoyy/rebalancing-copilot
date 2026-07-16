-- Rebalancing Copilot history store. Portable SQL (standard types, no SQLite-only
-- functions) so a Phase 3 migration to the shared PostgreSQL instance is a port,
-- not a redesign. Append-only: no UPDATE/DELETE path is exposed.

CREATE TABLE IF NOT EXISTS snapshots (
    id             INTEGER PRIMARY KEY,
    portfolio      TEXT    NOT NULL,
    effective_date TEXT    NOT NULL,          -- ISO date the allocation went/would go live
    sigma_target   REAL,                      -- as entered by user (nullable for seeds)
    trading_days   INTEGER,
    source         TEXT    NOT NULL,          -- 'user_committed' | 'seed' | 'candidate_preview'
    note           TEXT,
    created_at     TEXT    NOT NULL           -- append-only ordering key (ISO timestamp)
);

CREATE TABLE IF NOT EXISTS weights (
    snapshot_id INTEGER NOT NULL,
    isin        TEXT    NOT NULL,
    weight      REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    snapshot_id   INTEGER NOT NULL,
    total_return  REAL,
    ann_return    REAL,
    ann_vol       REAL,
    max_drawdown  REAL,
    sharpe        REAL,
    sortino       REAL,
    avg_rf        REAL
);

CREATE INDEX IF NOT EXISTS idx_snap_portfolio ON snapshots (portfolio, effective_date, created_at);
CREATE INDEX IF NOT EXISTS idx_weights_snap   ON weights (snapshot_id);
CREATE INDEX IF NOT EXISTS idx_metrics_snap   ON metrics (snapshot_id);

-- Active benchmark settings (append-only; the latest row is the active benchmark).
CREATE TABLE IF NOT EXISTS benchmarks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    periods_json TEXT NOT NULL,          -- {"annualized": 5.1, "1m": 0.42, ...} in percent
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
