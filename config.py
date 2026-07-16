"""Central configuration — the determinism contract.

Every constant the engine depends on lives here; no magic numbers inline in engine
code. The 365-calendar / 252-trading annualisation asymmetry below is deliberate and
documented in CLAUDE.md — do NOT "fix" it into consistency without reading that first.
"""
from __future__ import annotations

# --- Fund universe: 12 synthetic funds, canonical order -----------------------
# These are INVENTED funds. The `ZZ` ISIN prefix is reserved for user-assigned
# codes and never issued to a real security, so nothing here can collide with a
# traded instrument. Demo and test data are generated from this universe by
# `scripts/make_synthetic_fixtures.py` — see README §Data.
FUND_UNIVERSE: list[str] = [
    "ZZ0000000001", "ZZ0000000002", "ZZ0000000003", "ZZ0000000004",
    "ZZ0000000005", "ZZ0000000006", "ZZ0000000007", "ZZ0000000008",
    "ZZ0000000009", "ZZ0000000010", "ZZ0000000011", "ZZ0000000012",
]

PORTFOLIO_CODES: list[str] = ["SC", "SC+", "BAL", "BAL+", "ADV", "ADV+"]

PORTFOLIO_NAMES: dict[str, str] = {
    "SC": "Secure Cash", "SC+": "Secure Cash+", "BAL": "Balanced",
    "BAL+": "Balanced+", "ADV": "Advanced", "ADV+": "Advanced+",
}

# Illustrative starting weights — used to seed the history store so there is a
# "current" portfolio to attribute/compare against on first run. They are made up
# for the demo: risk rises from all-cash (SC) to equity/commodity-heavy (ADV+).
# Not advice, not anyone's model portfolio. Aligned to FUND_UNIVERSE order.
DEFAULT_WEIGHTS: dict[str, list[float]] = {
    "SC":   [0.60, 0.40, 0.00, 0.00, 0.00, 0.00, 0, 0.00, 0.00, 0.00, 0.00, 0.00],
    "SC+":  [0.30, 0.20, 0.10, 0.25, 0.15, 0.00, 0, 0.00, 0.00, 0.00, 0.00, 0.00],
    "BAL":  [0.10, 0.10, 0.00, 0.20, 0.25, 0.15, 0, 0.10, 0.05, 0.05, 0.00, 0.00],
    "BAL+": [0.05, 0.05, 0.00, 0.15, 0.20, 0.15, 0, 0.15, 0.10, 0.10, 0.05, 0.00],
    "ADV":  [0.00, 0.00, 0.00, 0.10, 0.10, 0.10, 0, 0.15, 0.25, 0.15, 0.10, 0.05],
    "ADV+": [0.00, 0.00, 0.00, 0.05, 0.05, 0.05, 0, 0.10, 0.30, 0.20, 0.15, 0.10],
}

# Per-portfolio eligible funds = ISINs with a nonzero default weight.
PORTFOLIO_ELIGIBILITY: dict[str, list[str]] = {
    code: [isin for isin, w in zip(FUND_UNIVERSE, weights) if w > 0]
    for code, weights in DEFAULT_WEIGHTS.items()
}

# --- Return / annualisation conventions (see CLAUDE.md before changing) ---
RETURN_TYPE = "simple"          # r_t = NAV_t/NAV_{t-1} - 1
VOL_ANNUALISATION = 252         # hardcoded sqrt(252) for volatility
RETURN_ANNUALISATION = "calendar365"  # 365 / elapsed calendar days CAGR
STDEV_DDOF = 1                  # sample stdev (ddof=1)

# --- Risk-free ---
RF_FALLBACK = 0.033             # 3.3% p.a. fallback when a yield date is missing

# --- Data quality gates ---
MIN_OBS = 60                    # minimum aligned daily observations

# --- Weight validation ---
WEIGHT_SUM = 1.0
WEIGHT_SUM_TOL = 1e-4

# --- Local LLM (offline, Ollama) ---
import os

# --- LLM backend selection --------------------------------------------------
# "ollama"  = local native Ollama on this machine (fully offline; the default).
# "gateway" = a remote OpenAI-compatible LLM gateway.
#             NOTE: gateway mode SENDS prompts + tool results (fund return data) off this
#             machine to the remote endpoint. That is a deliberate, approved departure
#             from the offline-only default — do not switch it silently.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()

# qwen3:14b -- reasoning ("thinking") + native tool calling, verified on the RX 9060 XT
# via ROCm (100% GPU). DeepSeek-R1 was evaluated and rejected: official Ollama builds
# have no tool-calling template, so it invents numbers instead of calling the engine.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")

# --- Remote gateway (any OpenAI-compatible endpoint) ------------------------
# Tool calling on `local-main` was probed against the live endpoint and CONFIRMED, so
# the "LLM never computes" invariant survives the switch.
GATEWAY_BASE = os.environ.get("GATEWAY_BASE", "https://llm-gateway.example.com")
GATEWAY_MODEL = os.environ.get("GATEWAY_MODEL", "local-main")
GATEWAY_PROJECT_KEY = os.environ.get("GATEWAY_PROJECT_KEY", "default")
GATEWAY_TIMEOUT_S = float(os.environ.get("GATEWAY_TIMEOUT_S", "120"))
# Prefer the env var; fall back to a token file on disk. Scope the token as
# narrowly as your provider allows.
GATEWAY_BEARER = os.environ.get("GATEWAY_BEARER", "").strip()
if not GATEWAY_BEARER:
    _bearer_file = os.environ.get(
        "GATEWAY_BEARER_FILE",
        os.path.join(os.path.expanduser("~"), ".llm-gateway", "bearer.txt"))
    try:
        with open(_bearer_file, encoding="utf-8-sig") as _fh:   # utf-8-sig strips the BOM
            GATEWAY_BEARER = _fh.read().strip()
    except OSError:
        GATEWAY_BEARER = ""

# Ask the model to emit its reasoning trace (qwen3 "thinking"). Shown in the UI.
LLM_THINK = os.environ.get("LLM_THINK", "1") not in ("0", "false", "False")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_TEMPERATURE = 0.0
MAX_TOOL_ITERS = 5

# --- Storage ---
_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("COPILOT_DB", os.path.join(_HERE, "data", "copilot.db"))
LOG_PATH = os.environ.get("COPILOT_LOG", os.path.join(_HERE, "data", "copilot.log"))
MEMORY_DB_PATH = os.environ.get("COPILOT_MEMORY_DB",
                                os.path.join(_HERE, "data", "memory.db"))

# --- Agent memory: self-hosted Supermemory (local embeddings, no data egress) ---
# NOTE: supermemory-server 0.0.3 writes/extracts fine but its search endpoint returns
# no results; agent/memory.py dual-writes and falls back to a local FTS5 recall.
SUPERMEMORY_URL = os.environ.get("SUPERMEMORY_URL", "http://localhost:8787")
SUPERMEMORY_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")

# --- Observability: self-hosted Langfuse v2 (telemetry disabled, nothing leaves host) ---
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-copilot-demo")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-copilot-demo")
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "1") not in ("0", "false", "False")

# --- Rebalancing engine (mean-variance, hard vol constraint) -----------------
# Optimizer conventions (mirrored from a spreadsheet Solver model):
#   Value 1 (expected return) = (prod(1+r))^(252/n_obs) - 1     [trading-day CAGR]
#   Value 2 (volatility)      = std(daily, ddof=1) * sqrt(252)
#   Objective: maximise expected return s.t. vol <= sigma target, sum(w)=1, min<=w<=max.
OPT_ANNUALISATION = 252

# Default per-fund bounds in PERCENT (min, max). Everything is unconstrained
# except one money-market fund, capped at 45 to demonstrate that the optimizer
# honours a per-fund ceiling (see tests/test_optimizer.py::test_bounds_respected).
DEFAULT_FUND_BOUNDS: dict[str, tuple[float, float]] = {
    isin: (0.0, 100.0) for isin in FUND_UNIVERSE
}
DEFAULT_FUND_BOUNDS["ZZ0000000004"] = (0.0, 45.0)

FUND_NAMES: dict[str, str] = {   # invented names for the synthetic universe
    "ZZ0000000001": "Alpha Money Market Fund",
    "ZZ0000000002": "Alpha Islamic Money Market Fund",
    "ZZ0000000003": "Beta Islamic Cash Fund",
    "ZZ0000000004": "Beta Money Market Fund",
    "ZZ0000000005": "Beta Conservative Bond Fund",
    "ZZ0000000006": "Alpha Shariah Income Fund",
    "ZZ0000000007": "Alpha Cash Reserve Fund",
    "ZZ0000000008": "Alpha Bond Fund",
    "ZZ0000000009": "Alpha Asia Equity Dividend Fund",
    "ZZ0000000010": "Beta Gold Opportunity Fund",
    "ZZ0000000011": "Gamma Growth Fund",
    "ZZ0000000012": "Alpha Gold Equity Fund",
}

# Classifications for the synthetic universe. Tools that surface these must still
# label them approximate: they describe each fund's mandate, not a look-through.
FUND_META: dict[str, dict[str, str]] = {
    "ZZ0000000001": {"asset_class": "Cash / Money Market", "geography": "Domestic"},
    "ZZ0000000002": {"asset_class": "Cash / Money Market", "geography": "Domestic"},
    "ZZ0000000003": {"asset_class": "Cash / Money Market", "geography": "Domestic"},
    "ZZ0000000004": {"asset_class": "Cash / Money Market", "geography": "Domestic"},
    "ZZ0000000005": {"asset_class": "Fixed Income", "geography": "Domestic"},
    "ZZ0000000006": {"asset_class": "Fixed Income", "geography": "Domestic"},
    "ZZ0000000007": {"asset_class": "Cash / Money Market", "geography": "Domestic"},
    "ZZ0000000008": {"asset_class": "Fixed Income", "geography": "Domestic"},
    "ZZ0000000009": {"asset_class": "Equity", "geography": "Asia Pacific"},
    "ZZ0000000010": {"asset_class": "Commodity (Gold)", "geography": "Global"},
    "ZZ0000000011": {"asset_class": "Equity", "geography": "Unclassified"},
    "ZZ0000000012": {"asset_class": "Commodity-linked Equity", "geography": "Global"},
}

# Per-fund return-generating parameters (annual drift, annual vol) used ONLY by
# scripts/make_synthetic_fixtures.py to synthesise demo/test series. They are not
# read at runtime and are not forecasts — they exist to give the demo data a
# realistic risk ladder from cash through to commodity-linked equity.
SYNTHETIC_FUND_PARAMS: dict[str, tuple[float, float]] = {
    "ZZ0000000001": (0.033, 0.0018),
    "ZZ0000000002": (0.034, 0.0020),
    "ZZ0000000003": (0.032, 0.0016),
    "ZZ0000000004": (0.035, 0.0022),
    "ZZ0000000005": (0.045, 0.0120),
    "ZZ0000000006": (0.048, 0.0150),
    "ZZ0000000007": (0.031, 0.0015),
    "ZZ0000000008": (0.052, 0.0280),
    "ZZ0000000009": (0.085, 0.1350),
    "ZZ0000000010": (0.120, 0.1600),
    "ZZ0000000011": (0.090, 0.1400),
    "ZZ0000000012": (0.150, 0.2600),
}

# --- Benchmark presets. Per-period returns in percent; user-editable in the UI and
# persisted via store.benchmarks. These are illustrative reference lines for the
# demo (a deposit-rate proxy, an inflation-plus proxy, and absolute targets) — swap
# in whatever your own mandate benchmarks against. ---
BENCHMARK_PERIODS = ["annualized", "1m", "3m", "6m", "ytd", "1y", "3y", "5y", "all"]
BENCHMARK_PRESETS: dict[str, dict[str, float]] = {
    "Deposit Rate Proxy (5% p.a.)":
        {"annualized": 5.0, "1m": 0.41, "3m": 1.23, "6m": 2.47, "ytd": 3.75,
         "1y": 5.0, "3y": 15.8, "5y": 27.6, "all": 27.6},
    "Deposit Rate Proxy + 0.5%":
        {"annualized": 5.5, "1m": 0.45, "3m": 1.35, "6m": 2.71, "ytd": 4.13,
         "1y": 5.5, "3y": 17.4, "5y": 30.7, "all": 30.7},
    "Inflation Proxy + 2% p.a.":
        {"annualized": 4.0, "1m": 0.33, "3m": 0.99, "6m": 1.98, "ytd": 3.0,
         "1y": 4.0, "3y": 12.5, "5y": 21.7, "all": 21.7},
    "4% p.a. Absolute Return Target":
        {"annualized": 4.0, "1m": 0.33, "3m": 0.99, "6m": 1.98, "ytd": 3.0,
         "1y": 4.0, "3y": 12.5, "5y": 21.7, "all": 21.7},
    "6% p.a. Absolute Return Target":
        {"annualized": 6.0, "1m": 0.49, "3m": 1.47, "6m": 2.96, "ytd": 4.5,
         "1y": 6.0, "3y": 19.1, "5y": 33.8, "all": 33.8},
    "8% p.a. Absolute Return Target":
        {"annualized": 8.0, "1m": 0.64, "3m": 1.94, "6m": 3.92, "ytd": 6.0,
         "1y": 8.0, "3y": 26.0, "5y": 46.9, "all": 46.9},
}
DEFAULT_BENCHMARK = "Deposit Rate Proxy (5% p.a.)"
