"""Offline LLM chat loop over the typed tools.

Determinism note: Ollama does not guarantee reproducible text even at
temperature 0, so the *chat transcript* is not the system of record — the
deterministic UI panels are. What IS guaranteed: the LLM never computes a number;
every figure it reports comes from a tool call whose args were pydantic-validated and
whose result was produced by the deterministic engine.
"""
from __future__ import annotations

from typing import Callable

import config
from agent import tools
from agent.tools import ToolContext

SYSTEM_PROMPT = (
    "You are the Rebalancing Copilot. You help an investment manager understand "
    "portfolio performance. RULES:\n"
    "1. You must NEVER compute or estimate a number yourself. Always call a tool and report "
    "its exact returned values.\n"
    "2. If a tool returns an 'error' field, explain it plainly and suggest the fix; do not "
    "invent a result.\n"
    "3. Portfolio codes are SC, SC+, BAL, BAL+, ADV, ADV+. Volatility/return targets are "
    "decimals (0.10 = 10%).\n"
    "4. Be concise and factual. Quote figures exactly as returned (e.g. Sharpe 1.3209).\n"
    "5. If a tool result contains a 'summary' field, use those exact labelled figures; do "
    "NOT relabel them (total return and annualised return are different numbers).\n"
    "6. 'Total return' is the whole-period return; 'annualised return' is per-year. Never "
    "swap them."
)

# Ollama tool schemas (JSON-schema function specs).
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_portfolios",
        "description": "List available portfolio codes and names.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_current_portfolio",
        "description": "Current live weights and effective date for a portfolio.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "get_history",
        "description": "Recent rebalance snapshots (newest first) for a portfolio.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "attribution",
        "description": "Realized performance metrics (total & annualised return, volatility, "
                       "max drawdown, Sharpe, Sortino) for a portfolio over the loaded price history.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "vol_vs_target",
        "description": "How far a portfolio's realized annualised volatility is from a target "
                       "(target is a decimal, e.g. 0.10 for 10%).",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)},
            "target": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}},
            "required": ["portfolio", "target"]}}},

    # --- analyst toolkit ---------------------------------------------------
    {"type": "function", "function": {
        "name": "periodic_returns",
        "description": "Return and volatility for EACH calendar year, quarter or month in the "
                       "period. USE THIS for questions like 'what was the return for each of the "
                       "3 years', 'yearly returns', 'quarterly breakdown', 'return per year'.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)},
            "freq": {"type": "string", "enum": ["year", "quarter", "month"]}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "volatility_trend",
        "description": "Whether volatility is INCREASING or DECREASING: rolling annualised "
                       "volatility, its direction, recent-vs-prior averages and trend slope. "
                       "USE THIS for 'is volatility rising or falling', 'volatility trend', "
                       "'is risk increasing', 'what is the trend right now'.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)},
            "window": {"type": "integer", "minimum": 10, "maximum": 252,
                       "description": "rolling window in trading days; 63 ~= one quarter"}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "attribution_period",
        "description": "Full metrics over a SPECIFIC date range (YYYY-MM-DD). Use to analyse or "
                       "compare sub-periods, e.g. 2024 vs 2025, or H1 performance.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)},
            "start": {"type": "string", "description": "YYYY-MM-DD inclusive"},
            "end": {"type": "string", "description": "YYYY-MM-DD inclusive"}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "fund_contribution",
        "description": "Which underlying funds drove the portfolio's return and its risk.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "drawdown_periods",
        "description": "The deepest drawdowns with start / trough / recovery dates.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)},
            "top": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "compare_portfolios",
        "description": "Side-by-side metrics for several portfolios, ranked by Sharpe.",
        "parameters": {"type": "object", "properties": {
            "portfolios": {"type": "array", "items": {
                "type": "string", "enum": list(config.PORTFOLIO_CODES)}}},
            "required": ["portfolios"]}}},
    {"type": "function", "function": {
        "name": "correlations",
        "description": "Pairwise correlation between the funds' daily returns.",
        "parameters": {"type": "object", "properties": {}}}},

    # --- rebalancing engine + benchmark suite --------------------------------
    {"type": "function", "function": {
        "name": "optimize_portfolio",
        "description": "Run the REBALANCING ENGINE: mean-variance optimisation maximising "
                       "expected return under a HARD volatility cap with per-fund min/max "
                       "bounds. Use for 'optimize at X% vol', 'what allocation should we "
                       "hold', 'rerun the engine'. Advisory — humans commit allocations.",
        "parameters": {"type": "object", "properties": {
            "sigma_target": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1,
                             "description": "vol cap as decimal, e.g. 0.035 = 3.5%"}},
            "required": ["sigma_target"]}}},
    {"type": "function", "function": {
        "name": "target_vs_realized",
        "description": "FIXED vs CURRENT: the committed volatility target vs the realized "
                       "volatility and returns. Use for 'fixed vs current vol', 'are we on "
                       "target', 'target vs actual'.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "composition_change",
        "description": "COMPOSITION BEFORE vs AFTER: previous allocation vs current one, "
                       "per-fund deltas and turnover. Use for 'what changed in the last "
                       "rebalance', 'before and after weights'.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "benchmark_metrics",
        "description": "Portfolio vs the ACTIVE BENCHMARK: per-period benchmark returns, "
                       "excess, tracking error, information ratio, monthly win rate (plus "
                       "beta/YTM availability notes). Use for any benchmark question.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "monthly_stats",
        "description": "BEST/WORST MONTH, positive-months %, win rate vs benchmark, and "
                       "month-over-month Value-at-Risk (95%/99% historical VaR). Pass "
                       "`year` to scope to one calendar year (e.g. best month in 2026).",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)},
            "year": {"type": "integer", "description": "optional calendar year, e.g. 2026"}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "rolling_returns",
        "description": "ROLLING 12- or 36-month compounded returns (latest, min, max, series).",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)},
            "window_months": {"type": "integer", "enum": [3, 6, 12, 24, 36]}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "portfolio_drift",
        "description": "PORTFOLIO DRIFT %: how far buy-and-hold has pulled weights away "
                       "from target since the last rebalance, per fund and in total.",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "allocation_breakdown",
        "description": "ASSET ALLOCATION and GEOGRAPHIC breakdown of the current weights "
                       "(mandate-derived, approximate; explains sector-data availability).",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)}},
            "required": ["portfolio"]}}},
    {"type": "function", "function": {
        "name": "top_holdings",
        "description": "TOP 5 HOLDINGS look-through from factsheet data (partial coverage, "
                       "clearly stated).",
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)}},
            "required": ["portfolio"]}}},

    # --- fenced sandbox (advisory, only when nothing above fits) -----------
    {"type": "function", "function": {
        "name": "run_analysis",
        "description": (
            "LAST RESORT ONLY — ad-hoc pandas/numpy over the raw returns, for questions NO "
            "other tool can answer (e.g. '% of days with a loss', 'longest losing streak'). "
            "Its results are ADVISORY and NOT regression-tested, so using it when a specific "
            "tool exists produces a WORSE answer. "
            "DO NOT use it for: yearly/quarterly/monthly returns (use periodic_returns); "
            "volatility trend or rising/falling (use volatility_trend); a date range (use "
            "attribution_period); best/worst month or VaR (use monthly_stats); fund "
            "contribution (use fund_contribution); drift (use portfolio_drift); benchmark "
            "comparison (use benchmark_metrics); Sharpe/vol/drawdown (use attribution). "
            "Call it at most ONCE per question. Assign the answer to `result`. Available: "
            "`df` (DataFrame indexed by date, one column per fund ISIN plus 'portfolio'), "
            "`returns` (portfolio daily-return Series), `pd`, `np`. No imports, no I/O. "
            "Example: result = (returns < 0).mean()"),
        "parameters": {"type": "object", "properties": {
            "portfolio": {"type": "string", "enum": list(config.PORTFOLIO_CODES)},
            "code": {"type": "string", "description": "Python assigning to `result`"}},
            "required": ["portfolio", "code"]}}},
]


def check_ollama() -> tuple[bool, str]:
    """Health of whichever LLM backend is configured (local Ollama or the remote
    gateway). Name kept for backwards compatibility with existing callers/tests.
    Never raises."""
    try:
        from agent.llm import build_provider
        return build_provider().health()
    except Exception as exc:  # noqa: BLE001
        return False, f"AI backend unavailable: {exc}"


def chat(user_message: str, ctx: ToolContext, transcript: list[dict] | None = None,
         *, _client=None) -> tuple[str, list[dict]]:
    """Run one user turn through the tool-calling loop. Returns (reply, new_transcript).

    `_client` is injectable for tests. Falls back to a real ollama.Client otherwise.
    """
    if _client is None:
        import ollama
        _client = ollama.Client(host=config.OLLAMA_HOST)

    messages = transcript[:] if transcript else [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_message})

    for _ in range(config.MAX_TOOL_ITERS):
        resp = _client.chat(model=config.OLLAMA_MODEL, messages=messages,
                            tools=TOOL_SCHEMAS, options={"temperature": config.LLM_TEMPERATURE})
        msg = resp["message"]
        messages.append({"role": "assistant", "content": msg.get("content", ""),
                         "tool_calls": msg.get("tool_calls")})
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return msg.get("content", ""), messages

        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            result = tools.execute_tool(name, args, ctx)
            import json
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result, default=str)})

    return ("I couldn't complete that within the tool-call limit. Please check the "
            "deterministic panels above for the figures."), messages
