# Rebalancing Copilot

An **offline** portfolio analytics dashboard with an LLM chat interface that *cannot make up
numbers*. It runs performance attribution, proposes rebalanced allocations under a hard
volatility cap, keeps an append-only history of every change, and answers risk/return
questions in plain English, entirely on your own machine.

The interesting constraint is the one in bold: **the model never computes anything.** It may
only choose which deterministic tool to call. Every figure in every answer comes out of plain
Python in `engine/`. The model's job is to decide what to ask and to explain what came back.
On financial data, a fluent wrong number is worse than no answer.

## Why it's built this way

**The LLM is a router, not a calculator.** Give a language model a table of returns and ask for
a Sharpe ratio and it will happily produce a plausible one. So it isn't given the option: the
tool layer computes, the model narrates. `tests/test_graph.py` pins this. If a refactor ever
lets a number reach the user without passing through a tool, the suite fails.

**DeepSeek-R1 was evaluated and rejected.** Official Ollama builds ship no tool-calling
template. Asked for a portfolio's volatility it ignored the tool, hallucinated a ticker
("BAC"), and invented a methodology. That's exactly the failure mode above, so the model here
is `qwen3:14b`, which does reasoning *and* native tool calls.

**Prompt ordering turned out to be load-bearing.** The system prompt is assembled as
`context, then memories, then rules`, with the rules strictly last. Measured against an
OpenAI-compatible gateway model with the same 22 tools and the same six questions: rules-first
produced **0/6** native tool calls, rules-last produced **6/6**. Even innocuous filler
appended after the rules degraded it. When tool calling silently degrades, the model starts
emitting prose JSON instead, which means ungrounded numbers. `tests/test_context.py` pins the
ordering so nobody tidies up the prompt and quietly breaks grounding.

**The model was never dumb, it was tool-starved.** Most "the AI can't answer this" moments
turned out to need a new tool, not a bigger model. Hence 22 of them.

## What it does

- **Ingest.** Daily NAV or return series (one column per fund, keyed by ISIN) plus a risk-free
  yield series. Auto-detects NAV vs returns, and reports every cleaning decision rather than
  silently fixing things.
- **Attribution.** Total and annualised return, annualised volatility, max drawdown, Sharpe,
  Sortino, across six model portfolios.
- **Rebalancing engine.** Mean-variance optimisation maximising expected return subject to a
  *hard* volatility cap, `Σw = 1`, and per-fund min/max bounds. SLSQP with deterministic
  multi-start, plus whole-percent rounding that preserves the constraints.
- **History store.** Append-only SQLite. Corrections are new rows; there is no update or
  delete path.
- **Offline chat.** A LangGraph harness: recall, model, deterministic tools, reason, remember.
  The UI shows the reasoning trace and the exact tool calls behind each answer.

Beyond whole-period attribution: per-year/quarter/month returns, volatility trend, date-range
analysis, fund contribution to return and risk, drawdown periods, portfolio comparison,
correlations, drift, VaR, rolling 12/36-month windows, and benchmark-relative TE/IR.

For genuinely novel questions there's a **fenced pandas sandbox** (`engine/sandbox.py`). It
uses an AST **allowlist**, not a blocklist: no imports, no I/O, no dunder or attribute escapes,
restricted builtins, wall-clock timeout. Its results are stamped `advisory: true` and the
generated code is shown to you, because unlike the tools it is not regression-tested.
`tests/test_sandbox.py` is the security boundary. If a case marked "blocked" ever starts
passing, the sandbox is compromised.

## Data: everything here is synthetic

The 12 funds in this repo are **invented**. Their ISINs use the `ZZ` prefix, which is reserved
for user-assigned codes and never issued to a real security. Their return series, the yield
series, the demo holdings, and the starting weights are all generated from a seeded RNG by
[`scripts/make_synthetic_fixtures.py`](scripts/make_synthetic_fixtures.py). The weights are
made up to give the demo a risk ladder; they are not investment advice.

**What that means for the tests.** The expected values in `tests/fixtures/` were produced by
running this engine over that synthetic data and recording the output. They are *regression*
baselines: they fail loudly if a refactor changes a number, which is what you want from a CI
gate. They are **not** an independent oracle. They cannot prove the metric definitions are
correct, because the engine is what computed them. Validating the definitions against an
external reference implementation would be a separate exercise, and this repository does not
claim to have done it.

Point it at your own data and that data stays in `data/`, which is gitignored.

## Quick start

```powershell
copy .env.example .env               # first time only
powershell -File scripts\start_stack.ps1
```

| Service | URL | Notes |
|---|---|---|
| Dashboard | http://localhost:8501 | Streamlit |
| Langfuse (tracing) | http://localhost:3000 | `admin@copilot.local` / `copilot-demo-1234` |
| Supermemory (memory) | http://localhost:8787 | local embeddings, local storage |
| Ollama (LLM) | http://localhost:11434 | **native**, GPU-accelerated |

Upload `sample/nav_sample.csv` and `sample/yield_sample.csv` to try it immediately.

Everything runs on the host: Ollama, Supermemory (local embeddings) and Langfuse (telemetry
disabled). Nothing leaves the machine unless you deliberately set `LLM_PROVIDER=gateway`,
which sends prompts and tool results to whatever OpenAI-compatible endpoint you configure.

### Why Ollama is not in Docker
Docker on Windows cannot pass through an AMD GPU. Run it natively and `qwen3:14b` loads 100%
into VRAM (ROCm, `gfx1200`, RX 9060 XT) instead of crawling on CPU.

## Input file formats

Ingestion cleans real-world files and reports what it did. It never cleans silently. It
tolerates BOM headers, quoted fields, blank/footer rows, thousands separators, descending date
order, and both `M/D/YYYY` and `D/M/YYYY`.

- **Fund file** (`.csv`/`.xlsx`): a `Date` column plus one column per fund, header = the fund's
  **ISIN** (must be one of the 12 in `config.FUND_UNIVERSE`; unknown headers are rejected by
  name). Values may be **either NAV prices or daily returns, auto-detected** (signed and small
  means returns, used as-is; strictly positive and larger means NAV, differenced into simple
  returns). Blank return cells are treated as `0`.
- **Yield file** (`.csv`/`.xlsx`): a `Date` column plus `Yield`, `Price`, `Close`, `Rate`, or
  `Last` (percent, e.g. `3.35`; a `%` suffix is stripped). Dates outside the file's coverage
  fall back to 3.3% p.a., logged rather than silent. If the yield file covers less than half
  the analysis window the UI warns you, because Sharpe and Sortino depend on it (total return
  and volatility do not).

## Tests

```powershell
python -m pytest -q                      # full suite
python -m pytest tests/test_soak.py -q   # determinism / no-silent-failure soak
```

The suite covers the regression gates above plus the grounding invariant, the sandbox security
boundary, ingest edge cases, append-only history, and a soak test for determinism.
Observability and memory are tested to **fail open**: if Langfuse or Supermemory is dead, the
product still answers (`test_agent_survives_dead_memory_and_tracing`).

## Honest gaps

The tools state these rather than guessing:

- **Beta** against a flat-rate benchmark is undefined (zero variance). The tool returns `null`
  and says why.
- **YTM** and **sector allocation** need factsheet/holdings data that isn't loaded.
- **Top-5 look-through** covers 3 of 12 funds, and says so rather than implying full coverage.
- The `run_analysis` sandbox is advisory and labelled as such.

## Architecture invariants

- **`app.py` is the only file that may import `streamlit`.** Everything in `engine/`, `store/`,
  and `agent/` is UI-framework-agnostic pure Python, so the whole thing can be wrapped in a
  FastAPI/JSON layer without touching the math.
- **The LLM never computes.** Guarded by `tests/test_graph.py`.
- **Observability must never break the product.** Langfuse and Supermemory failures degrade to
  no-ops.
- **History is append-only.** No UPDATE, no DELETE.

## Project layout

```
config.py            # the determinism contract: every constant, and the synthetic universe
engine/              # ingest, rf, perf, analytics, optimizer, sandbox, validate. Pure Python
store/               # schema.sql + history.py (SQLite, WAL, append-only)
agent/               # tools.py (typed tools) · graph.py (LangGraph harness)
                     # memory.py (Supermemory + local FTS) · llm.py (pluggable backend)
app.py               # Streamlit UI (the ONLY file importing streamlit)
tests/               # regression gates, grounding, sandbox, ingest, history, soak
scripts/             # start_stack.ps1 · make_synthetic_fixtures.py
sample/              # runnable NAV + yield sample data (synthetic)
```
