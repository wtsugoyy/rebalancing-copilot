# Engineering notes

Problems this project actually hit, and how they were solved. Kept because the reasoning is
the useful part. Several of these look like "the AI is bad" until you find the real cause.

---

## The AI was never dumb, it was tool-starved

**Symptom.** It couldn't answer *"what was the return for each of the last 3 years?"* or *"is
volatility increasing?"*. Easy conclusion: the model is too small.

**Actual cause.** It had five tools, and `attribution()` was whole-period only. There was **no
way for it to be right**. No amount of model capacity fixes a missing capability.

**Fix.** Built the analyst toolkit: `periodic_returns`, `volatility_trend`,
`attribution_period`, `fund_contribution`, `drawdown_periods`, `compare_portfolios`,
`correlations`, plus a fenced sandbox for genuinely novel questions.

**Lesson,** now recorded in `CLAUDE.md`: if it can't answer something, the fix is almost always
a missing tool, not a bigger model.

Two bugs the tests caught during that work:

- The sandbox **rejected `lambda`**, which breaks `df.apply(lambda ...)`, i.e. core pandas.
- The volatility-trend logic **conflated two different questions**: *"is vol rising right
  now?"* and *"has vol risen over the whole period?"*. These can have **opposite answers**.
  They're now reported separately, because collapsing them produces confident nonsense.

---

## Context-blindness: the model's training prior beat the data

**Symptom.** Asked for *"best and worst month in 2026"*, the copilot **refused**: "2026 has not
yet occurred (assuming the current date is 2023)". It also asked for a portfolio code the
dashboard already had on screen.

**Actual cause.** The system prompt was **static**. The model was never told what data was
loaded, what today's date is, or which portfolio was selected, so its training prior won. It
wasn't hallucinating. It was reasoning correctly from the only context it had.

**Fix.** A dynamic **DATASET CONTEXT** block injected every turn: the real current date, the
loaded window and observation count, an explicit list of years covered (marked "NOT in the
future"), the selected portfolio, and the active benchmark, plus a hard rule against
calendar-based refusals. `monthly_stats` gained a `year` argument.

---

## Prompt ordering is load-bearing (found only by soaking)

The system prompt is assembled as context, then memories, then rules, with the rules **strictly
last**. This is not stylistic. Measured against an OpenAI-compatible gateway model, 22 tools,
same six questions:

| Ordering | Native tool calls |
|---|---|
| rules first (context appended last) | **0/6** |
| context first (rules last) | **6/6** |

Even innocuous filler appended after the rules degraded it (1/3). The failure is silent and
nasty: instead of calling tools, the model emits prose JSON that *looks* like an answer. On
financial data, that means ungrounded numbers reaching a user.

`qwen3:14b` tolerated the bad ordering; the gateway model did not. Pinned by
`tests/test_context.py::test_rules_come_last_in_system_prompt` so nobody tidies the prompt and
quietly breaks grounding.

---

## DeepSeek-R1 rejected: it would have fabricated financial numbers

R1 14B was the initial ask. **Empirically tested:** asked for a portfolio's volatility, it
ignored the tool, **hallucinated the ticker "BAC"**, and invented a methodology. Confirmed by
[ollama#10935](https://github.com/ollama/ollama/issues/10935): official R1 builds ship no
tool-calling template.

**Fix.** `qwen3:14b`, which gives reasoning ("thinking") *and* native tool calling, so
reasoning was gained without losing grounding. Written down so it doesn't get re-litigated.

---

## Two sources of truth is one too many

The app seeded portfolio weights from a hand-transcribed table in `config.py`, while the golden
test read weights from a separately-extracted fixture. They disagreed: one portfolio displayed
23.91% where it should have shown 11.66%. **The golden test passed the whole time**, because it
was reading the other source.

**Fix.** `tests/test_config.py` now asserts the two agree, and the fixtures are generated *from*
`config.py` rather than transcribed alongside it. The disagreement is now structurally
impossible rather than merely tested for.

---

## Real files broke ingestion

Synthetic test data is well-behaved. Real files were not: daily **returns** rather than NAV
(signed, tiny values), 83 blank trailing rows, a BOM, quoted fields, `MM/DD/YYYY` dates, and
the yield hiding in a column called `Price`.

**Fix.** `ingest.py` became an analyst-grade cleaner: auto-detects NAV vs returns, drops
blank/footer rows, handles BOM, quotes, mixed date conventions and descending order, and
**reports every cleaning action** in the sidebar. Silent cleaning is how you get numbers nobody
can reproduce.

---

## Langfuse was silently dead

The `langfuse>=2.50.0` pin had **no upper bound**, so a rebuild installed SDK **4.13.2** against
a **v2 server**. `auth_check()` died with a pydantic `ValidationError` and tracing turned itself
off, silently, which is the worst kind.

**Fix.** Pinned `langfuse>=2.60,<3`, with the reason written in `requirements.txt` so the pin
doesn't look arbitrary later. Bump the server image before relaxing it.

---

## Supermemory recall is broken upstream

Self-hosted supermemory-server `0.0.3`: writes succeed, documents reach `status=done`, the
memory agent extracts memories, but `/v3/search` **and** `/v4/search` always return `total: 0`
(verified with and without `containerTag`, threshold 0). It also 401s on every auth header
variant from inside a container; only unauthenticated localhost works.

**Fix.** `agent/memory.py` dual-writes to Supermemory and **reads from a local SQLite FTS5
index** that actually works. It auto-prefers Supermemory the moment upstream returns hits. Its
extraction agent also needs a **non-thinking** model (`qwen2.5:3b`); a thinking model's
`<think>` blocks break its structured output.

---

## Principles that fell out of this

- **The LLM routes, it never computes.** Every number comes from deterministic code.
- **No silent failures.** Bad input raises a typed error; the UI renders it, the agent returns
  it structured. Degrading quietly is worse than erroring loudly.
- **Observability must never break the product.** Tracing and memory failures are no-ops.
- **History is append-only.** Corrections are new rows. There is no UPDATE path.
- **Say "I don't know" in the data model.** Beta against a flat benchmark is undefined, so the
  tool returns `null` and the reason. It does not return a number-shaped guess.
