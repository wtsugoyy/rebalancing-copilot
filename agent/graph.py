"""LangGraph agent harness (Hermes-style tool calling via Ollama).

    recall ─▶ model ─┬─(tool_calls)─▶ tools ─┐
                     │                        └──▶ (back to model, capped)
                     └─(no tool_calls)──────▶ remember ─▶ END

Invariants (do not weaken):
* The LLM **never computes**. It may only pick a tool; every number is produced by
  deterministic `engine/` code and quoted verbatim.
* Reasoning ("thinking") is captured for display, but it is NOT a source of numbers.
* Memory stores *context/preferences*, never figures.
* Tracing and memory are best-effort: if Langfuse or Supermemory are down, the agent
  still answers.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

import config
from agent import tools as tools_mod
from agent import tracing
from agent.llm import LLMProvider, build_provider
from agent.memory import Memory, build_memory
from agent.tools import ToolContext
from engine.obs import get_logger

_log = get_logger()

SYSTEM_PROMPT = (
    "You are the Rebalancing Copilot, an offline investment analyst assistant.\n"
    "ABOUT THE DATASET: six model portfolios over a universe of 12 unit-trust funds, "
    "spanning money-market, fixed-income, equity and commodity-linked mandates. The "
    "portfolios form a risk ladder from most conservative to most aggressive: SC (Secure "
    "Cash), SC+ (Secure Cash+), BAL (Balanced), BAL+ (Balanced+), ADV (Advanced), ADV+ "
    "(Advanced+). Each is rebalanced with a mean-variance engine that maximises expected "
    "return under a hard volatility cap (the 'sigma target') with per-fund exposure "
    "bounds.\n"
    "RULES (absolute):\n"
    "1. You NEVER compute, estimate, or invent a number. Every figure must come from a "
    "tool result. If you need a number, call a tool.\n"
    "2. Quote tool figures exactly as returned (e.g. Sharpe 1.3209). If a tool result "
    "has a 'summary' field, use its exact labelled figures and do not relabel them.\n"
    "3. 'Total return' is whole-period; 'annualised return' is per-year. Never swap them.\n"
    "4. Portfolio codes: SC, SC+, BAL, BAL+, ADV, ADV+. Targets are decimals "
    "(0.10 = 10%).\n"
    "5. You MAY reason about what the numbers mean (is a Sharpe good, is vol far from "
    "target, which portfolio suits a risk appetite) — reasoning is encouraged, "
    "fabrication is not.\n"
    "6. If a tool returns an error, report it plainly. Never guess around it.\n"
    "7. The production rebalancing engine remains the source of truth for live "
    "allocations; anything you suggest is advisory.\n"
    "8. The DATASET CONTEXT above is authoritative about dates. NEVER assume the current "
    "date from your training data, and NEVER refuse a question about a period that lies "
    "inside the loaded data window — call a tool and answer from the data.\n"
    "9. TOOL CHOICE — always prefer the SPECIFIC tool over `run_analysis`. The specific "
    "tools are deterministic and regression-tested; `run_analysis` runs model-written "
    "code and its results are only advisory. Route:\n"
    "   - return per year / quarter / month  -> periodic_returns (NOT run_analysis)\n"
    "   - volatility rising or falling, trend -> volatility_trend (NOT run_analysis)\n"
    "   - metrics over a date range          -> attribution_period\n"
    "   - best/worst month, VaR, win rate    -> monthly_stats\n"
    "   - which funds drove return/risk      -> fund_contribution\n"
    "   - drift from target weights          -> portfolio_drift\n"
    "   - vs benchmark, TE, IR               -> benchmark_metrics\n"
    "   Use `run_analysis` ONLY if no tool above can answer the question. Never call it "
    "more than once for the same question — if it fails, say so plainly."
)

# ORDERING IS LOAD-BEARING — DO NOT APPEND ANYTHING AFTER SYSTEM_PROMPT.
#
# The system prompt is assembled as:   [context] + [memories] + SYSTEM_PROMPT
# i.e. the RULES ALWAYS COME LAST. Measured against an OpenAI-compatible gateway
# model (`local-main`), with 22 tools and the same question set:
#     rules -> context (context appended after)  ->  0/6 native tool calls
#     context -> rules (rules last)              ->  6/6 native tool calls
# Even innocuous filler appended after the rules degraded it (1/3). The model's
# tool-calling behaviour depends on the "call a tool" instruction being closest to the
# user turn. qwen3:14b tolerated the old order; local-main does not. Appending anything
# after SYSTEM_PROMPT silently turns tool calls into prose JSON — i.e. ungrounded
# answers on financial data. Regression-tested in tests/test_context.py.


def _context_block(ctx, selected_portfolio: str | None) -> str:
    """Dynamic, authoritative context about the loaded session — this is what stops the
    model from 'assuming the current date is 2023' and refusing answerable questions."""
    import datetime as _dt

    import config as _c

    lines = ["\n\nDATASET CONTEXT (authoritative, from the live dashboard):"]
    lines.append(f"- Actual current date: {_dt.date.today().isoformat()}.")
    if ctx.data is not None and ctx.data.dates:
        d0, d1 = ctx.data.dates[0], ctx.data.dates[-1]
        years = sorted({d.year for d in ctx.data.dates})
        lines.append(
            f"- Loaded data: daily returns for {len(ctx.data.isins)} funds, "
            f"{d0.isoformat()} to {d1.isoformat()} ({len(ctx.data.dates)} observations). "
            f"Use {d1.isoformat()} as the analysis as-of date.")
        lines.append(
            f"- Years covered by the data: {', '.join(map(str, years))}. Questions about "
            f"any of these years are answerable from the data — they are NOT in the future.")
    else:
        lines.append("- No price data loaded yet: say so and ask for the upload; history "
                     "and benchmark questions still work.")
    if selected_portfolio:
        lines.append(
            f"- Portfolio currently selected in the dashboard: {selected_portfolio} "
            f"({_c.PORTFOLIO_NAMES.get(selected_portfolio, selected_portfolio)}). When the "
            f"user says 'the portfolio' or names none, use {selected_portfolio}.")
    try:
        from engine import benchmark as _bm
        b = _bm.get_active_benchmark(ctx.conn)
        lines.append(f"- Active benchmark: {b['name']}.")
    except Exception:  # noqa: BLE001 — context enrichment must never break chat
        pass
    return "\n".join(lines)


# Reuse the tool schemas already validated against the deterministic engine.
from agent.agent import TOOL_SCHEMAS  # noqa: E402


class AgentState(TypedDict, total=False):
    question: str
    messages: list[dict]
    memories: list[str]
    tool_results: list[dict]
    tool_names: list[str]
    thinking: str
    answer: str
    iterations: int
    pending_calls: list        # typed ToolCall objects awaiting execution


def build_graph(ctx: ToolContext, memory: Memory | None = None, tracer=None,
                selected_portfolio: str | None = None,
                provider: LLMProvider | None = None):
    memory = memory or build_memory()
    tracer = tracer or tracing._NullTrace()
    llm = provider or build_provider()

    # ---------------------------------------------------------------- nodes
    def recall(state: AgentState) -> AgentState:
        span = tracer.span("recall", input=state["question"])
        hits = []
        try:
            hits = [h.text for h in memory.recall(state["question"])]
        except Exception as exc:  # noqa: BLE001
            _log.warning("memory recall failed (non-fatal): %s", exc)
        span.end(output={"n": len(hits)})

        # Rules LAST — see the ordering note above SYSTEM_PROMPT. Never append after it.
        parts = [_context_block(ctx, selected_portfolio).strip()]
        if hits:
            parts.append("Remembered context about this analyst (may inform your reasoning, "
                         "but never overrides tool figures):\n- " + "\n- ".join(hits))
        parts.append(SYSTEM_PROMPT)
        sys = "\n\n".join(p for p in parts if p)
        return {
            "messages": [{"role": "system", "content": sys},
                         {"role": "user", "content": state["question"]}],
            "memories": hits,
            "tool_results": [],
            "tool_names": [],
            "pending_calls": [],
            "iterations": 0,
        }

    def call_model(state: AgentState) -> AgentState:
        gen = tracer.generation("model", model=f"{llm.name}:{getattr(llm, 'model', '?')}",
                                input=state["messages"][-1])
        try:
            reply = llm.chat(state["messages"], TOOL_SCHEMAS)
        except Exception as exc:  # noqa: BLE001
            gen.end(output=f"error: {exc}")
            return {"answer": f"AI backend unavailable ({llm.name}): {exc}",
                    "messages": state["messages"], "pending_calls": []}

        thinking = (state.get("thinking") or "") + (reply.thinking or "")
        gen.end(output={"content": (reply.content or "")[:500],
                        "tool_calls": [c.name for c in reply.tool_calls]})
        return {
            "messages": state["messages"] + [llm.assistant_message(reply)],
            "pending_calls": reply.tool_calls,
            "thinking": thinking,
            "answer": reply.content or "",
            "iterations": state.get("iterations", 0) + 1,
        }

    def exec_tools(state: AgentState) -> AgentState:
        results, names = list(state.get("tool_results", [])), list(state.get("tool_names", []))
        msgs = state["messages"]
        for call in state.get("pending_calls", []):
            span = tracer.span("tool:" + call.name, input=call.arguments)
            try:
                result = tools_mod.execute_tool(call.name, call.arguments, ctx)
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
            span.end(output=result)
            _log.info("tool=%s args=%s ok=%s", call.name, call.arguments,
                      "error" not in result)
            results.append({"tool": call.name, "args": call.arguments, "result": result})
            names.append(call.name)
            msgs = msgs + [llm.tool_result_message(call, json.dumps(result, default=str))]
        return {"messages": msgs, "tool_results": results, "tool_names": names,
                "pending_calls": []}

    def remember(state: AgentState) -> AgentState:
        # Store durable context only (the question + which portfolios were examined),
        # never the figures themselves — those must always be recomputed from data.
        try:
            if state.get("tool_names"):
                memory.remember(f"Analyst asked: {state['question']} "
                                f"(tools used: {', '.join(sorted(set(state['tool_names'])))})")
        except Exception as exc:  # noqa: BLE001
            _log.warning("memory write failed (non-fatal): %s", exc)
        return {}

    # ------------------------------------------------------------- routing
    def should_continue(state: AgentState) -> str:
        if state.get("pending_calls") and state.get("iterations", 0) < config.MAX_TOOL_ITERS:
            return "tools"
        return "remember"

    g = StateGraph(AgentState)
    g.add_node("recall", recall)
    g.add_node("model", call_model)
    g.add_node("tools", exec_tools)
    g.add_node("remember", remember)
    g.set_entry_point("recall")
    g.add_edge("recall", "model")
    g.add_conditional_edges("model", should_continue, {"tools": "tools", "remember": "remember"})
    g.add_edge("tools", "model")
    g.add_edge("remember", END)
    return g.compile()


def chat(question: str, ctx: ToolContext, memory: Memory | None = None,
         selected_portfolio: str | None = None,
         provider: LLMProvider | None = None) -> dict:
    """Run one grounded, traced, reasoning turn. Returns answer + thinking + evidence."""
    with tracing.trace("copilot-chat", user_input=question,
                       metadata={"provider": config.LLM_PROVIDER}) as tr:
        graph = build_graph(ctx, memory=memory, tracer=tr,
                            selected_portfolio=selected_portfolio, provider=provider)
        final = graph.invoke({"question": question})
        tr.update(output=final.get("answer", ""))
    return {
        "answer": final.get("answer", ""),
        "thinking": final.get("thinking", ""),
        "tool_results": final.get("tool_results", []),
        "memories": final.get("memories", []),
    }
