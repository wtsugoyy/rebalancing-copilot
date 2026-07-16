"""Rebalancing Copilot: Streamlit UI.

This is the ONLY module allowed to import streamlit. All math/data lives in
engine/ and store/ as UI-framework-agnostic functions, so a Phase 3 FastAPI reskin
(to merge with the Lightweight-Charts dashboard) wraps them without touching logic.
"""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

import config
import json

from agent import agent, graph
from agent import tracing as _tracing
from agent.tools import AppData, ToolContext
from engine import ingest
from engine.validate import CopilotError
from store import history

st.set_page_config(page_title="Rebalancing Copilot", page_icon="📊", layout="wide")


# --- persistent resources ---------------------------------------------------
@st.cache_resource
def get_conn():
    conn = history.connect()
    history.init_db(conn)
    return conn


@st.cache_data(show_spinner=False)
def _parse_nav(content: bytes, name: str):
    import io
    buf = io.BytesIO(content); buf.name = name
    return ingest.load_nav(buf)


@st.cache_data(show_spinner=False)
def _parse_yield(content: bytes, name: str):
    import io
    buf = io.BytesIO(content); buf.name = name
    return ingest.load_yields(buf)


def seed_defaults(conn, first_date: str):
    """Seed each portfolio's starting weights if none committed yet."""
    for code in config.PORTFOLIO_CODES:
        try:
            history.get_current(conn, code)
        except CopilotError:
            wmap = dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS[code]))
            history.save_snapshot(conn, code, first_date, wmap, source="seed",
                                  note="seeded from default allocation")


conn = get_conn()

# --- header + health --------------------------------------------------------
st.title("📊 Rebalancing Copilot")
st.caption("Offline attribution, rebalancing and history. Every figure is computed locally "
           "by deterministic code, never by the language model.")

ok, detail = agent.check_ollama()
_lf_ok, _lf_msg = _tracing.health()
c1, c2 = st.columns([2, 1])
with c1:
    (st.success if ok else st.info)(f"**Local AI:** {detail}")
with c2:
    st.caption(f"{'🟢' if _lf_ok else '⚪'} Tracing: {'Langfuse (self-hosted)' if _lf_ok else 'off'}"
               f" · 🧠 Memory: {'Supermemory + local' if config.SUPERMEMORY_URL else 'local'}")

# --- sidebar: data ----------------------------------------------------------
with st.sidebar:
    st.header("Data")
    nav_file = st.file_uploader("Daily NAV file (columns = ISINs)", type=["csv", "xlsx"])
    yld_file = st.file_uploader("MY 3Y yield file (Date, Yield %)", type=["csv", "xlsx"])
    portfolio = st.selectbox("Portfolio", config.PORTFOLIO_CODES,
                             format_func=lambda c: f"{c}: {config.PORTFOLIO_NAMES[c]}")

data: AppData | None = None
if nav_file and yld_file:
    try:
        bundle = _parse_nav(nav_file.getvalue(), nav_file.name)
        yields = _parse_yield(yld_file.getvalue(), yld_file.name)
        seed_defaults(conn, bundle.dates[0].isoformat())
        data = AppData(dates=bundle.dates, fund_returns=bundle.fund_returns,
                       yields=yields, isins=bundle.isins)
        st.sidebar.success(f"Loaded {bundle.n_obs} observations for {len(bundle.isins)} "
                           f"funds ({bundle.dates[0]} → {bundle.dates[-1]}).")

        # transparency: what the cleaner did (data-analyst behaviour, never silent)
        r = bundle.report
        notes = [f"detected **{r.get('mode', '?')}** data ({r.get('date_convention','?')} dates)"]
        if r.get("blank_rows_dropped"):
            notes.append(f"dropped {r['blank_rows_dropped']} blank/footer rows")
        if r.get("unparseable_date_rows_dropped"):
            notes.append(f"dropped {r['unparseable_date_rows_dropped']} rows with bad dates")
        if r.get("blank_return_cells_filled_zero"):
            notes.append(f"treated {r['blank_return_cells_filled_zero']} blank return cells as 0")
        st.sidebar.caption("🧹 Cleaning: " + "; ".join(notes) + ".")

        # warn if the yield file barely covers the analysis window (affects Sharpe/Sortino)
        ymin, ymax = min(d for d, _ in yields), max(d for d, _ in yields)
        covered = sum(1 for d in bundle.dates if ymin <= d <= ymax)
        frac = covered / len(bundle.dates)
        if frac < 0.5:
            st.sidebar.warning(
                f"⚠️ Yield file covers only {frac:.0%} of the NAV window "
                f"({ymin} → {ymax}). Risk-free falls back to {config.RF_FALLBACK:.1%} "
                f"elsewhere, so **Sharpe/Sortino are approximate**. Upload a yield file "
                f"spanning {bundle.dates[0]} → {bundle.dates[-1]} for exact risk-adjusted metrics.")
    except CopilotError as exc:
        st.sidebar.error(f"❌ {exc.message}")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"❌ Unexpected error: {exc}")

ctx = ToolContext(conn=conn, data=data)

tab_overview, tab_engine, tab_history, tab_bench = st.tabs(
    ["📊 Overview", "⚖️ Rebalancing Engine", "📜 History", "🎯 Benchmark"])

# --- Overview tab -------------------------------------------------------------
with tab_overview:
    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"Current allocation: {portfolio}")
        try:
            cur = history.get_current(conn, portfolio)
            wdf = pd.DataFrame(
                [{"ISIN": k, "Fund": config.FUND_NAMES.get(k, k), "Weight": f"{v:.2%}"}
                 for k, v in cur.weights.items() if v])
            st.dataframe(wdf, hide_index=True, use_container_width=True)
            st.caption(f"Effective {cur.effective_date} · source: {cur.source}")
        except CopilotError:
            st.info("No allocation yet. Upload data to seed the current portfolio.")
    with right:
        st.subheader("Performance attribution")
        if data is None:
            st.info("Upload a NAV file and a yield file to compute attribution.")
        else:
            try:
                from agent.tools import _attribution
                m = _attribution(ctx, portfolio)
                rows = [
                    ("Total return", f"{m['total_return']:.2%}"),
                    ("Annualised return", f"{m['ann_return']:.2%}"),
                    ("Annualised volatility", f"{m['ann_vol']:.2%}"),
                    ("Max drawdown", f"{m['max_drawdown']:.2%}"),
                    ("Sharpe", f"{m['sharpe']:.4f}"),
                    ("Sortino", f"{m['sortino']:.4f}"),
                    ("Avg risk-free (p.a.)", f"{m['avg_rf']:.2%}"),
                    ("Observations", str(m["n_obs"])),
                ]
                st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]),
                             hide_index=True, use_container_width=True)
                st.caption(f"Window {m['period_start']} → {m['period_end']} · matches the "
                           "attribution model, and every figure is computed here, not by the model.")
            except CopilotError as exc:
                st.error(f"❌ {exc.message}")

# --- Rebalancing Engine tab (exclusive) ----------------------------------------
with tab_engine:
    st.subheader("⚖️ Rebalancing Engine: mean-variance, hard volatility cap")
    st.caption("Maximises expected return subject to your volatility target and per-fund "
               "exposure bounds (Value 1 = expected return, Value 2 = volatility). "
               "The cap is hard: the solver will not trade it away for return.")
    if data is None:
        st.info("Upload the daily returns/NAV file first. The engine optimises over the "
                "loaded history.")
    else:
        sigma_pct = st.number_input("Sigma target: annualised volatility cap (%)",
                                    min_value=0.1, max_value=50.0, value=3.5, step=0.1)
        bounds_df = pd.DataFrame([
            {"ISIN": i, "Fund": config.FUND_NAMES.get(i, i),
             "Min %": config.DEFAULT_FUND_BOUNDS[i][0],
             "Max %": config.DEFAULT_FUND_BOUNDS[i][1]}
            for i in config.FUND_UNIVERSE])
        edited = st.data_editor(bounds_df, hide_index=True, use_container_width=True,
                                disabled=["ISIN", "Fund"], key="bounds_editor")
        if st.button("Run optimizer", type="primary"):
            from engine import optimizer as _opt
            bounds = {r["ISIN"]: (float(r["Min %"]), float(r["Max %"]))
                      for _, r in edited.iterrows()}
            try:
                st.session_state.opt_result = _opt.optimize(
                    data.fund_returns, data.dates, sigma_pct / 100.0, bounds,
                    isins=list(config.FUND_UNIVERSE)).to_dict()
            except CopilotError as exc:
                st.session_state.opt_result = None
                st.error(f"❌ {exc.message}")
        res = st.session_state.get("opt_result")
        if res:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected return", f"{res['expected_return']:.2%}")
            c2.metric("Volatility", f"{res['expected_vol']:.2%}",
                      delta=f"cap {res['sigma_target']:.2%}", delta_color="off")
            c3.metric("Sharpe", f"{res['sharpe']:.2f}")
            c4.metric("Rounded: ER / vol",
                      f"{res['rounded_return']:.2%} / {res['rounded_vol']:.2%}")
            tbl = pd.DataFrame([
                {"ISIN": f["isin"], "Fund": f["fund"],
                 "Min %": f["min_pct"], "Max %": f["max_pct"],
                 "Value 1 (ER)": f"{f['value1_expected_return']:.2%}",
                 "Value 2 (Vol)": f"{f['value2_volatility']:.2%}",
                 "Weight": f"{f['weight']:.2%}",
                 "Rounded": f"{f['weight_rounded']:.0%}"}
                for f in res["fund_table"]])
            st.dataframe(tbl, hide_index=True, use_container_width=True)
            note = ("Vol cap binding. The risk budget is fully used."
                    if res["vol_cap_binding"] else "Vol cap NOT binding.")
            st.caption(f"{note} Window {res['window']} ({res['n_obs']} obs).")
            eff = st.date_input("Effective date for commit", value=data.dates[-1])
            if st.button(f"✅ Commit rounded allocation to {portfolio} history"):
                import json as _json
                wmap = {i: 0.0 for i in config.FUND_UNIVERSE}
                wmap.update(res["weights_rounded"])
                history.save_snapshot(
                    conn, portfolio, eff.isoformat(), wmap,
                    sigma_target=res["sigma_target"], source="engine",
                    note=_json.dumps({"expected_return": res["rounded_return"],
                                      "expected_vol": res["rounded_vol"]}))
                st.success(f"Committed to {portfolio} history (effective {eff}). "
                           "The production engine remains the source of truth for live "
                           "allocations.")

# --- History tab ----------------------------------------------------------------
with tab_history:
    st.subheader(f"📜 Rebalance history: {portfolio}")
    snaps = history.list_history(conn, portfolio, limit=50)
    if not snaps:
        st.info("No snapshots yet.")
    else:
        rows = []
        for s in snaps:
            nz = {k: v for k, v in s.weights.items() if v}
            rows.append({
                "Effective": s.effective_date, "Source": s.source,
                "σ target": f"{s.sigma_target:.2%}" if s.sigma_target else "n/a",
                "Weights": ", ".join(
                    f"{config.FUND_NAMES.get(k, k).split(' Fund')[0]} {v*100:.0f}%"
                    for k, v in sorted(nz.items(), key=lambda kv: -kv[1])),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Append-only record. The copilot reads this history through its tools.")

# --- Benchmark tab ---------------------------------------------------------------
with tab_bench:
    from engine import benchmark as bm
    st.subheader("🎯 Portfolio benchmark")
    active = bm.get_active_benchmark(conn)
    st.caption(f"Active: **{active['name']}** ({active['source']})")
    preset = st.radio("Portfolio benchmarks", list(config.BENCHMARK_PRESETS),
                      index=list(config.BENCHMARK_PRESETS).index(active["name"])
                      if active["name"] in config.BENCHMARK_PRESETS else 0)
    seed = active["periods"] if preset == active["name"] \
        else config.BENCHMARK_PRESETS[preset]
    per_df = pd.DataFrame([{"Period": p, "Benchmark Return %": float(seed.get(p, 0.0))}
                           for p in config.BENCHMARK_PERIODS])
    edited_b = st.data_editor(per_df, hide_index=True, disabled=["Period"],
                              use_container_width=True, key="bench_editor")
    if st.button("💾 Save & Apply Benchmark", type="primary"):
        periods = {r["Period"]: float(r["Benchmark Return %"])
                   for _, r in edited_b.iterrows()}
        bm.save_benchmark(conn, preset, periods)
        st.success(f"'{preset}' saved and applied. The copilot's benchmark tools now "
                   "use these values.")
        st.rerun()

# --- chat -------------------------------------------------------------------
st.divider()
st.subheader("Ask the copilot")
st.caption("Every figure the copilot reports comes from a deterministic calculation above. "
           "it narrates, it never invents numbers.")

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []

for turn in st.session_state.chat_display:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("thinking"):
            with st.expander("🧠 Reasoning"):
                st.caption(turn["thinking"])
        if turn.get("evidence"):
            with st.expander("🔗 Grounding: tools the model called"):
                for e in turn["evidence"]:
                    st.code(f"{e['tool']}({e['args']})\n→ {e['summary']}", language="text")

if not ok:
    st.chat_input("Local AI unavailable. Deterministic panels above still work.", disabled=True)
elif data is None:
    st.chat_input("Upload data first to enable questions.", disabled=True)
else:
    prompt = st.chat_input("e.g. Is BAL's Sharpe good, and how far is vol from 10%?")
    if prompt:
        st.session_state.chat_display.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(f"Reasoning with {config.OLLAMA_MODEL}…"):
                try:
                    out = graph.chat(prompt, ctx, selected_portfolio=portfolio)
                    reply = out["answer"] or "(no answer)"
                    thinking = out.get("thinking") or ""
                    evidence = [
                        {"tool": t["tool"], "args": t["args"],
                         "summary": (t["result"].get("summary")
                                     or json.dumps(t["result"], default=str)[:220])}
                        for t in out.get("tool_results", [])
                    ]
                except Exception as exc:  # noqa: BLE001
                    reply = (f"Local AI error: {exc}. The deterministic panels above are "
                             "unaffected.")
                    thinking, evidence = "", []
            st.markdown(reply)
            if thinking:
                with st.expander("🧠 Reasoning"):
                    st.caption(thinking)
            if evidence:
                with st.expander("🔗 Grounding: tools the model called"):
                    for e in evidence:
                        st.code(f"{e['tool']}({e['args']})\n→ {e['summary']}", language="text")
        st.session_state.chat_display.append(
            {"role": "assistant", "content": reply, "thinking": thinking, "evidence": evidence})
