"""Fenced code sandbox — the "data scientist" capability.

The model may write pandas/numpy over the loaded return data to answer open-ended
questions the hand-built tools don't cover. Its numbers are REAL (computed from real
data by real code) but they are NOT golden-tested like `engine/perf.py`, so every
result is labelled advisory and the executed code is surfaced to the user for audit.

Security model — allowlist, not blocklist:
  * The code is parsed to an AST and every node type is checked against an allowlist.
    Anything not explicitly permitted is rejected before execution.
  * Banned outright: import, exec/eval/compile, open, __builtins__ access, attribute
    access to dunders, global/nonlocal, class/function definitions, with, try, raise,
    lambda, comprehension over arbitrary calls to unknown names, del, assert.
  * Name resolution is restricted to an explicit environment (pd, np, df, returns,
    dates, plus a tiny set of safe builtins). There is no filesystem, no network, no
    os/sys/subprocess — they simply do not exist in the namespace.
  * Execution is capped by a wall-clock timeout in a worker thread.
  * The result must be JSON-serialisable scalars/records; huge frames are truncated.

This is defence in depth for a LOCAL, single-user tool. It is not a hostile-multi-tenant
jail — do not expose this endpoint to untrusted users over a network without revisiting.
"""
from __future__ import annotations

import ast
import queue
import threading
from datetime import date

import numpy as np
import pandas as pd

from engine.validate import CopilotError


class SandboxError(CopilotError):
    code = "SANDBOX_ERROR"


# --- what the model is allowed to write --------------------------------------
_ALLOWED_NODES = {
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.Slice, ast.Subscript,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Attribute,
    ast.keyword, ast.Starred, ast.IfExp, ast.If, ast.For, ast.Break, ast.Continue,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    # lambdas are required for idiomatic pandas (.apply/.agg). Their bodies are walked
    # and checked against this same allowlist, so they cannot smuggle anything in.
    ast.Lambda, ast.arguments, ast.arg,
    # operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub,
    ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt,
    ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Invert, ast.BitAnd, ast.BitOr,
    ast.BitXor,
}

_BANNED_ATTRS = {
    "__class__", "__bases__", "__subclasses__", "__mro__", "__globals__", "__code__",
    "__builtins__", "__import__", "__dict__", "__getattribute__", "__reduce__",
    "__reduce_ex__", "eval", "exec", "compile", "open", "system", "popen", "spawn",
    "to_csv", "to_pickle", "to_excel", "read_csv", "read_pickle", "read_excel",
}

_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "round": round,
    "sorted": sorted, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "range": range, "enumerate": enumerate, "zip": zip, "float": float, "int": int,
    "str": str, "bool": bool, "any": any, "all": all, "reversed": reversed,
}


def _validate(code: str) -> ast.Module:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"Syntax error in analysis code: {exc}") from exc

    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise SandboxError(
                f"Disallowed syntax: {type(node).__name__}. The sandbox permits only "
                "expressions, assignments, comparisons, if/for, and comprehensions over "
                "pandas/numpy — no imports, function/class definitions, or I/O.")
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_ATTRS:
            raise SandboxError(f"Disallowed attribute access: .{node.attr}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError(f"Disallowed dunder access: .{node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxError(f"Disallowed name: {node.id}")
    return tree


def _jsonable(obj, max_rows: int = 50):
    """Coerce a result to something JSON-serialisable and bounded."""
    if isinstance(obj, pd.DataFrame):
        truncated = len(obj) > max_rows
        out = obj.head(max_rows).reset_index()
        return {"type": "dataframe", "rows": len(obj), "truncated": truncated,
                "data": out.astype(object).where(pd.notna(out), None).to_dict("records")}
    if isinstance(obj, pd.Series):
        truncated = len(obj) > max_rows
        s = obj.head(max_rows)
        return {"type": "series", "rows": len(obj), "truncated": truncated,
                "data": {str(k): (None if pd.isna(v) else _scalar(v))
                         for k, v in s.items()}}
    if isinstance(obj, (np.ndarray, list, tuple)):
        seq = list(obj)[:max_rows]
        return {"type": "list", "rows": len(obj), "data": [_scalar(v) for v in seq]}
    if isinstance(obj, dict):
        return {"type": "dict", "data": {str(k): _scalar(v) for k, v in obj.items()}}
    return {"type": "scalar", "data": _scalar(obj)}


def _scalar(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (date, pd.Timestamp)):
        return str(v)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def run_analysis(code: str, dates: list[date], port_returns: list[float],
                 fund_returns: dict[str, list[float]], timeout_s: float = 10.0) -> dict:
    """Execute LLM-written pandas/numpy against the loaded data.

    The code must assign its answer to a variable named `result`.

    Namespace provided:
      df       DataFrame indexed by date: one column per fund ISIN + 'portfolio'
      returns  Series of portfolio daily returns (indexed by date)
      pd, np   pandas / numpy
    """
    tree = _validate(code)

    idx = pd.DatetimeIndex(pd.to_datetime([str(d) for d in dates]))
    df = pd.DataFrame({k: pd.Series(v, index=idx) for k, v in fund_returns.items()})
    df["portfolio"] = pd.Series(port_returns, index=idx)
    returns = df["portfolio"]

    env = {
        "pd": pd, "np": np, "df": df, "returns": returns,
        "__builtins__": _SAFE_BUILTINS,
    }

    box: queue.Queue = queue.Queue()

    def _run():
        try:
            exec(compile(tree, "<analysis>", "exec"), env, env)  # noqa: S102
            box.put(("ok", env.get("result", None)))
        except Exception as exc:  # noqa: BLE001
            box.put(("err", exc))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise SandboxError(f"Analysis exceeded the {timeout_s:.0f}s time limit "
                           "(likely an unbounded loop).")

    status, payload = box.get()
    if status == "err":
        raise SandboxError(f"Analysis failed: {type(payload).__name__}: {payload}")
    if payload is None:
        raise SandboxError("Analysis produced no `result` variable. Assign your answer "
                           "to `result`, e.g. `result = returns.resample('YE').apply("
                           "lambda s: (1+s).prod()-1)`.")

    return {
        "code": code,
        "result": _jsonable(payload),
        "advisory": True,
        "note": ("Ad-hoc analysis computed by model-written code over the real data. "
                 "NOT part of the Excel-validated attribution engine — verify before "
                 "relying on it for reporting."),
    }
