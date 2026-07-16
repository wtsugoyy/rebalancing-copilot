"""Sandbox: it must compute real answers, and it must refuse to escape.

The sandbox runs model-written code. These tests are the security boundary — if any
"blocked" case starts passing, the sandbox is compromised.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from engine import sandbox
from engine.sandbox import SandboxError


@pytest.fixture
def data():
    dates = [date.fromordinal(date(2024, 1, 1).toordinal() + i) for i in range(300)]
    rng = np.random.default_rng(3)
    fr = {"AAA": list(rng.normal(0.0004, 0.01, 300)),
          "BBB": list(rng.normal(0.0002, 0.005, 300))}
    port = [0.6 * a + 0.4 * b for a, b in zip(fr["AAA"], fr["BBB"])]
    return dates, port, fr


# --- it actually works -------------------------------------------------------
def test_computes_yearly_returns(data):
    dates, port, fr = data
    out = sandbox.run_analysis(
        "result = returns.resample('YE').apply(lambda s: (1+s).prod()-1)", *data)
    assert out["result"]["type"] == "series"
    assert out["advisory"] is True          # always labelled advisory
    assert out["code"]                       # code surfaced for audit


def test_scalar_and_dataframe_results(data):
    out = sandbox.run_analysis("result = float(returns.std() * (252 ** 0.5))", *data)
    assert out["result"]["type"] == "scalar"
    assert out["result"]["data"] > 0

    out2 = sandbox.run_analysis("result = df.head(3)", *data)
    assert out2["result"]["type"] == "dataframe"


def test_large_result_is_truncated(data):
    out = sandbox.run_analysis("result = df", *data)
    assert out["result"]["truncated"] is True
    assert len(out["result"]["data"]) <= 50


def test_missing_result_variable_is_a_clear_error(data):
    with pytest.raises(SandboxError, match="no `result`"):
        sandbox.run_analysis("x = 1 + 1", *data)


# --- it refuses to escape ----------------------------------------------------
@pytest.mark.parametrize("code, why", [
    ("import os\nresult = os.listdir('/')",                "import"),
    ("from pathlib import Path\nresult = 1",               "import-from"),
    ("result = open('/etc/passwd').read()",                "file read"),
    ("result = eval('1+1')",                               "eval"),
    ("result = exec('x=1')",                               "exec"),
    ("result = __import__('os').system('echo hi')",        "dunder import"),
    ("result = (1).__class__.__bases__",                   "class escape"),
    ("result = returns.__class__.__mro__",                 "mro escape"),
    ("result = [].__class__.__subclasses__()",             "subclasses escape"),
    ("def f():\n    return 1\nresult = f()",               "function def"),
    ("class C:\n    pass\nresult = 1",                     "class def"),
    ("result = df.to_csv('/tmp/leak.csv')",                "file write"),
    ("result = pd.read_csv('/etc/passwd')",                "file read via pandas"),
    ("global x\nresult = 1",                               "global"),
    ("result = compile('1','','eval')",                    "compile"),
])
def test_escape_attempts_are_blocked(data, code, why):
    with pytest.raises(SandboxError):
        sandbox.run_analysis(code, *data)


def test_infinite_loop_hits_the_timeout(data):
    with pytest.raises(SandboxError, match="time limit"):
        sandbox.run_analysis("x = 0\nfor i in range(10**12):\n    x = x + 1\nresult = x",
                             *data, timeout_s=2.0)


def test_no_network_names_available(data):
    # httpx / socket / requests simply do not exist in the namespace
    with pytest.raises(SandboxError):
        sandbox.run_analysis("result = httpx.get('http://x')", *data)


def test_syntax_error_is_reported_cleanly(data):
    with pytest.raises(SandboxError, match="Syntax error"):
        sandbox.run_analysis("result = (1 +", *data)
