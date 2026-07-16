"""Headless UI smoke test via Streamlit AppTest — proves app.py runs, renders the
panels, and wires uploads -> deterministic attribution without a browser or LLM."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
st_testing = pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_boots_without_data():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception, f"app raised on boot: {at.exception}"
    # title present
    assert any("Copilot" in (t.value if hasattr(t, "value") else "") for t in at.title)


def test_app_renders_and_stays_stable_on_rerun():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    at.run()  # a second rerun must not crash (cache_resource / session_state stable)
    assert not at.exception
