import json
import sys
from datetime import date
from pathlib import Path

import pytest

# make repo root importable (config, engine, store)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE = Path(__file__).parent / "fixtures" / "golden.json"


@pytest.fixture(scope="session")
def golden():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["_dates"] = [date.fromisoformat(d) for d in data["dates"]]
    data["_yields"] = [(date.fromisoformat(y["date"]), y["yield_pct"]) for y in data["yields"]]
    return data


@pytest.fixture
def seeded_ctx(tmp_path, golden):
    """ToolContext with every portfolio seeded — used by the LangGraph harness tests."""
    import config
    from agent.tools import AppData, ToolContext
    from store import history

    conn = history.connect(str(tmp_path / "graph.db"))
    history.init_db(conn)
    dates = golden["_dates"]
    for code in config.PORTFOLIO_CODES:
        wmap = dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS[code]))
        history.save_snapshot(conn, code, dates[0].isoformat(), wmap, source="seed")
    data = AppData(dates=dates, fund_returns=golden["fund_returns"],
                   yields=golden["_yields"], isins=golden["isins"])
    yield ToolContext(conn=conn, data=data)
    conn.close()
