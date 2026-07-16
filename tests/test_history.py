"""TC-6: history store — current/previous ordering, append-only re-run (W5),
chained timeline, and atomic writes."""
from datetime import date

import pytest

import config
from engine.validate import NotFoundError
from store import history


@pytest.fixture
def conn(tmp_path):
    c = history.connect(str(tmp_path / "t.db"))
    history.init_db(c)
    yield c
    c.close()


def _w(a=0.6, b=0.4):
    return {config.FUND_UNIVERSE[0]: a, config.FUND_UNIVERSE[1]: b}


def test_current_is_latest_effective_date(conn):
    history.save_snapshot(conn, "BAL", "2024-01-01", _w())
    history.save_snapshot(conn, "BAL", "2024-03-01", _w(0.5, 0.5))
    history.save_snapshot(conn, "BAL", "2024-02-01", _w(0.7, 0.3))
    cur = history.get_current(conn, "BAL")
    assert cur.effective_date == "2024-03-01"
    prev = history.get_previous(conn, "BAL")
    assert prev.effective_date == "2024-02-01"


def test_rerun_same_date_appends_new_row_not_overwrite(conn):
    history.save_snapshot(conn, "BAL", "2024-01-01", _w(0.6, 0.4))
    history.save_snapshot(conn, "BAL", "2024-01-01", _w(0.9, 0.1))  # correction
    hist = history.list_history(conn, "BAL")
    assert len(hist) == 2  # both versions retained (append-only)
    # current resolves to the newest created_at for that date
    assert history.get_current(conn, "BAL").weights[config.FUND_UNIVERSE[0]] == 0.9


def test_no_snapshot_raises_notfound(conn):
    with pytest.raises(NotFoundError):
        history.get_current(conn, "ADV")


def test_chained_timeline_is_date_ordered(conn):
    history.save_snapshot(conn, "ADV", "2024-05-01", _w(0.5, 0.5))
    history.save_snapshot(conn, "ADV", "2024-01-01", _w(0.6, 0.4))
    tl = history.chained_timeline(conn, "ADV")
    assert [d for d, _ in tl] == [date(2024, 1, 1), date(2024, 5, 1)]


def test_metrics_persisted_and_hydrated(conn):
    m = {"total_return": 0.1, "ann_return": 0.03, "ann_vol": 0.05,
         "max_drawdown": -0.02, "sharpe": 1.1, "sortino": 1.5, "avg_rf": 0.033}
    history.save_snapshot(conn, "BAL", "2024-01-01", _w(), metrics=m)
    got = history.get_current(conn, "BAL").metrics
    assert got["sharpe"] == 1.1 and got["avg_rf"] == 0.033


def test_candidate_preview_not_counted_as_current(conn):
    history.save_snapshot(conn, "BAL", "2024-01-01", _w(), source="user_committed")
    history.save_snapshot(conn, "BAL", "2024-06-01", _w(0.2, 0.8), source="candidate_preview")
    # preview has a later date but must not become "current"
    assert history.get_current(conn, "BAL").effective_date == "2024-01-01"
