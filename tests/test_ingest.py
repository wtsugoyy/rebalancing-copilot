"""TC-9: ingest + validation matrix. Every §8 rule must fire with the correct typed
error; the NAV->simple-returns path must be numerically correct."""
import io

import numpy as np
import pandas as pd
import pytest

import config
from engine import ingest
from engine.validate import (InsufficientHistoryError, PriceDataError,
                             UniverseMismatchError, WeightError)
from engine import universe

ISIN_A, ISIN_B = config.FUND_UNIVERSE[0], config.FUND_UNIVERSE[1]


def _nav_csv(dates, a, b, cols=(ISIN_A, ISIN_B)):
    df = pd.DataFrame({"Date": dates, cols[0]: a, cols[1]: b})
    buf = io.BytesIO(df.to_csv(index=False).encode())
    buf.name = "nav.csv"
    return buf


def test_nav_to_simple_returns_correct():
    dates = pd.date_range("2024-01-01", periods=config.MIN_OBS + 5).astype(str)
    a = np.linspace(1.0, 1.1, len(dates))
    b = np.linspace(2.0, 2.2, len(dates))
    bundle = ingest.load_nav(_nav_csv(dates, a, b))
    assert bundle.n_obs == len(dates) - 1
    # first return of A = a[1]/a[0]-1
    assert abs(bundle.fund_returns[ISIN_A][0] - (a[1] / a[0] - 1)) < 1e-12


def test_unknown_isin_rejected():
    dates = pd.date_range("2024-01-01", periods=config.MIN_OBS + 5).astype(str)
    ones = np.ones(len(dates))
    with pytest.raises(UniverseMismatchError):
        ingest.load_nav(_nav_csv(dates, ones, ones, cols=(ISIN_A, "NOT_AN_ISIN")))


def test_non_positive_nav_rejected():
    dates = pd.date_range("2024-01-01", periods=config.MIN_OBS + 5).astype(str)
    a = np.ones(len(dates)); a[3] = 0.0
    with pytest.raises(PriceDataError):
        ingest.load_nav(_nav_csv(dates, a, np.ones(len(dates))))


def test_insufficient_history_rejected():
    dates = pd.date_range("2024-01-01", periods=10).astype(str)
    ones = np.ones(10)
    with pytest.raises(InsufficientHistoryError):
        ingest.load_nav(_nav_csv(dates, ones, ones))


def test_stray_bad_date_is_cleaned_not_fatal():
    # analyst behaviour: a single junk date among enough good rows is dropped + reported
    good = list(pd.date_range("2024-01-01", periods=config.MIN_OBS + 4).astype(str))
    dates = good + ["not-a-date"]
    a = np.linspace(1.0, 1.1, len(dates)); b = np.linspace(2.0, 2.2, len(dates))
    bundle = ingest.load_nav(_nav_csv(dates, a, b))
    assert bundle.report.get("unparseable_date_rows_dropped") == 1
    assert bundle.n_obs >= config.MIN_OBS


def test_all_bad_dates_is_fatal():
    dates = ["junk"] * (config.MIN_OBS + 4)
    ones = np.ones(len(dates))
    with pytest.raises((InsufficientHistoryError, PriceDataError)):
        ingest.load_nav(_nav_csv(dates, ones, ones))


def test_returns_file_autodetected():
    # signed small values => returns mode, used directly (no differencing)
    dates = pd.date_range("2024-01-01", periods=config.MIN_OBS + 2).astype(str)
    a = np.random.default_rng(0).normal(0, 0.01, len(dates))
    b = np.random.default_rng(1).normal(0, 0.01, len(dates))
    bundle = ingest.load_nav(_nav_csv(dates, a, b))
    assert bundle.mode == "returns"
    assert bundle.n_obs == len(dates)  # no row lost to differencing


def test_blank_footer_rows_dropped():
    dates = list(pd.date_range("2024-01-01", periods=config.MIN_OBS + 2).astype(str)) + ["", ""]
    a = list(np.linspace(1.0, 1.1, config.MIN_OBS + 2)) + [None, None]
    b = list(np.linspace(2.0, 2.2, config.MIN_OBS + 2)) + [None, None]
    df = pd.DataFrame({"Date": dates, ISIN_A: a, ISIN_B: b})
    buf = io.BytesIO(df.to_csv(index=False).encode()); buf.name = "nav.csv"
    bundle = ingest.load_nav(buf)
    assert bundle.report.get("blank_rows_dropped") == 2


def test_weight_normalization_percent_and_sum():
    # accepts percents, normalizes >1.5 by /100
    w = universe.normalize_weights({ISIN_A: 60, ISIN_B: 40})
    assert abs(w[ISIN_A] - 0.6) < 1e-12 and abs(sum(w.values()) - 1.0) < 1e-12


def test_weight_bad_sum_rejected():
    with pytest.raises(WeightError):
        universe.normalize_weights({ISIN_A: 0.6, ISIN_B: 0.5})


def test_weight_unknown_isin_rejected():
    with pytest.raises(WeightError):
        universe.normalize_weights({"NOPE": 1.0})


def test_yield_parsing_sorted():
    df = pd.DataFrame({"Date": ["2024-01-03", "2024-01-01"], "Yield": [3.2, 3.1]})
    buf = io.BytesIO(df.to_csv(index=False).encode()); buf.name = "y.csv"
    ys = ingest.load_yields(buf)
    assert ys[0][0].isoformat() == "2024-01-01" and ys[0][1] == 3.1
