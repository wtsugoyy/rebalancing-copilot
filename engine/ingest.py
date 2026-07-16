"""Robust ingestion — behaves like a data analyst: cleans and sanitises real-world
files, auto-detects whether the fund data is NAV prices or daily returns, and reports
exactly what it did (transparent, never silent). Failures on genuinely unusable data
still raise typed errors.

Handles, without failing: BOM headers, quoted fields, whitespace, blank/footer rows,
mixed/ambiguous date formats (M/D/YYYY vs D/M/YYYY), descending date order, extra
columns, thousands separators.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from engine.universe import resolve_isins
from engine.validate import InsufficientHistoryError, PriceDataError


@dataclass(frozen=True)
class PriceBundle:
    dates: list[date]
    fund_returns: dict[str, list[float]]   # simple daily returns
    isins: list[str]
    n_obs: int
    mode: str = "returns"                   # "returns" (used as-is) or "nav" (differenced)
    report: dict = field(default_factory=dict)  # human-readable cleaning summary


# --- reading -----------------------------------------------------------------
def _read_table(source) -> pd.DataFrame:
    name = getattr(source, "name", str(source))
    suffix = Path(name).suffix.lower()
    try:
        if suffix in (".xlsx", ".xlsm", ".xls"):
            return pd.read_excel(source, engine="openpyxl")
        # utf-8-sig strips a BOM; skip fully-blank lines
        if hasattr(source, "read"):
            data = source.read()
            if isinstance(data, bytes):
                source = io.StringIO(data.decode("utf-8-sig"))
            else:
                source = io.StringIO(data)
            return pd.read_csv(source, skip_blank_lines=True)
        return pd.read_csv(source, encoding="utf-8-sig", skip_blank_lines=True)
    except Exception as exc:  # noqa: BLE001
        raise PriceDataError(f"Could not parse file '{name}': {exc}") from exc


def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df


def _find_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).strip().lower() in ("date", "dates", "asofdate", "as_of_date", "as of date"):
            return col
    first = df.columns[0]
    if pd.to_datetime(df[first], errors="coerce").notna().mean() > 0.5:
        return first
    raise PriceDataError("No date column found (expected a 'Date' column or a parseable "
                         "first column).")


def _parse_dates(series: pd.Series) -> tuple[pd.Series, str]:
    """Parse dates robustly; retry day-first if month-first fails a lot. Returns
    (parsed, convention_used)."""
    mf = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=False)
    if mf.isna().mean() <= 0.2:
        return mf, "month-first"
    df_first = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    if df_first.notna().sum() > mf.notna().sum():
        return df_first, "day-first"
    return mf, "month-first"


# --- NAV vs returns detection ------------------------------------------------
def _detect_mode(values: pd.DataFrame) -> str:
    arr = values.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "returns"
    has_negative = bool((finite < 0).any())
    median_abs = float(np.median(np.abs(finite)))
    # returns are small and signed; NAV prices are strictly positive and larger
    if has_negative or median_abs < 0.05:
        return "returns"
    return "nav"


# --- main --------------------------------------------------------------------
def load_nav(source, required_isins: list[str] | None = None) -> PriceBundle:
    """Load a fund data file (NAV prices OR daily returns — auto-detected) and return
    aligned simple daily returns plus a cleaning report."""
    df = _clean_headers(_read_table(source))
    date_col = _find_date_column(df)

    fund_cols = [c for c in df.columns if c != date_col]
    resolve_isins([str(c) for c in fund_cols])  # unknown ISIN -> UniverseMismatchError

    use_cols = required_isins if required_isins else [str(c) for c in fund_cols]
    if required_isins:
        missing = [i for i in required_isins if i not in [str(c) for c in fund_cols]]
        if missing:
            raise PriceDataError(f"File missing required fund column(s): {missing}",
                                 offenders=missing)

    df = df[[date_col] + use_cols].copy()

    report: dict = {}
    n_start = len(df)

    # 1) coerce fund values to numeric (strip thousands separators / stray chars)
    for c in use_cols:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce")

    # 2) drop fully-blank rows (footer / spacer rows) — junk, not an error
    blank_mask = df[use_cols].isna().all(axis=1)
    n_blank = int(blank_mask.sum())
    df = df[~blank_mask]
    if n_blank:
        report["blank_rows_dropped"] = n_blank

    # 3) parse dates; drop rows with an unparseable date (report, don't fail)
    df[date_col], date_conv = _parse_dates(df[date_col])
    bad_dates = int(df[date_col].isna().sum())
    if bad_dates:
        df = df.dropna(subset=[date_col])
        report["unparseable_date_rows_dropped"] = bad_dates
    report["date_convention"] = date_conv

    df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

    # 4) detect NAV vs returns
    mode = _detect_mode(df[use_cols])
    report["mode"] = mode

    if mode == "nav":
        if (df[use_cols] <= 0).any().any():
            raise PriceDataError("Detected NAV prices but some values are <= 0. If this file "
                                 "contains returns, they should be signed/small.")
        df = df.dropna(subset=use_cols)
        rets = df[use_cols].astype(float).pct_change().iloc[1:]
        ret_dates = list(df[date_col].iloc[1:])
    else:  # returns
        # A blank return cell counts as 0 in the weighted sum — match the source model.
        n_filled = int(df[use_cols].isna().sum().sum())
        if n_filled:
            report["blank_return_cells_filled_zero"] = n_filled
        rets = df[use_cols].fillna(0.0).astype(float)
        ret_dates = list(df[date_col])

    ret_dates = [d.date() if isinstance(d, (pd.Timestamp, datetime)) else d for d in ret_dates]
    report["rows_in"] = n_start
    report["observations_out"] = len(ret_dates)

    if len(ret_dates) < config.MIN_OBS:
        raise InsufficientHistoryError(
            f"Only {len(ret_dates)} usable observations after cleaning "
            f"(dropped {n_blank} blank + {bad_dates} bad-date rows); need >= {config.MIN_OBS}.",
            report=report)

    fund_returns = {isin: rets[isin].tolist() for isin in use_cols}

    from engine.obs import get_logger
    get_logger().info("ingest mode=%s funds=%d obs=%d report=%s",
                      mode, len(use_cols), len(ret_dates), report)
    return PriceBundle(dates=ret_dates, fund_returns=fund_returns, isins=list(use_cols),
                       n_obs=len(ret_dates), mode=mode, report=report)


# --- yields ------------------------------------------------------------------
_YIELD_PREFERRED = ("yield", "price", "close", "rate", "last")


def load_yields(source) -> list[tuple[date, float]]:
    """Parse a yield file -> sorted [(date, yield_pct)]. Robust to BOM, quotes,
    a 'Price'/'Yield'/'Close' column among several, %-suffixed values, any order."""
    df = _clean_headers(_read_table(source))
    date_col = _find_date_column(df)
    candidates = [c for c in df.columns if c != date_col]
    if not candidates:
        raise PriceDataError("Yield file has no value column.")

    # prefer a sensibly-named yield column, else the first numeric one
    ycol = None
    for pref in _YIELD_PREFERRED:
        for c in candidates:
            if pref in str(c).strip().lower():
                ycol = c
                break
        if ycol:
            break
    if ycol is None:
        ycol = candidates[0]

    df = df[[date_col, ycol]].copy()
    df[date_col], _ = _parse_dates(df[date_col])
    df[ycol] = pd.to_numeric(
        df[ycol].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
        .str.strip(), errors="coerce")
    df = df.dropna().sort_values(date_col)
    if df.empty:
        raise PriceDataError("Yield file produced no usable (date, yield) rows.")
    return [(d.date() if isinstance(d, (pd.Timestamp, datetime)) else d, float(y))
            for d, y in zip(df[date_col], df[ycol])]
