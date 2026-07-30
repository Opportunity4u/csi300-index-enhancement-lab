from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLUMNS = {"Date", "Ticker", "Close"}
REQUIRED_WEIGHT_COLUMNS = {"AsOfDate", "Ticker", "BenchmarkWeight"}
REQUIRED_CONSTITUENT_COLUMNS = {"AsOfDate", "Ticker"}


def _normalize_ticker(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_price_panel(path: Path) -> pd.DataFrame:
    """Load a normalized point-in-time-safe price panel.

    Required columns are ``Date``, ``Ticker`` and ``Close``. Optional columns
    such as ``Open``, ``Volume`` and ``Amount`` are preserved.
    """
    df = pd.read_csv(path, parse_dates=["Date"], dtype={"Ticker": str})
    _require_columns(df, REQUIRED_PRICE_COLUMNS, "price panel")
    df["Ticker"] = _normalize_ticker(df["Ticker"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Close"])
    if (df["Close"] <= 0).any():
        raise ValueError("price panel contains non-positive Close values")
    if df.duplicated(["Date", "Ticker"]).any():
        raise ValueError("price panel contains duplicate Date/Ticker rows")
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def load_benchmark_snapshot(path: Path) -> pd.DataFrame:
    """Load one or more benchmark snapshots.

    The illustrative pipeline uses the latest snapshot to construct a fixed-share
    historical proxy. Institutional use should replace this adapter with genuine
    point-in-time membership and weights.
    """
    df = pd.read_csv(path, parse_dates=["AsOfDate"], dtype={"Ticker": str})
    _require_columns(df, REQUIRED_WEIGHT_COLUMNS, "benchmark weights")
    df["Ticker"] = _normalize_ticker(df["Ticker"])
    df["BenchmarkWeight"] = pd.to_numeric(df["BenchmarkWeight"], errors="coerce")
    df = df.dropna(subset=["AsOfDate", "Ticker", "BenchmarkWeight"])
    if (df["BenchmarkWeight"] < 0).any():
        raise ValueError("benchmark weights must be non-negative")
    totals = df.groupby("AsOfDate")["BenchmarkWeight"].transform("sum")
    if (totals <= 0).any():
        raise ValueError("benchmark snapshot has a non-positive total weight")
    df["BenchmarkWeight"] = df["BenchmarkWeight"] / totals
    return df.sort_values(["AsOfDate", "Ticker"]).reset_index(drop=True)


def load_constituents(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["AsOfDate"], dtype={"Ticker": str})
    _require_columns(df, REQUIRED_CONSTITUENT_COLUMNS, "constituents")
    df["Ticker"] = _normalize_ticker(df["Ticker"])
    return df.sort_values(["AsOfDate", "Ticker"]).reset_index(drop=True)


def make_benchmark_proxy_weights(
    prices: pd.DataFrame,
    official_weights: pd.DataFrame,
    anchor_date: pd.Timestamp,
) -> pd.DataFrame:
    """Back-cast a transparent fixed-share benchmark proxy.

    Share units are calibrated to one benchmark snapshot and then held constant.
    This is deliberately *not* presented as official point-in-time index history.
    """
    close = prices.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    anchor_rows = close.loc[close.index <= anchor_date]
    if anchor_rows.empty:
        raise ValueError("no prices on or before benchmark anchor date")
    anchor_px = anchor_rows.ffill().iloc[-1]
    weights = (
        official_weights.set_index("Ticker")["BenchmarkWeight"]
        .reindex(close.columns)
        .fillna(0.0)
    )
    shares_proxy = weights.div(anchor_px).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = close.ffill().mul(shares_proxy, axis=1)
    result = values.div(values.sum(axis=1), axis=0).fillna(0.0)
    result.index.name = "Date"
    return result
