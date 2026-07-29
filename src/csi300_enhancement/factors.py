from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES


def _cross_sectional_winsor_zscore(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 10:
        return pd.Series(np.nan, index=series.index)
    lo, hi = valid.quantile([lower, upper])
    clipped = series.clip(lo, hi)
    std = clipped.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (clipped - clipped.mean()) / std


def build_factor_panel(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    df = prices.sort_values(["Ticker", "Date"]).copy()
    g = df.groupby("Ticker", group_keys=False)
    df["Return1D"] = g["Close"].pct_change(fill_method=None)
    df["ForwardReturn1D"] = g["Close"].shift(-1).div(df["Close"]).sub(1.0)
    df[f"ForwardReturn{horizon}D"] = g["Close"].shift(-horizon).div(df["Close"]).sub(1.0)
    df[f"TargetDate{horizon}D"] = g["Date"].shift(-horizon)

    df["momentum_20_5_raw"] = g["Close"].shift(5).div(g["Close"].shift(20)).sub(1.0)
    df["momentum_60_5_raw"] = g["Close"].shift(5).div(g["Close"].shift(60)).sub(1.0)
    df["reversal_5_raw"] = -g["Close"].pct_change(5, fill_method=None)
    df["volatility_20_raw"] = -g["Return1D"].rolling(20, min_periods=15).std().reset_index(level=0, drop=True)
    downside = df["Return1D"].clip(upper=0.0)
    df["downside_volatility_20_raw"] = -downside.groupby(df["Ticker"]).rolling(20, min_periods=15).std().reset_index(level=0, drop=True)
    rolling_max = g["Close"].rolling(60, min_periods=40).max().reset_index(level=0, drop=True)
    df["drawdown_60_raw"] = df["Close"].div(rolling_max).sub(1.0)

    raw_map = {name: f"{name}_raw" for name in FEATURES}
    for factor, raw in raw_map.items():
        df[factor] = df.groupby("Date", group_keys=False)[raw].transform(_cross_sectional_winsor_zscore)
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def rebalance_dates_from_panel(panel: pd.DataFrame, weekday: int = 4) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(sorted(panel["Date"].unique()))
    calendar = pd.DataFrame(index=dates)
    calendar["period"] = calendar.index.to_period("W-FRI")
    # Last available trading day in each Friday-ended week.
    return pd.DatetimeIndex(calendar.groupby("period").apply(lambda x: x.index.max()).values)
