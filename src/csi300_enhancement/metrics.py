from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown(nav: pd.Series) -> pd.Series:
    return nav.div(nav.cummax()).sub(1.0)


def performance_metrics(
    strategy: pd.DataFrame,
    benchmark_returns: pd.Series,
    name: str,
    periods_per_year: int = 252,
) -> dict:
    net = strategy["NetReturn"].fillna(0.0)
    bench = benchmark_returns.reindex(net.index).fillna(0.0)
    active = net - bench
    n = len(net)
    years = n / periods_per_year
    nav = (1.0 + net).cumprod()
    bench_nav = (1.0 + bench).cumprod()
    total_return = nav.iloc[-1] - 1.0
    ann_return = nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and nav.iloc[-1] > 0 else np.nan
    bench_ann = bench_nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and bench_nav.iloc[-1] > 0 else np.nan
    ann_vol = net.std(ddof=1) * np.sqrt(periods_per_year)
    tracking_error = active.std(ddof=1) * np.sqrt(periods_per_year)
    ann_active_arithmetic = active.mean() * periods_per_year
    return {
        "Strategy": name,
        "Start": net.index.min(), "End": net.index.max(), "Observations": n,
        "TotalReturn": total_return, "AnnualizedReturn": ann_return,
        "AnnualizedVolatility": ann_vol,
        "SharpeRatio": ann_return / ann_vol if ann_vol > 0 else np.nan,
        "MaxDrawdown": drawdown(nav).min(),
        "BenchmarkAnnualizedReturn": bench_ann,
        "AnnualizedActiveReturn": ann_return - bench_ann,
        "ArithmeticAnnualizedActiveReturn": ann_active_arithmetic,
        "TrackingError": tracking_error,
        "InformationRatio": ann_active_arithmetic / tracking_error if tracking_error > 0 else np.nan,
        "AnnualOneWayTurnover": strategy["OneWayTurnover"].sum() / years if years > 0 else np.nan,
        "TotalTradingCost": strategy["TradingCost"].sum(),
        "PositiveActiveDayRate": (active > 0).mean(),
    }


def annual_returns(return_series: pd.Series) -> pd.Series:
    return return_series.groupby(return_series.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)

