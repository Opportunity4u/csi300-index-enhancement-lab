from __future__ import annotations

import numpy as np
import pandas as pd


def _spearman_by_date(frame: pd.DataFrame, factor: str, target: str) -> pd.Series:
    return frame.groupby("Date").apply(
        lambda x: x[[factor, target]].corr(method="spearman").iloc[0, 1]
        if x[[factor, target]].dropna().shape[0] >= 20 else np.nan,
        include_groups=False,
    ).dropna()


def factor_ic_summary(panel: pd.DataFrame, factors: list[str], target: str, oos_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for factor in factors:
        ic = _spearman_by_date(panel, factor, target)
        daily_rows.append(pd.DataFrame({"Date": ic.index, "Factor": factor, "RankIC": ic.values}))
        for label, subset in {
            "full": ic,
            "in_sample": ic[ic.index < pd.Timestamp(oos_start)],
            "out_of_sample": ic[ic.index >= pd.Timestamp(oos_start)],
        }.items():
            mean = subset.mean()
            std = subset.std(ddof=1)
            summary_rows.append({
                "Factor": factor, "Period": label, "Observations": len(subset),
                "MeanRankIC": mean, "RankICStd": std,
                "ICIR": mean / std if std and np.isfinite(std) else np.nan,
                "PositiveICRate": (subset > 0).mean() if len(subset) else np.nan,
            })
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    return pd.DataFrame(summary_rows), daily


def factor_decay(panel: pd.DataFrame, factors: list[str], horizons: tuple[int, ...] = (1, 5, 10, 20)) -> pd.DataFrame:
    work = panel.sort_values(["Ticker", "Date"]).copy()
    g = work.groupby("Ticker", group_keys=False)
    rows: list[dict] = []
    for horizon in horizons:
        target = g["Close"].shift(-horizon).div(work["Close"]).sub(1.0)
        work[f"_fwd_{horizon}"] = target
        for factor in factors:
            ic = _spearman_by_date(work, factor, f"_fwd_{horizon}")
            rows.append({
                "Factor": factor, "HorizonDays": horizon, "MeanRankIC": ic.mean(),
                "ICIR": ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) else np.nan,
                "Observations": len(ic),
            })
    return pd.DataFrame(rows)


def quantile_returns(panel: pd.DataFrame, factors: list[str], target: str, quantiles: int = 5) -> pd.DataFrame:
    rows: list[dict] = []
    for factor in factors:
        subset = panel[["Date", factor, target]].dropna().copy()
        subset["Quantile"] = subset.groupby("Date")[factor].transform(
            lambda x: pd.qcut(x.rank(method="first"), quantiles, labels=False, duplicates="drop") + 1
        )
        grouped = subset.groupby("Quantile")[target].agg(["mean", "std", "count"]).reset_index()
        for row in grouped.itertuples(index=False):
            rows.append({"Factor": factor, "Quantile": int(row.Quantile), "MeanForwardReturn": row.mean,
                         "StdForwardReturn": row.std, "Observations": int(row.count)})
        qmeans = grouped.set_index("Quantile")["mean"]
        if 1 in qmeans.index and quantiles in qmeans.index:
            rows.append({"Factor": factor, "Quantile": "Q5-Q1", "MeanForwardReturn": qmeans[quantiles] - qmeans[1],
                         "StdForwardReturn": np.nan, "Observations": int(grouped["count"].sum())})
    return pd.DataFrame(rows)

