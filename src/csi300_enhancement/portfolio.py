from __future__ import annotations

import numpy as np
import pandas as pd


def _capped_allocation(raw: pd.Series, budget: float, cap: float) -> pd.Series:
    raw = raw.clip(lower=0.0).fillna(0.0)
    result = pd.Series(0.0, index=raw.index)
    remaining = budget
    eligible = raw[raw > 0].index
    for _ in range(20):
        if remaining <= 1e-12 or len(eligible) == 0:
            break
        weights = raw.loc[eligible] / raw.loc[eligible].sum()
        proposed = weights * remaining
        room = cap - result.loc[eligible]
        add = pd.concat([proposed, room], axis=1).min(axis=1).clip(lower=0.0)
        result.loc[eligible] += add
        remaining = budget - result.sum()
        eligible = result[(result < cap - 1e-12) & (raw > 0)].index
    return result


def construct_active_target(
    predicted: pd.Series,
    benchmark_weight: pd.Series,
    trailing_returns: pd.DataFrame | None,
    active_budget: float = 0.10,
    active_cap: float = 0.0075,
    te_cap: float = 0.04,
) -> tuple[pd.Series, dict]:
    idx = benchmark_weight.index.intersection(predicted.index)
    bench = benchmark_weight.reindex(idx).fillna(0.0)
    bench /= bench.sum()
    score = predicted.reindex(idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    score = score.clip(score.quantile(0.01), score.quantile(0.99))
    score = (score - score.mean()) / (score.std(ddof=0) or 1.0)

    positive = _capped_allocation(score.clip(lower=0.0), active_budget, active_cap)
    # Underweights are capped by both active cap and the available benchmark weight.
    sell_cap = pd.concat([pd.Series(active_cap, index=idx), bench], axis=1).min(axis=1)
    negative_raw = (-score).clip(lower=0.0) * bench.pow(0.5)
    negative = _capped_allocation(negative_raw, positive.sum(), active_cap)
    negative = pd.concat([negative, sell_cap], axis=1).min(axis=1)
    if negative.sum() > 0:
        matched = min(positive.sum(), negative.sum())
        positive *= matched / positive.sum()
        negative *= matched / negative.sum()
    active = positive - negative

    ex_ante_te = np.nan
    te_scale = 1.0
    if trailing_returns is not None and len(trailing_returns) >= 60:
        cov = trailing_returns.reindex(columns=idx).fillna(0.0).cov().values
        a = active.values
        variance = float(a @ cov @ a) * 252.0
        ex_ante_te = float(np.sqrt(max(variance, 0.0)))
        if ex_ante_te > te_cap and ex_ante_te > 0:
            te_scale = te_cap / ex_ante_te
            active *= te_scale
            ex_ante_te = te_cap

    target = (bench + active).clip(lower=0.0)
    target /= target.sum()
    active = target - bench
    diagnostics = {
        "ActiveLong": float(active.clip(lower=0).sum()),
        "ActiveShort": float(-active.clip(upper=0).sum()),
        "ActiveL1": float(active.abs().sum()),
        "ExAnteTrackingError": ex_ante_te,
        "TrackingErrorScale": te_scale,
        "MaxActiveWeight": float(active.abs().max()),
    }
    return target, diagnostics


def construct_topk_equal(predicted: pd.Series, k: int = 50) -> pd.Series:
    selected = predicted.dropna().nlargest(min(k, predicted.notna().sum())).index
    target = pd.Series(0.0, index=predicted.index)
    if len(selected):
        target.loc[selected] = 1.0 / len(selected)
    return target


def build_target_weight_tables(
    predictions: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    covariance_lookback: int,
    active_budget: float,
    active_cap: float,
    te_cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enhanced: list[pd.Series] = []
    topk: list[pd.Series] = []
    diag: list[dict] = []
    for decision_date, frame in predictions.groupby("DecisionDate"):
        if decision_date not in benchmark_weights.index:
            continue
        pred = frame.set_index("Ticker")["PredictedReturn"]
        bench = benchmark_weights.loc[decision_date]
        trailing = daily_returns.loc[daily_returns.index < decision_date].tail(covariance_lookback)
        target, info = construct_active_target(pred, bench, trailing, active_budget, active_cap, te_cap)
        target.name = decision_date
        enhanced.append(target)
        top = construct_topk_equal(pred.reindex(bench.index), 50)
        top.name = decision_date
        topk.append(top)
        diag.append({"DecisionDate": decision_date, **info})
    return pd.DataFrame(enhanced).fillna(0.0), pd.DataFrame(topk).fillna(0.0), pd.DataFrame(diag)

