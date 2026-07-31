from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame


def _matched_trade(delta: pd.Series) -> pd.Series:
    buys = delta.clip(lower=0.0)
    sells = -delta.clip(upper=0.0)
    matched = min(float(buys.sum()), float(sells.sum()))
    if matched <= 0:
        return pd.Series(0.0, index=delta.index)
    return buys * (matched / buys.sum()) - sells * (matched / sells.sum())


def constrain_rebalance(
    current_weights: pd.Series,
    target_weights: pd.Series,
    min_trade_weight: float = 0.0,
    one_way_turnover_cap: float | None = None,
) -> tuple[pd.Series, pd.Series, dict]:
    """Convert an ideal long-only target into a cash-neutral executable target."""
    idx = current_weights.index.union(target_weights.index)
    current = current_weights.reindex(idx).fillna(0.0).clip(lower=0.0)
    target = target_weights.reindex(idx).fillna(0.0).clip(lower=0.0)
    if current.sum() <= 0 or target.sum() <= 0:
        raise ValueError("current and target weights must each have positive total weight")
    current /= current.sum()
    target /= target.sum()
    raw_delta = target - current
    thresholded = raw_delta.where(raw_delta.abs() >= min_trade_weight, 0.0)
    delta = _matched_trade(thresholded)
    pre_cap_turnover = 0.5 * float(delta.abs().sum())
    if one_way_turnover_cap is not None and pre_cap_turnover > one_way_turnover_cap > 0:
        delta *= one_way_turnover_cap / pre_cap_turnover
    actual_target = (current + delta).clip(lower=0.0)
    actual_target /= actual_target.sum()
    diagnostics = {
        "RawOneWayTurnover": 0.5 * float(raw_delta.abs().sum()),
        "ThresholdedOneWayTurnover": pre_cap_turnover,
        "ExecutedOneWayTurnover": 0.5 * float(delta.abs().sum()),
    }
    return actual_target, delta, diagnostics


def map_decisions_to_execution_dates(targets: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    dates = pd.DatetimeIndex(trading_dates)
    for decision_date, row in targets.iterrows():
        future = dates[dates > decision_date]
        if len(future):
            copy = row.copy()
            copy.name = future[0]
            rows.append(copy)
    if not rows:
        return pd.DataFrame(columns=targets.columns)
    out = pd.DataFrame(rows)
    out.index.name = "ExecutionDate"
    return out


def run_backtest(
    targets_by_decision: pd.DataFrame,
    asset_returns: pd.DataFrame,
    amount: pd.DataFrame,
    initial_weights: pd.Series,
    min_trade_weight: float = 0.0,
    one_way_turnover_cap: float | None = None,
    portfolio_value: float = 10_000_000.0,
    max_adv_participation: float = 1.0,
    buy_cost_bps: float = 0.0,
    sell_cost_bps: float = 0.0,
    impact_coefficient: float = 0.0,
) -> BacktestResult:
    dates = asset_returns.index
    tickers = asset_returns.columns
    targets = map_decisions_to_execution_dates(targets_by_decision, dates).reindex(columns=tickers).fillna(0.0)
    weights = initial_weights.reindex(tickers).fillna(0.0)
    weights /= weights.sum()
    nav = 1.0
    daily_rows: list[dict] = []
    weight_rows: list[pd.Series] = []
    trade_rows: list[dict] = []
    adv20 = amount.rolling(20, min_periods=5).mean()

    for date in dates:
        r = asset_returns.loc[date].fillna(0.0)
        gross_return = float((weights * r).sum())
        drifted = weights * (1.0 + r)
        drifted = drifted / drifted.sum() if drifted.sum() > 0 else weights
        traded_l1 = 0.0
        one_way_turnover = 0.0
        cost = 0.0

        if date in targets.index:
            target = targets.loc[date]
            target = target / target.sum() if target.sum() > 0 else drifted
            delta = target - drifted
            delta = delta.where(delta.abs() >= min_trade_weight, 0.0)
            delta = _matched_trade(delta)

            if one_way_turnover_cap is not None:
                current_one_way = 0.5 * float(delta.abs().sum())
                if current_one_way > one_way_turnover_cap > 0:
                    delta *= one_way_turnover_cap / current_one_way

            available_adv = adv20.loc[date].reindex(tickers).fillna(0.0)
            liquidity_cap = available_adv * max_adv_participation / portfolio_value
            delta = delta.clip(lower=-liquidity_cap, upper=liquidity_cap)
            delta = _matched_trade(delta)

            buy = delta.clip(lower=0.0)
            sell = -delta.clip(upper=0.0)
            participation = (delta.abs() * portfolio_value).div(available_adv.replace(0, np.nan)).fillna(0.0)
            impact_rate = impact_coefficient * np.sqrt(participation.clip(lower=0.0))
            base_cost = buy.sum() * buy_cost_bps / 10_000.0 + sell.sum() * sell_cost_bps / 10_000.0
            impact_cost = float((delta.abs() * impact_rate).sum())
            cost = float(base_cost + impact_cost)
            traded_l1 = float(delta.abs().sum())
            one_way_turnover = 0.5 * traded_l1
            new_weights = drifted + delta
            new_weights = new_weights.clip(lower=0.0)
            weights = new_weights / new_weights.sum()

            for ticker, trade_weight in delta[delta.abs() > 1e-12].items():
                trade_rows.append({
                    "Date": date, "Ticker": ticker, "TradeWeight": trade_weight,
                    "PreTradeWeight": drifted[ticker], "PostTradeWeight": weights[ticker],
                    "ADV20": available_adv[ticker], "Participation": participation[ticker],
                })
        else:
            weights = drifted

        net_return = gross_return - cost
        nav = nav * (1.0 + net_return)
        daily_rows.append({
            "Date": date, "GrossReturn": gross_return, "TradingCost": cost,
            "NetReturn": net_return, "NAV": nav, "TurnoverL1": traded_l1,
            "OneWayTurnover": one_way_turnover,
        })
        row = weights.copy()
        row.name = date
        weight_rows.append(row)
    return BacktestResult(
        daily=pd.DataFrame(daily_rows).set_index("Date"),
        weights=pd.DataFrame(weight_rows).fillna(0.0),
        trades=pd.DataFrame(trade_rows),
    )
