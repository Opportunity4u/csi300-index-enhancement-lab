from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import constrain_rebalance
from .config import ForwardMonitorConfig
from .shadow import target_weights_to_board_lot_orders


@dataclass
class PaperAccountState:
    cash: float
    shares: dict[str, int] = field(default_factory=dict)
    previous_equity: float | None = None
    benchmark_nav: float = 1.0
    cumulative_cost: float = 0.0
    market_sessions: int = 0
    last_market_date: str | None = None
    pending_target: dict | None = None
    performance_live: bool = False
    performance_start_date: str | None = None


def load_paper_state(path: Path, initial_cash: float) -> PaperAccountState:
    if not path.exists():
        return PaperAccountState(cash=float(initial_cash))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["shares"] = {str(key).zfill(6): int(value) for key, value in payload.get("shares", {}).items()}
    state = PaperAccountState(**payload)
    if not state.performance_live:
        # Migrate pre-live states created by earlier monitor versions. Cash
        # waiting for its first executable target has no benchmark history yet.
        state.previous_equity = None
        state.benchmark_nav = 1.0
    return state


def save_paper_state(state: PaperAccountState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def account_equity(state: PaperAccountState, prices: pd.Series) -> float:
    shares = pd.Series(state.shares, dtype=float).reindex(prices.index).fillna(0.0)
    return float(state.cash + (shares * prices).sum())


def account_weights(state: PaperAccountState, prices: pd.Series) -> pd.Series:
    shares = pd.Series(state.shares, dtype=float).reindex(prices.index).fillna(0.0)
    values = shares * prices
    equity = float(state.cash + values.sum())
    if equity <= 0:
        raise ValueError("paper account equity must be positive")
    return values / equity


def queue_target(
    state: PaperAccountState,
    target_weights: pd.Series,
    decision_date: pd.Timestamp,
    model_train_end: pd.Timestamp,
    model_version: str,
) -> None:
    state.pending_target = {
        "DecisionDate": pd.Timestamp(decision_date).date().isoformat(),
        "ModelTrainEnd": pd.Timestamp(model_train_end).date().isoformat(),
        "ModelVersion": model_version,
        "Status": "PENDING",
        "Weights": {str(key).zfill(6): float(value) for key, value in target_weights.items()},
    }


def execute_pending_target(
    state: PaperAccountState,
    prices: pd.Series,
    market_date: pd.Timestamp,
    config: ForwardMonitorConfig,
) -> tuple[pd.DataFrame, dict]:
    pending = state.pending_target
    if not pending or pending.get("Status") != "PENDING":
        return pd.DataFrame(), {"TradingCost": 0.0, "OneWayTurnover": 0.0}
    if pd.Timestamp(pending["DecisionDate"]) >= pd.Timestamp(market_date):
        return pd.DataFrame(), {"TradingCost": 0.0, "OneWayTurnover": 0.0}

    ideal = pd.Series(pending["Weights"], dtype=float).reindex(prices.index).fillna(0.0)
    current = account_weights(state, prices)
    invested = float(current.sum())
    if invested > 0.01:
        normalized_current = current / invested
        executable, _, rebalance_diag = constrain_rebalance(
            normalized_current,
            ideal,
            min_trade_weight=config.min_trade_weight,
            one_way_turnover_cap=config.one_way_turnover_cap,
        )
    else:
        executable = ideal / ideal.sum()
        rebalance_diag = {
            "RawOneWayTurnover": 0.5,
            "ThresholdedOneWayTurnover": 0.5,
            "ExecutedOneWayTurnover": 0.5,
        }

    current_shares = pd.Series(state.shares, dtype=float)
    orders = target_weights_to_board_lot_orders(
        executable,
        current_shares,
        prices,
        state.cash,
        lot_size=config.lot_size,
        min_notional=config.min_trade_notional_cny,
    )
    if orders.empty:
        pending["Status"] = "EXECUTED_NO_TRADE"
        pending["ExecutionDate"] = pd.Timestamp(market_date).date().isoformat()
        return orders, {"TradingCost": 0.0, **rebalance_diag}

    fills: list[dict] = []
    total_cost = 0.0
    traded_notional = 0.0
    ordered = pd.concat([
        orders[orders["Side"] == "SELL"],
        orders[orders["Side"] == "BUY"],
    ])
    for row in ordered.itertuples(index=False):
        ticker = str(row.Ticker).zfill(6)
        price = float(row.ReferencePrice)
        requested = int(row.Shares)
        if row.Side == "SELL":
            executed = min(requested, int(state.shares.get(ticker, 0)))
            notional = executed * price
            cost = notional * config.sell_cost_bps / 10_000.0
            state.cash += notional - cost
            state.shares[ticker] = int(state.shares.get(ticker, 0) - executed)
        else:
            per_lot_cash = config.lot_size * price * (1.0 + config.buy_cost_bps / 10_000.0)
            affordable_lots = int(np.floor(state.cash / per_lot_cash))
            executed = min(requested, affordable_lots * config.lot_size)
            notional = executed * price
            cost = notional * config.buy_cost_bps / 10_000.0
            state.cash -= notional + cost
            state.shares[ticker] = int(state.shares.get(ticker, 0) + executed)
        if executed <= 0:
            continue
        total_cost += cost
        traded_notional += notional
        fills.append({
            "ExecutionDate": pd.Timestamp(market_date),
            "DecisionDate": pd.Timestamp(pending["DecisionDate"]),
            "Ticker": ticker,
            "Side": row.Side,
            "RequestedShares": requested,
            "ExecutedShares": executed,
            "FillPrice": price,
            "Notional": notional,
            "TradingCost": cost,
            "ModelVersion": pending["ModelVersion"],
        })

    state.shares = {ticker: shares for ticker, shares in state.shares.items() if shares > 0}
    state.cumulative_cost += total_cost
    pending["Status"] = "EXECUTED"
    pending["ExecutionDate"] = pd.Timestamp(market_date).date().isoformat()
    equity_before_cost = account_equity(state, prices) + total_cost
    one_way_turnover = 0.5 * traded_notional / equity_before_cost if equity_before_cost > 0 else 0.0
    diagnostics = {
        "TradingCost": total_cost,
        "OneWayTurnover": one_way_turnover,
        **rebalance_diag,
    }
    return pd.DataFrame(fills), diagnostics
