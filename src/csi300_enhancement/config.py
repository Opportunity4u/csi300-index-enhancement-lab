from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchConfig:
    start_date: str = "2017-01-01"
    end_date: str = "2026-07-28"
    oos_start: str = "2021-01-01"
    target_horizon_days: int = 5
    rebalance_weekday: int = 4  # Friday decision; next trading close is execution.
    retrain_frequency: str = "monthly"
    training_mode: str = "expanding"
    rolling_train_years: int = 3
    ridge_alpha: float = 10.0
    active_one_way_budget: float = 0.10
    active_weight_cap: float = 0.0075
    tracking_error_cap: float = 0.04
    covariance_lookback: int = 252
    min_trade_weight: float = 0.0020
    one_way_turnover_cap: float = 0.10
    portfolio_value_cny: float = 10_000_000.0
    max_adv_participation: float = 1.00
    buy_cost_bps: float = 8.0
    sell_cost_bps: float = 13.0
    impact_coefficient: float = 0.0
    random_state: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class ForwardMonitorConfig:
    """Configuration for the forward-only daily monitoring workflow."""

    target_horizon_days: int = 5
    rebalance_weekday: int = 4
    ridge_alpha: float = 10.0
    training_mode: str = "expanding"
    rolling_train_years: int = 3
    min_train_rows: int = 5_000
    paper_capital_cny: float = 10_000_000.0
    lot_size: int = 100
    min_trade_notional_cny: float = 1_000.0
    min_trade_weight: float = 0.0020
    one_way_turnover_cap: float = 0.10
    buy_cost_bps: float = 8.0
    sell_cost_bps: float = 13.0
    active_one_way_budget: float = 0.10
    active_weight_cap: float = 0.0075
    tracking_error_cap: float = 0.04
    covariance_lookback: int = 252
    min_price_coverage: float = 0.99
    max_missing_benchmark_weight: float = 0.005
    structural_review_sessions: int = 60

    def to_dict(self) -> dict:
        return asdict(self)


FEATURES = [
    "momentum_20_5",
    "momentum_60_5",
    "reversal_5",
    "volatility_20",
    "downside_volatility_20",
    "drawdown_60",
]
