from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .backtest import run_backtest
from .config import FEATURES, ResearchConfig
from .data import (
    load_benchmark_snapshot,
    load_constituents,
    load_price_panel,
    make_benchmark_proxy_weights,
)
from .factor_analysis import factor_decay, factor_ic_summary, quantile_returns
from .factors import build_factor_panel, rebalance_dates_from_panel
from .metrics import performance_metrics
from .model import walk_forward_predictions
from .portfolio import build_target_weight_tables
from .reporting import make_charts, write_markdown_report


def run_pipeline(project_root: Path, config: ResearchConfig | None = None) -> dict:
    config = config or ResearchConfig()
    raw = project_root / "data" / "raw"
    processed = project_root / "data" / "processed"
    results = project_root / "results"
    figures = results / "figures"
    for directory in (processed, results, figures):
        directory.mkdir(parents=True, exist_ok=True)
    config.save(results / "config_used.json")

    prices = load_price_panel(raw / "prices.csv.gz")
    weights = load_benchmark_snapshot(raw / "csi300_weights.csv")
    constituents = load_constituents(raw / "csi300_constituents.csv")
    anchor_date = weights["AsOfDate"].max()

    panel = build_factor_panel(prices, config.target_horizon_days)
    panel.to_csv(processed / "factor_panel.csv.gz", index=False, compression="gzip")
    target_col = f"ForwardReturn{config.target_horizon_days}D"
    ic_summary, daily_ic = factor_ic_summary(panel, FEATURES, target_col, config.oos_start)
    decay = factor_decay(panel, FEATURES)
    qreturns = quantile_returns(panel, FEATURES, target_col)
    ic_summary.to_csv(results / "factor_ic_summary.csv", index=False)
    daily_ic.to_csv(results / "factor_ic_daily.csv", index=False)
    decay.to_csv(results / "factor_decay.csv", index=False)
    qreturns.to_csv(results / "factor_quantile_returns.csv", index=False)

    rebal_dates = rebalance_dates_from_panel(panel, config.rebalance_weekday)
    main_predictions, main_coefs = walk_forward_predictions(
        panel, FEATURES, rebal_dates, config.oos_start, config.target_horizon_days,
        config.ridge_alpha, config.training_mode, config.rolling_train_years,
    )
    rolling_predictions, rolling_coefs = walk_forward_predictions(
        panel, FEATURES, rebal_dates, config.oos_start, config.target_horizon_days,
        config.ridge_alpha, "rolling", config.rolling_train_years,
    )
    main_predictions.to_csv(results / "predictions_expanding.csv.gz", index=False, compression="gzip")
    rolling_predictions.to_csv(results / "predictions_rolling.csv.gz", index=False, compression="gzip")
    pd.concat([main_coefs, rolling_coefs], ignore_index=True).to_csv(results / "model_coefficients.csv", index=False)

    benchmark_weights = make_benchmark_proxy_weights(prices, weights, anchor_date)
    close = prices.pivot(index="Date", columns="Ticker", values="Close").sort_index().ffill()
    asset_returns = close.pct_change(fill_method=None).fillna(0.0)
    # Close-only datasets use a deliberately non-binding amount matrix and fixed
    # all-in bps costs. No historical ADV impact claim is made in that mode.
    if "Amount" in prices.columns:
        amount = prices.pivot(index="Date", columns="Ticker", values="Amount")
        amount = amount.reindex_like(asset_returns).fillna(0.0)
    else:
        amount = pd.DataFrame(1e15, index=asset_returns.index, columns=asset_returns.columns)
    benchmark_weights = benchmark_weights.reindex(asset_returns.index).ffill().fillna(0.0)
    benchmark_returns = (benchmark_weights.shift(1).fillna(benchmark_weights.iloc[0]) * asset_returns).sum(axis=1)

    enhanced_targets, topk_targets, diagnostics = build_target_weight_tables(
        main_predictions, benchmark_weights, asset_returns, config.covariance_lookback,
        config.active_one_way_budget, config.active_weight_cap, config.tracking_error_cap,
    )
    enhanced_targets.to_csv(results / "target_weights_enhanced.csv.gz", compression="gzip")
    diagnostics.to_csv(results / "portfolio_diagnostics.csv", index=False)
    rolling_targets, _, rolling_diagnostics = build_target_weight_tables(
        rolling_predictions, benchmark_weights, asset_returns, config.covariance_lookback,
        config.active_one_way_budget, config.active_weight_cap, config.tracking_error_cap,
    )
    rolling_targets.to_csv(results / "target_weights_rolling.csv.gz", compression="gzip")
    rolling_diagnostics.to_csv(results / "portfolio_diagnostics_rolling.csv", index=False)

    oos_dates = asset_returns.index[asset_returns.index >= pd.Timestamp(config.oos_start)]
    first_oos = oos_dates[0]
    initial = benchmark_weights.loc[first_oos]
    raw_result = run_backtest(
        enhanced_targets, asset_returns.loc[oos_dates], amount.loc[oos_dates], initial,
        min_trade_weight=0.0, one_way_turnover_cap=None, portfolio_value=config.portfolio_value_cny,
        max_adv_participation=1.0, buy_cost_bps=0.0, sell_cost_bps=0.0, impact_coefficient=0.0,
    )
    cost_result = run_backtest(
        enhanced_targets, asset_returns.loc[oos_dates], amount.loc[oos_dates], initial,
        min_trade_weight=config.min_trade_weight,
        one_way_turnover_cap=config.one_way_turnover_cap,
        portfolio_value=config.portfolio_value_cny,
        max_adv_participation=config.max_adv_participation,
        buy_cost_bps=config.buy_cost_bps, sell_cost_bps=config.sell_cost_bps,
        impact_coefficient=config.impact_coefficient,
    )
    top_result = run_backtest(
        topk_targets, asset_returns.loc[oos_dates], amount.loc[oos_dates], initial,
        min_trade_weight=0.0, one_way_turnover_cap=None,
        portfolio_value=config.portfolio_value_cny, max_adv_participation=1.0,
        buy_cost_bps=10.0, sell_cost_bps=10.0, impact_coefficient=0.0,
    )
    rolling_result = run_backtest(
        rolling_targets, asset_returns.loc[oos_dates], amount.loc[oos_dates], initial,
        min_trade_weight=config.min_trade_weight,
        one_way_turnover_cap=config.one_way_turnover_cap,
        portfolio_value=config.portfolio_value_cny,
        max_adv_participation=config.max_adv_participation,
        buy_cost_bps=config.buy_cost_bps, sell_cost_bps=config.sell_cost_bps,
        impact_coefficient=config.impact_coefficient,
    )

    benchmark_daily = pd.DataFrame(index=oos_dates)
    benchmark_daily["GrossReturn"] = benchmark_returns.reindex(oos_dates).fillna(0.0)
    benchmark_daily["TradingCost"] = 0.0
    benchmark_daily["NetReturn"] = benchmark_daily["GrossReturn"]
    benchmark_daily["NAV"] = (1.0 + benchmark_daily["NetReturn"]).cumprod()
    benchmark_daily["TurnoverL1"] = 0.0
    benchmark_daily["OneWayTurnover"] = 0.0

    strategy_map = {
        "Benchmark proxy": benchmark_daily,
        "Top50 equal-weight": top_result.daily,
        "Raw benchmark-aware": raw_result.daily,
        "Cost-aware enhanced": cost_result.daily,
        "Rolling-window robustness": rolling_result.daily,
    }
    metric_rows = [performance_metrics(frame, benchmark_daily["NetReturn"], name) for name, frame in strategy_map.items()]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results / "performance_summary.csv", index=False)
    combined = pd.DataFrame({name: frame["NetReturn"] for name, frame in strategy_map.items()})
    combined.to_csv(results / "daily_strategy_returns.csv")
    cost_result.weights.to_csv(results / "actual_weights_cost_aware.csv.gz", compression="gzip")
    cost_result.trades.to_csv(results / "trades_cost_aware.csv.gz", index=False, compression="gzip")
    raw_result.daily.to_csv(results / "backtest_raw.csv")
    cost_result.daily.to_csv(results / "backtest_cost_aware.csv")
    top_result.daily.to_csv(results / "backtest_top50.csv")
    rolling_result.daily.to_csv(results / "backtest_rolling_cost_aware.csv")

    sensitivity_rows = []
    for cost_bps in (5.0, 10.0, 20.0, 30.0):
        sensitivity_result = run_backtest(
            enhanced_targets, asset_returns.loc[oos_dates], amount.loc[oos_dates], initial,
            min_trade_weight=config.min_trade_weight,
            one_way_turnover_cap=config.one_way_turnover_cap,
            portfolio_value=config.portfolio_value_cny,
            max_adv_participation=config.max_adv_participation,
            buy_cost_bps=cost_bps, sell_cost_bps=cost_bps, impact_coefficient=0.0,
        )
        row = performance_metrics(sensitivity_result.daily, benchmark_daily["NetReturn"], f"{cost_bps:.0f} bps")
        row["CostBps"] = cost_bps
        sensitivity_rows.append(row)
    cost_sensitivity = pd.DataFrame(sensitivity_rows)
    cost_sensitivity.to_csv(results / "cost_sensitivity.csv", index=False)

    make_charts(combined, metrics, ic_summary, figures)
    data_quality = {
        "constituents": int(constituents["Ticker"].nunique()),
        "constituent_date": str(constituents["AsOfDate"].max().date()),
        "weight_date": str(anchor_date.date()),
        "start": str(prices["Date"].min().date()), "end": str(prices["Date"].max().date()),
        "tickers_with_prices": int(prices["Ticker"].nunique()), "price_rows": int(len(prices)),
    }
    (results / "data_quality.json").write_text(json.dumps(data_quality, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(metrics, ic_summary, diagnostics, config.to_dict(), data_quality, cost_sensitivity, project_root / "report" / "quant_research_report.md")
    return {"metrics": metrics, "data_quality": data_quality}
