from pathlib import Path

import numpy as np
import pandas as pd

from csi300_enhancement.config import FEATURES, ForwardMonitorConfig
from csi300_enhancement.factors import rebalance_dates_from_panel
from csi300_enhancement.model import fit_model_as_of
from csi300_enhancement.monitor import (
    _write_public_report,
    classify_health,
    evaluate_matured_signal,
)
from csi300_enhancement.paper import (
    PaperAccountState,
    execute_pending_target,
    load_paper_state,
    queue_target,
)


def test_partial_week_is_not_a_formal_decision_until_week_end():
    panel = pd.DataFrame(
        {"Date": pd.to_datetime(["2026-07-24", "2026-07-27", "2026-07-28"])}
    )
    on_tuesday = rebalance_dates_from_panel(panel, as_of="2026-07-28")
    on_friday = rebalance_dates_from_panel(panel, as_of="2026-07-31")
    assert on_tuesday.tolist() == [pd.Timestamp("2026-07-24")]
    assert on_friday.tolist() == [pd.Timestamp("2026-07-24"), pd.Timestamp("2026-07-28")]


def test_daily_fit_uses_only_strictly_mature_labels():
    dates = pd.bdate_range("2025-01-01", periods=90)
    tickers = [f"{index:06d}" for index in range(20)]
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            row = {"Date": date, "Ticker": ticker}
            for feature_index, feature in enumerate(FEATURES):
                row[feature] = np.sin((date_index + ticker_index + feature_index) / 7.0)
            row["ForwardReturn5D"] = (ticker_index - 10) / 10_000 + date_index / 1_000_000
            target_index = date_index + 5
            row["TargetDate5D"] = dates[target_index] if target_index < len(dates) else pd.NaT
            rows.append(row)
    panel = pd.DataFrame(rows)
    prediction_date = dates[-1]
    weekly = rebalance_dates_from_panel(panel, as_of=prediction_date)
    predictions, coefficients, metadata = fit_model_as_of(
        panel,
        FEATURES,
        weekly,
        prediction_date,
        min_train_rows=20,
    )
    assert len(predictions) == len(tickers)
    assert len(coefficients) == len(FEATURES)
    assert metadata["TrainEnd"] < prediction_date


def test_matured_signal_evaluation_uses_five_session_realized_return():
    dates = pd.bdate_range("2026-07-20", periods=6)
    tickers = [f"{index:06d}" for index in range(20)]
    predicted = pd.Series(np.arange(20), index=tickers, dtype=float)
    history = pd.DataFrame(
        {
            "DecisionDate": dates[0],
            "Ticker": tickers,
            "PredictedReturn": predicted.values,
        }
    )
    close = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    close.iloc[0] = 100.0
    close.iloc[-1] = 100.0 * (1.0 + np.linspace(-0.02, 0.02, 20))
    close = close.ffill()
    result = evaluate_matured_signal(history, close, dates[-1], horizon=5)
    assert result["MaturedSignalDate"] == dates[0].date().isoformat()
    assert result["MaturedRankIC"] > 0.99
    assert result["TopBottomSpread"] > 0


def test_health_turns_red_only_after_joint_signal_and_active_failure():
    history = pd.DataFrame(
        {
            "MaturedRankIC": [-0.03] * 20,
            "ActiveReturn": [-0.001] * 20,
        }
    )
    status, diagnostics = classify_health(history)
    assert status == "RED"
    assert diagnostics["RankIC20D"] < -0.02


def test_paper_execution_respects_board_lots_and_cash():
    config = ForwardMonitorConfig(paper_capital_cny=100_000.0)
    state = PaperAccountState(cash=100_000.0)
    target = pd.Series({"000001": 0.6, "000002": 0.4})
    queue_target(
        state,
        target,
        pd.Timestamp("2026-07-30"),
        pd.Timestamp("2026-07-24"),
        "model-test",
    )
    fills, diagnostics = execute_pending_target(
        state,
        pd.Series({"000001": 10.0, "000002": 20.0}),
        pd.Timestamp("2026-07-31"),
        config,
    )
    assert not fills.empty
    assert all(shares % 100 == 0 for shares in state.shares.values())
    assert state.cash >= 0
    assert diagnostics["TradingCost"] > 0


def test_pre_live_account_does_not_accumulate_fake_benchmark_return(tmp_path: Path):
    state_path = tmp_path / "paper.json"
    state_path.write_text(
        '{"cash": 10000000, "shares": {}, "previous_equity": 10000000, '
        '"benchmark_nav": 1.02, "market_sessions": 1}',
        encoding="utf-8",
    )
    state = load_paper_state(state_path, 10_000_000.0)
    assert state.previous_equity is None
    assert state.benchmark_nav == 1.0
    assert not state.performance_live


def test_public_report_contains_no_ticker_level_payload(tmp_path: Path):
    history = pd.DataFrame(
        [{
            "MarketDate": "2026-07-31",
            "AsOfDate": "2026-07-31",
            "HealthStatus": "GREEN_WARMUP",
            "ModelVersion": "abcdef123456",
            "ModelTrainEnd": "2026-07-24",
            "DataCoverage": 1.0,
            "MaturedRankIC": 0.01,
            "TopBottomSpread": 0.002,
            "PortfolioReturn": 0.001,
            "BenchmarkReturn": 0.0005,
            "ActiveReturn": 0.0005,
            "OneWayTurnover": 0.0,
            "TrackingError20D": np.nan,
            "InformationRatio20D": np.nan,
            "ActiveDrawdown": 0.0,
        }]
    )
    path = tmp_path / "forward.md"
    _write_public_report(history, path)
    content = path.read_text(encoding="utf-8")
    assert "TargetWeight" not in content
    assert "000001" not in content
