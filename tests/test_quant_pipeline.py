import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csi300_enhancement.backtest import _matched_trade, map_decisions_to_execution_dates
from csi300_enhancement.portfolio import construct_active_target


def test_active_target_is_long_only_and_fully_invested():
    tickers = [f"{i:06d}" for i in range(100)]
    pred = pd.Series(np.linspace(-1, 1, 100), index=tickers)
    bench = pd.Series(0.01, index=tickers)
    target, diag = construct_active_target(pred, bench, None)
    assert np.isclose(target.sum(), 1.0)
    assert (target >= 0).all()
    assert np.isclose((target - bench).sum(), 0.0)
    assert diag["ActiveL1"] > 0


def test_matched_trade_has_zero_net_cash_flow():
    delta = pd.Series([0.04, 0.03, -0.02, -0.01])
    out = _matched_trade(delta)
    assert np.isclose(out.sum(), 0.0)
    assert np.isclose(out.clip(lower=0).sum(), -out.clip(upper=0).sum())


def test_decision_is_executed_next_trading_day():
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])
    targets = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex(["2024-01-05"]))
    mapped = map_decisions_to_execution_dates(targets, dates)
    assert mapped.index[0] == pd.Timestamp("2024-01-08")
