from pathlib import Path

import pandas as pd
import pytest

from csi300_enhancement.data import load_price_panel


def test_price_loader_normalizes_tickers(tmp_path: Path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Ticker": ["1", "000001"],
            "Close": [10.0, 10.2],
        }
    ).to_csv(path, index=False)
    loaded = load_price_panel(path)
    assert loaded["Ticker"].tolist() == ["000001", "000001"]


def test_price_loader_rejects_duplicate_date_ticker(tmp_path: Path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-02"],
            "Ticker": ["000001", "000001"],
            "Close": [10.0, 10.1],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_price_panel(path)


def test_price_loader_rejects_nonpositive_price(tmp_path: Path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {"Date": ["2024-01-02"], "Ticker": ["000001"], "Close": [0.0]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-positive"):
        load_price_panel(path)
