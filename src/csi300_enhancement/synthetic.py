from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_inputs(
    root: Path,
    n_tickers: int = 120,
    start: str = "2017-01-02",
    end: str = "2025-12-31",
    seed: int = 42,
) -> dict[str, Path]:
    """Create deterministic inputs that exercise the full research pipeline.

    The series are deliberately synthetic and must never be interpreted as
    estimates of CSI 300 performance.
    """
    if n_tickers < 60:
        raise ValueError("n_tickers must be at least 60 for the Top-50 baseline")
    dates = pd.bdate_range(start, end)
    tickers = [f"{i:06d}" for i in range(1, n_tickers + 1)]
    rng = np.random.default_rng(seed)

    market = rng.normal(0.00015, 0.009, len(dates))
    style = rng.normal(0.0, 0.006, len(dates))
    betas = rng.uniform(0.7, 1.3, n_tickers)
    style_loadings = rng.normal(0.0, 0.7, n_tickers)
    idio_vol = rng.uniform(0.008, 0.018, n_tickers)
    idiosyncratic = rng.normal(size=(len(dates), n_tickers)) * idio_vol
    daily = market[:, None] * betas + style[:, None] * style_loadings + idiosyncratic
    daily = np.clip(daily, -0.095, 0.095)
    initial = rng.uniform(8.0, 80.0, n_tickers)
    close = initial * np.exp(np.cumsum(np.log1p(daily), axis=0))
    amount = rng.lognormal(mean=19.2, sigma=0.55, size=(len(dates), n_tickers))

    prices = pd.DataFrame(
        {
            "Date": np.repeat(dates.values, n_tickers),
            "Ticker": np.tile(tickers, len(dates)),
            "Close": close.reshape(-1),
            "Amount": amount.reshape(-1),
        }
    )
    raw_dir = Path(root) / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    price_path = raw_dir / "prices.csv.gz"
    prices.to_csv(price_path, index=False, compression="gzip")

    anchor = dates[-1]
    capitalization = close[-1] * rng.lognormal(mean=19.0, sigma=0.8, size=n_tickers)
    weights = capitalization / capitalization.sum()
    weight_path = raw_dir / "csi300_weights.csv"
    pd.DataFrame(
        {"AsOfDate": anchor, "Ticker": tickers, "BenchmarkWeight": weights}
    ).to_csv(weight_path, index=False)

    constituent_path = raw_dir / "csi300_constituents.csv"
    pd.DataFrame({"AsOfDate": anchor, "Ticker": tickers}).to_csv(
        constituent_path, index=False
    )
    return {
        "prices": price_path,
        "weights": weight_path,
        "constituents": constituent_path,
    }
