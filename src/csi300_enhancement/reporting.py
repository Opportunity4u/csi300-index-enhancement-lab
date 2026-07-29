from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_charts(
    returns: pd.DataFrame,
    metrics: pd.DataFrame,
    factor_ic: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    ax = nav.plot(figsize=(12, 6), linewidth=1.3)
    ax.set_title("Out-of-sample cumulative net value")
    ax.set_ylabel("NAV (start = 1.0)")
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "01_oos_nav.png", dpi=180)
    plt.close(ax.figure)

    drawdown = nav.div(nav.cummax()).sub(1.0)
    ax = drawdown.plot(figsize=(12, 5), linewidth=1.1)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "02_drawdown.png", dpi=180)
    plt.close(ax.figure)

    ordered = factor_ic.sort_values("MeanRankIC")
    colors = ["#f04438" if value < 0 else "#12b76a" for value in ordered["MeanRankIC"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(ordered["Factor"], ordered["MeanRankIC"], color=colors)
    ax.axvline(0.0, color="#344054", linewidth=0.8)
    ax.set_title("Out-of-sample mean Rank IC by factor")
    ax.set_xlabel("Mean daily Spearman IC")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "03_factor_rank_ic.png", dpi=180)
    plt.close(fig)


def write_markdown_report(
    metrics: pd.DataFrame,
    factor_ic: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: dict,
    data_quality: dict,
    cost_sensitivity: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a compact, public-facing report from generated results."""
    implementable = metrics.loc[metrics["Strategy"] == "Cost-aware enhanced"].iloc[0]
    report = f"""# CSI 300 Index-Enhancement Research Report

## Scope

This repository demonstrates a benchmark-aware, walk-forward research and
execution pipeline. It is a research artifact, not investment advice or a live
fund track record.

## Data contract

- Constituents in the run: {data_quality.get("constituents", "N/A")}
- Price period: {data_quality.get("start", "N/A")} to {data_quality.get("end", "N/A")}
- Out-of-sample start: {config["oos_start"]}
- Forecast horizon: {config["target_horizon_days"]} trading days

The illustrative historical result may use a fixed-current constituent universe
and a fixed-share benchmark proxy. That approximation introduces survivorship
and future-universe bias. Replace it with point-in-time membership, adjusted
total-return prices and dated benchmark weights before drawing investment
conclusions.

## Implementable portfolio result

- Annualized return: {implementable["AnnualizedReturn"]:.2%}
- Annualized active return: {implementable["AnnualizedActiveReturn"]:.2%}
- Tracking error: {implementable["TrackingError"]:.2%}
- Information ratio: {implementable["InformationRatio"]:.2f}
- Maximum drawdown: {implementable["MaxDrawdown"]:.2%}
- Annual one-way turnover: {implementable["AnnualOneWayTurnover"]:.2f}x
- Total modeled trading cost: {implementable["TotalTradingCost"]:.2%}

## Audit position

The code separates target weights from drifted and actually executed weights,
delays decisions to the next trading date, caps turnover, applies a no-trade
band and deducts transaction costs. Results should be evaluated out of sample
and after costs.

See `docs/methodology.md`, `docs/data-contract.md` and the generated CSV files
for full definitions.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
