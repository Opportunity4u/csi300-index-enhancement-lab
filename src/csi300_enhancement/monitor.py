from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FEATURES, ForwardMonitorConfig
from .data import (
    load_benchmark_snapshot,
    load_constituents,
    load_price_panel,
    make_benchmark_proxy_weights,
)
from .factors import build_factor_panel, rebalance_dates_from_panel
from .market_data import update_price_panel
from .model import fit_model_as_of
from .paper import (
    account_equity,
    execute_pending_target,
    load_paper_state,
    queue_target,
    save_paper_state,
)
from .portfolio import construct_active_target

PUBLIC_COLUMNS = [
    "MarketDate",
    "AsOfDate",
    "EventType",
    "DataCoverage",
    "ModelVersion",
    "ModelTrainEnd",
    "ModelRefit",
    "FormalSignal",
    "MaturedSignalDate",
    "MaturedRankIC",
    "TopDecileReturn",
    "BottomDecileReturn",
    "TopBottomSpread",
    "DirectionHitRate",
    "PortfolioReturn",
    "BenchmarkReturn",
    "ActiveReturn",
    "OneWayTurnover",
    "TradingCostRate",
    "TrackingError20D",
    "InformationRatio20D",
    "ActiveDrawdown",
    "HealthStatus",
    "StructuralReviewEligible",
    "PublicationStatus",
]


def _atomic_csv(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def _append_by_key(frame: pd.DataFrame, path: Path, key: str) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame.copy()
    combined[key] = pd.to_datetime(combined[key]).dt.date.astype(str)
    combined = combined.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    _atomic_csv(combined, path)
    return combined


def _append_prediction_history(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_csv(
            path,
            parse_dates=["DecisionDate", "ModelTrainStart", "ModelTrainEnd"],
            dtype={"Ticker": str},
        )
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame.copy()
    combined["Ticker"] = combined["Ticker"].astype(str).str.zfill(6)
    combined = combined.drop_duplicates(["DecisionDate", "Ticker"], keep="last")
    combined = combined.sort_values(["DecisionDate", "Ticker"]).reset_index(drop=True)
    _atomic_csv(combined, path, compression="gzip")
    return combined


def _rank_ic(predicted: pd.Series, realized: pd.Series) -> float:
    common = predicted.dropna().index.intersection(realized.dropna().index)
    if len(common) < 10:
        return np.nan
    return float(predicted.loc[common].rank().corr(realized.loc[common].rank()))


def evaluate_matured_signal(
    prediction_history: pd.DataFrame,
    close: pd.DataFrame,
    market_date: pd.Timestamp,
    horizon: int,
) -> dict:
    dates = close.index[close.index <= market_date]
    if len(dates) <= horizon:
        return {}
    signal_date = pd.Timestamp(dates[-(horizon + 1)])
    frame = prediction_history[
        pd.to_datetime(prediction_history["DecisionDate"]) == signal_date
    ]
    if frame.empty:
        return {}
    predicted = frame.set_index("Ticker")["PredictedReturn"].astype(float)
    realized = close.loc[market_date].div(close.loc[signal_date]).sub(1.0)
    common = predicted.index.intersection(realized.dropna().index)
    if len(common) < 10:
        return {}
    ranked = predicted.loc[common].rank(pct=True)
    top = realized.loc[common][ranked >= 0.9]
    bottom = realized.loc[common][ranked <= 0.1]
    return {
        "MaturedSignalDate": signal_date.date().isoformat(),
        "MaturedRankIC": _rank_ic(predicted, realized),
        "TopDecileReturn": float(top.mean()),
        "BottomDecileReturn": float(bottom.mean()),
        "TopBottomSpread": float(top.mean() - bottom.mean()),
        "DirectionHitRate": float(
            (np.sign(predicted.loc[common]) == np.sign(realized.loc[common])).mean()
        ),
    }


def classify_health(history: pd.DataFrame) -> tuple[str, dict]:
    recent = history.tail(20).copy()
    rank_ic = pd.to_numeric(recent.get("MaturedRankIC"), errors="coerce").dropna()
    active = pd.to_numeric(recent.get("ActiveReturn"), errors="coerce").fillna(0.0)
    if len(history) < 20 or len(rank_ic) < 10:
        return "GREEN_WARMUP", {
            "RankIC20D": float(rank_ic.mean()) if len(rank_ic) else np.nan,
            "ActiveReturn20D": float((1.0 + active).prod() - 1.0),
        }
    rank_ic_20 = float(rank_ic.mean())
    active_20 = float((1.0 + active).prod() - 1.0)
    active_nav = (1.0 + pd.to_numeric(history["ActiveReturn"], errors="coerce").fillna(0.0)).cumprod()
    active_drawdown = float(active_nav.div(active_nav.cummax()).sub(1.0).iloc[-1])
    last_three_negative = len(rank_ic) >= 3 and bool((rank_ic.tail(3) < 0).all())
    if (rank_ic_20 < -0.02 and active_20 < -0.01) or active_drawdown < -0.02:
        status = "RED"
    elif last_three_negative or rank_ic_20 < 0:
        status = "YELLOW"
    else:
        status = "GREEN"
    return status, {
        "RankIC20D": rank_ic_20,
        "ActiveReturn20D": active_20,
        "ActiveDrawdown": active_drawdown,
    }


def _rolling_active_metrics(history: pd.DataFrame) -> tuple[float, float, float]:
    recent = history.tail(20)
    active = pd.to_numeric(recent["ActiveReturn"], errors="coerce").dropna()
    if len(active) < 2:
        return np.nan, np.nan, 0.0
    tracking_error = float(active.std(ddof=1) * np.sqrt(252.0))
    information_ratio = float(active.mean() * 252.0 / tracking_error) if tracking_error > 0 else np.nan
    all_active = pd.to_numeric(history["ActiveReturn"], errors="coerce").fillna(0.0)
    nav = (1.0 + all_active).cumprod()
    drawdown = float(nav.div(nav.cummax()).sub(1.0).iloc[-1])
    return tracking_error, information_ratio, drawdown


def _model_version(config: ForwardMonitorConfig, coefficients: pd.DataFrame, train_end: pd.Timestamp) -> str:
    payload = {
        "features": FEATURES,
        "ridge_alpha": config.ridge_alpha,
        "training_mode": config.training_mode,
        "train_end": pd.Timestamp(train_end).date().isoformat(),
        "coefficients": coefficients["Coefficient"].round(12).tolist(),
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _write_public_report(history: pd.DataFrame, path: Path) -> None:
    latest = history.iloc[-1]
    metric_rows = [
        ("Market date", latest["MarketDate"]),
        ("Health", latest["HealthStatus"]),
        ("Model version", latest["ModelVersion"]),
        ("Model train end", latest["ModelTrainEnd"]),
        ("Data coverage", f"{float(latest['DataCoverage']):.1%}"),
        ("Matured 5D Rank IC", _format_number(latest["MaturedRankIC"], 4)),
        ("Top-minus-bottom", _format_percent(latest["TopBottomSpread"])),
        ("Paper active return", _format_percent(latest["ActiveReturn"])),
        ("20D tracking error", _format_percent(latest["TrackingError20D"])),
        ("20D information ratio", _format_number(latest["InformationRatio20D"], 2)),
        ("Active drawdown", _format_percent(latest["ActiveDrawdown"])),
    ]
    metrics = "\n".join(f"| {name} | {value} |" for name, value in metric_rows)
    recent = history.tail(20)[
        ["MarketDate", "HealthStatus", "MaturedRankIC", "ActiveReturn", "OneWayTurnover"]
    ]
    recent_rows = []
    for row in recent.itertuples(index=False):
        recent_rows.append(
            f"| {row.MarketDate} | {row.HealthStatus} | "
            f"{_format_number(row.MaturedRankIC, 4)} | {_format_percent(row.ActiveReturn)} | "
            f"{_format_percent(row.OneWayTurnover)} |"
        )
    content = f"""# Forward paper-monitoring record

Last updated: **{latest['AsOfDate']}** (Asia/Shanghai)

This is an unaudited paper-trading research record. It is not a live fund,
broker statement or investment advice. The empirical universe remains subject
to the fixed-current-constituent limitation described in `DATA_NOTICE.md`.

## Latest aggregate diagnostics

| Metric | Value |
|---|---:|
{metrics}

## Recent observations

| Market date | Status | Matured Rank IC | Active return | One-way turnover |
|---|---|---:|---:|---:|
{chr(10).join(recent_rows)}

Ticker-level predictions, holdings and proposed orders are intentionally kept
outside this public repository.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _format_number(value: object, digits: int) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "N/A" if pd.isna(number) else f"{float(number):.{digits}f}"


def _format_percent(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "N/A" if pd.isna(number) else f"{float(number):.2%}"


def _write_forward_chart(history: pd.DataFrame, path: Path) -> None:
    frame = history.copy()
    frame["MarketDate"] = pd.to_datetime(frame["MarketDate"])
    portfolio = (1.0 + pd.to_numeric(frame["PortfolioReturn"], errors="coerce").fillna(0.0)).cumprod()
    benchmark = (1.0 + pd.to_numeric(frame["BenchmarkReturn"], errors="coerce").fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(frame["MarketDate"], portfolio, label="Paper portfolio", linewidth=2)
    ax.plot(frame["MarketDate"], benchmark, label="Benchmark proxy", linewidth=2)
    ax.set_title("Forward paper record (after modeled costs)")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _git_status(public_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={public_root.as_posix()}", "status", "--porcelain"],
        cwd=public_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def publish_public_outputs(public_root: Path, market_date: pd.Timestamp) -> str:
    paths = [
        "docs/forward-monitoring.md",
        "results/forward/daily_summary.csv",
        "results/forward/forward_nav.png",
    ]
    git = ["git", "-c", f"safe.directory={public_root.as_posix()}"]
    subprocess.run([*git, "add", "--", *paths], cwd=public_root, check=True)
    staged = subprocess.run(
        [*git, "diff", "--cached", "--quiet"], cwd=public_root, check=False
    )
    if staged.returncode != 0:
        subprocess.run(
            [*git, "commit", "-m", f"chore: update forward monitor {market_date.date()}"],
            cwd=public_root,
            check=True,
        )
    ahead = subprocess.run(
        [*git, "rev-list", "--count", "origin/main..HEAD"],
        cwd=public_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if ahead.returncode == 0 and ahead.stdout.strip() == "0":
        return "UNCHANGED"
    pushed = subprocess.run(
        [
            *git,
            "push",
            "origin",
            "main",
        ],
        cwd=public_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return "PUSHED" if pushed.returncode == 0 else "COMMITTED_PENDING_PUSH"


def run_forward_monitor(
    private_root: Path,
    public_root: Path,
    *,
    as_of: pd.Timestamp | str | None = None,
    update_data: bool = True,
    publish: bool = False,
    config: ForwardMonitorConfig | None = None,
) -> dict:
    config = config or ForwardMonitorConfig()
    private_root = private_root.resolve()
    public_root = public_root.resolve()
    as_of = pd.Timestamp(as_of or pd.Timestamp.now(tz="Asia/Shanghai").date()).normalize()
    raw = private_root / "data" / "raw"
    private = private_root / "monitoring" / "private"
    state_dir = private_root / "monitoring" / "state"
    private.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    preexisting_dirty = _git_status(public_root) if publish else []
    if preexisting_dirty:
        raise RuntimeError("public repository has pre-existing changes; automatic publish stopped")

    constituents = load_constituents(raw / "csi300_constituents.csv")
    weights_snapshot = load_benchmark_snapshot(raw / "csi300_weights.csv")
    tickers = constituents["Ticker"].drop_duplicates().tolist()
    price_path = raw / "prices.csv.gz"
    if update_data:
        update = update_price_panel(price_path, tickers, as_of)
        prices = update.prices
        update.log.to_csv(private / f"market_data_{as_of.date()}.csv", index=False)
        if not update.index_prices.empty:
            _atomic_csv(update.index_prices, private / "csi300_index.csv.gz", compression="gzip")
        latest_market_date = update.latest_market_date
    else:
        prices = load_price_panel(price_path)
        latest_market_date = pd.Timestamp(prices["Date"].max())
    if latest_market_date is None:
        raise RuntimeError("CSI 300 index cross-check returned no market date")
    latest_market_date = pd.Timestamp(latest_market_date).normalize()
    if latest_market_date > as_of:
        raise RuntimeError("market data is dated after the requested as-of date")

    market_session = latest_market_date == as_of
    if not market_session and as_of.weekday() != config.rebalance_weekday:
        return {
            "status": "NO_MARKET_SESSION",
            "market_date": latest_market_date.date().isoformat(),
            "as_of": as_of.date().isoformat(),
            "publication_status": "UNCHANGED",
        }
    event_type = "MARKET_CLOSE" if market_session else "NO_MARKET_SESSION"
    decision_cutoff = as_of
    latest_rows = prices[prices["Date"] == latest_market_date]
    available = set(latest_rows["Ticker"])
    missing = sorted(set(tickers).difference(available))
    latest_weights = weights_snapshot[
        weights_snapshot["AsOfDate"] == weights_snapshot["AsOfDate"].max()
    ].set_index("Ticker")["BenchmarkWeight"]
    missing_weight = float(latest_weights.reindex(missing).fillna(0.0).sum())
    coverage = 1.0 - len(missing) / len(tickers)
    if coverage < config.min_price_coverage or missing_weight > config.max_missing_benchmark_weight:
        raise RuntimeError(
            f"insufficient current-price coverage: {coverage:.2%}, missing benchmark weight "
            f"{missing_weight:.2%}"
        )
    if missing:
        prior = prices[prices["Ticker"].isin(missing) & (prices["Date"] < latest_market_date)]
        synthetic = prior.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(1).copy()
        synthetic["Date"] = latest_market_date
        prices = pd.concat([prices, synthetic], ignore_index=True)
        prices = prices.drop_duplicates(["Date", "Ticker"], keep="first")

    panel = build_factor_panel(prices, config.target_horizon_days)
    weekly_dates = rebalance_dates_from_panel(
        panel,
        config.rebalance_weekday,
        as_of=decision_cutoff,
        completed_only=True,
    )
    predictions, coefficients, metadata = fit_model_as_of(
        panel,
        FEATURES,
        weekly_dates,
        latest_market_date,
        knowledge_date=as_of,
        horizon=config.target_horizon_days,
        ridge_alpha=config.ridge_alpha,
        training_mode=config.training_mode,
        rolling_years=config.rolling_train_years,
        min_train_rows=config.min_train_rows,
    )
    model_version = _model_version(config, coefficients, metadata["TrainEnd"])
    predictions["ModelVersion"] = model_version
    predictions["SignalType"] = "DAILY_DIAGNOSTIC"
    formal_signal = bool(len(weekly_dates) and weekly_dates.max() == latest_market_date)
    if formal_signal:
        predictions["SignalType"] = "FORMAL_WEEKLY"
        if not market_session:
            event_type = "WEEK_FINALIZATION"

    history_path = private / "prediction_history.csv.gz"
    if not history_path.exists():
        seed_path = private_root / "results" / "predictions_expanding.csv.gz"
        if seed_path.exists():
            seed = pd.read_csv(
                seed_path,
                parse_dates=["DecisionDate", "ModelTrainStart", "ModelTrainEnd"],
                dtype={"Ticker": str},
            )
            seed = seed[seed["DecisionDate"].isin(weekly_dates)].copy()
            seed["ModelVersion"] = "historical-monthly"
            seed["SignalType"] = "FORMAL_WEEKLY"
            _atomic_csv(seed, history_path, compression="gzip")
    prediction_history = _append_prediction_history(predictions, history_path)
    coefficients.to_csv(private / f"coefficients_{latest_market_date.date()}.csv", index=False)

    close = prices.pivot(index="Date", columns="Ticker", values="Close").sort_index().ffill()
    asset_returns = close.pct_change(fill_method=None).fillna(0.0)
    benchmark_weights = make_benchmark_proxy_weights(
        prices,
        latest_weights.reset_index(),
        weights_snapshot["AsOfDate"].max(),
    )
    benchmark_weights = benchmark_weights.reindex(close.index).ffill().fillna(0.0)
    current_prices = close.loc[latest_market_date]

    state_path = state_dir / "paper_account.json"
    paper_state = load_paper_state(state_path, config.paper_capital_cny)
    previous_equity = paper_state.previous_equity
    fills = pd.DataFrame()
    execution = {"TradingCost": 0.0, "OneWayTurnover": 0.0}
    new_market_session = market_session and paper_state.last_market_date != latest_market_date.date().isoformat()
    if new_market_session:
        performance_live_before = paper_state.performance_live
        pre_trade_equity = account_equity(paper_state, current_prices)
        fills, execution = execute_pending_target(
            paper_state,
            current_prices,
            latest_market_date,
            config,
        )
        ending_equity = account_equity(paper_state, current_prices)
        if performance_live_before:
            portfolio_return = (
                ending_equity / previous_equity - 1.0
                if previous_equity not in (None, 0.0)
                else 0.0
            )
            dates = close.index[close.index <= latest_market_date]
            if len(dates) >= 2:
                previous_date = dates[-2]
                benchmark_return = float(
                    (
                        benchmark_weights.loc[previous_date]
                        * asset_returns.loc[latest_market_date]
                    ).sum()
                )
            else:
                benchmark_return = 0.0
        else:
            # Before the first executed allocation there is no investable
            # strategy period to compare with the benchmark. On the first fill
            # date, retain the initial execution cost but start benchmark
            # performance from the close after execution.
            portfolio_return = (
                ending_equity / pre_trade_equity - 1.0 if not fills.empty else 0.0
            )
            benchmark_return = 0.0
            if not fills.empty:
                paper_state.performance_live = True
                paper_state.performance_start_date = latest_market_date.date().isoformat()
        paper_state.previous_equity = ending_equity if paper_state.performance_live else None
        paper_state.benchmark_nav *= 1.0 + benchmark_return
        paper_state.market_sessions += 1
        paper_state.last_market_date = latest_market_date.date().isoformat()
        if not fills.empty:
            fills.to_csv(private / f"fills_{latest_market_date.date()}.csv", index=False)
    else:
        portfolio_return = 0.0
        benchmark_return = 0.0
        pre_trade_equity = account_equity(paper_state, current_prices)
        ending_equity = pre_trade_equity

    matured = evaluate_matured_signal(
        prediction_history,
        close,
        latest_market_date,
        config.target_horizon_days,
    )
    public_path = public_root / "results" / "forward" / "daily_summary.csv"
    existing_public = pd.read_csv(public_path) if public_path.exists() else pd.DataFrame(columns=PUBLIC_COLUMNS)
    previous_model_version = (
        str(existing_public.iloc[-1]["ModelVersion"]) if not existing_public.empty else None
    )
    row = {
        "MarketDate": latest_market_date.date().isoformat(),
        "AsOfDate": as_of.date().isoformat(),
        "EventType": event_type,
        "DataCoverage": coverage,
        "ModelVersion": model_version,
        "ModelTrainEnd": pd.Timestamp(metadata["TrainEnd"]).date().isoformat(),
        "ModelRefit": model_version != previous_model_version,
        "FormalSignal": formal_signal,
        **matured,
        "PortfolioReturn": portfolio_return,
        "BenchmarkReturn": benchmark_return,
        "ActiveReturn": portfolio_return - benchmark_return,
        "OneWayTurnover": execution.get("OneWayTurnover", 0.0),
        "TradingCostRate": execution.get("TradingCost", 0.0) / pre_trade_equity
        if pre_trade_equity > 0
        else 0.0,
        "StructuralReviewEligible": paper_state.market_sessions >= config.structural_review_sessions,
        "PublicationStatus": "AUTOMATED" if publish else "LOCAL_ONLY",
    }
    provisional = pd.concat([existing_public, pd.DataFrame([row])], ignore_index=True)
    provisional["MarketDate"] = pd.to_datetime(provisional["MarketDate"]).dt.date.astype(str)
    provisional = provisional.drop_duplicates("MarketDate", keep="last").sort_values("MarketDate")
    health, _ = classify_health(provisional)
    row["HealthStatus"] = health
    tracking_error, information_ratio, active_drawdown = _rolling_active_metrics(provisional)
    row["TrackingError20D"] = tracking_error
    row["InformationRatio20D"] = information_ratio
    row["ActiveDrawdown"] = active_drawdown

    if formal_signal:
        pred = predictions.set_index("Ticker")["PredictedReturn"]
        bench = benchmark_weights.loc[latest_market_date]
        trailing = asset_returns.loc[asset_returns.index < latest_market_date].tail(
            config.covariance_lookback
        )
        target, diagnostics = construct_active_target(
            pred,
            bench,
            trailing,
            config.active_one_way_budget,
            config.active_weight_cap,
            config.tracking_error_cap,
        )
        if health == "RED":
            target = bench / bench.sum()
        current_pending = paper_state.pending_target or {}
        if current_pending.get("DecisionDate") != latest_market_date.date().isoformat():
            queue_target(
                paper_state,
                target,
                latest_market_date,
                pd.Timestamp(metadata["TrainEnd"]),
                model_version,
            )
        target.rename("TargetWeight").to_csv(
            private / f"target_weights_{latest_market_date.date()}.csv"
        )
        (private / f"target_diagnostics_{latest_market_date.date()}.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    save_paper_state(paper_state, state_path)
    private_report = {
        "config": asdict(config),
        "row": row,
        "model_metadata": {
            key: value.date().isoformat() if isinstance(value, pd.Timestamp) else value
            for key, value in metadata.items()
        },
        "missing_tickers": missing,
        "missing_benchmark_weight": missing_weight,
        "ending_equity": ending_equity,
        "pending_target": paper_state.pending_target,
    }
    (private / f"daily_report_{as_of.date()}.json").write_text(
        json.dumps(private_report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    public_row = pd.DataFrame([{column: row.get(column, np.nan) for column in PUBLIC_COLUMNS}])
    public_history = _append_by_key(public_row, public_path, "MarketDate")
    _write_public_report(public_history, public_root / "docs" / "forward-monitoring.md")
    _write_forward_chart(
        public_history,
        public_root / "results" / "forward" / "forward_nav.png",
    )
    publication_status = "LOCAL_ONLY"
    if publish:
        publication_status = publish_public_outputs(public_root, latest_market_date)
    return {
        "status": health,
        "market_date": latest_market_date.date().isoformat(),
        "model_version": model_version,
        "formal_signal": formal_signal,
        "active_return": row["ActiveReturn"],
        "publication_status": publication_status,
        "private_report": str(private / f"daily_report_{as_of.date()}.json"),
        "public_report": str(public_root / "docs" / "forward-monitoring.md"),
    }
