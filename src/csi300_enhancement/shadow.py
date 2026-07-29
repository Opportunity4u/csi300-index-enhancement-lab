from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def target_weights_to_board_lot_orders(
    target_weights: pd.Series,
    current_shares: pd.Series,
    prices: pd.Series,
    cash: float,
    lot_size: int = 100,
    min_notional: float = 1_000.0,
) -> pd.DataFrame:
    """Translate target weights into executable, board-lot-sized paper orders.

    Orders are rounded toward zero so rounding alone cannot create leverage.
    The function is broker-agnostic and performs no external submission.
    """
    if cash < 0:
        raise ValueError("cash cannot be negative")
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    idx = target_weights.index.union(current_shares.index).intersection(prices.index)
    price = prices.reindex(idx).astype(float)
    if price.isna().any() or (price <= 0).any():
        raise ValueError("all securities require positive reference prices")
    shares = current_shares.reindex(idx).fillna(0.0).astype(float)
    target = target_weights.reindex(idx).fillna(0.0).clip(lower=0.0)
    if target.sum() <= 0:
        raise ValueError("target weights must contain a positive allocation")
    target /= target.sum()

    equity = float(cash + (shares * price).sum())
    raw_target_shares = target * equity / price
    target_shares = np.floor(raw_target_shares / lot_size) * lot_size
    delta = target_shares - shares
    notional = delta.abs() * price
    delta = delta.where(notional >= min_notional, 0.0)

    frame = pd.DataFrame(
        {
            "Ticker": idx,
            "Side": np.where(delta > 0, "BUY", np.where(delta < 0, "SELL", "HOLD")),
            "Shares": delta.abs().astype(int),
            "ReferencePrice": price.values,
            "EstimatedNotional": notional.values,
            "CurrentShares": shares.values.astype(int),
            "TargetShares": target_shares.values.astype(int),
            "TargetWeight": target.values,
        }
    )
    return frame.loc[frame["Shares"] > 0].sort_values(
        ["Side", "EstimatedNotional"], ascending=[True, False]
    ).reset_index(drop=True)


def write_order_packet(
    orders: pd.DataFrame,
    output_path: Path,
    decision_date: pd.Timestamp,
    model_train_end: pd.Timestamp | None = None,
) -> Path:
    """Write a reviewable order packet for manual paper-account entry."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packet = orders.copy()
    packet.insert(0, "DecisionDate", pd.Timestamp(decision_date).date().isoformat())
    packet.insert(
        1,
        "ModelTrainEnd",
        "" if model_train_end is None else pd.Timestamp(model_train_end).date().isoformat(),
    )
    packet["Status"] = "PROPOSED"
    packet.to_csv(output_path, index=False)
    return output_path
