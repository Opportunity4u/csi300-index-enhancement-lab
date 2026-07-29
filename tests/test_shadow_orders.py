from pathlib import Path

import pandas as pd

from csi300_enhancement.shadow import (
    target_weights_to_board_lot_orders,
    write_order_packet,
)


def test_shadow_orders_use_board_lots_and_do_not_create_fractional_shares():
    target = pd.Series({"000001": 0.6, "000002": 0.4})
    current = pd.Series({"000001": 0, "000002": 100})
    prices = pd.Series({"000001": 10.0, "000002": 20.0})
    orders = target_weights_to_board_lot_orders(
        target, current, prices, cash=98_000.0, lot_size=100
    )
    assert (orders["Shares"] % 100 == 0).all()
    assert set(orders["Side"]).issubset({"BUY", "SELL"})


def test_order_packet_is_marked_proposed(tmp_path: Path):
    orders = pd.DataFrame(
        {
            "Ticker": ["000001"],
            "Side": ["BUY"],
            "Shares": [100],
            "ReferencePrice": [10.0],
        }
    )
    path = write_order_packet(
        orders, tmp_path / "orders.csv", pd.Timestamp("2026-07-30")
    )
    saved = pd.read_csv(path)
    assert saved.loc[0, "Status"] == "PROPOSED"
    assert saved.loc[0, "DecisionDate"] == "2026-07-30"
