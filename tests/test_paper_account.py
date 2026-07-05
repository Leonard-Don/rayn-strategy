from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.paper_account import build_paper_account_snapshot


def test_build_paper_account_snapshot_marks_short_position():
    state = {
        "cash": 99.0,
        "peak": 100.0,
        "halted": False,
        "positions": {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "entry_time": "t0",
                "entry_price": 100.0,
                "notional": 20.0,
                "side": -1,
                "stop_price": 110.0,
                "water": 100.0,
                "bars_held": 2,
            }
        },
    }

    snapshot = build_paper_account_snapshot(state, {"BTCUSDT": 90.0}, timestamp="t1", label="sim")

    assert snapshot["label"] == "sim"
    assert snapshot["cash"] == 99.0
    assert snapshot["unrealized_pnl"] == pytest.approx(2.0)
    assert snapshot["marked_equity"] == pytest.approx(101.0)
    assert snapshot["open_notional"] == 20.0
    assert round(snapshot["open_stop_loss"], 4) == 2.0
    assert snapshot["positions"][0]["symbol"] == "BTCUSDT"
    assert snapshot["positions"][0]["side"] == "SHORT"
