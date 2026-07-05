"""Helpers for tracking local paper-trading account equity."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def build_paper_account_snapshot(
    state: dict[str, Any],
    marks: dict[str, float],
    *,
    timestamp: str,
    label: str,
) -> dict[str, Any]:
    cash = float(state.get("cash", 0.0))
    positions = []
    unrealized = 0.0
    open_notional = 0.0
    open_stop_loss = 0.0

    for symbol, raw in sorted(state.get("positions", {}).items()):
        entry = float(raw["entry_price"])
        mark = float(marks[symbol])
        notional = float(raw["notional"])
        side = int(raw["side"])
        stop = float(raw["stop_price"])
        pnl = side * notional * ((mark / entry) - 1.0)
        stop_pnl = side * notional * ((stop / entry) - 1.0)
        unrealized += pnl
        open_notional += notional
        open_stop_loss += max(0.0, -stop_pnl)
        positions.append(
            {
                "symbol": symbol,
                "side": "LONG" if side > 0 else "SHORT",
                "entry_time": raw.get("entry_time"),
                "entry_price": entry,
                "mark_price": mark,
                "notional": notional,
                "unrealized_pnl": pnl,
                "stop_price": stop,
                "stop_loss": max(0.0, -stop_pnl),
                "bars_held": int(raw.get("bars_held", 0)),
            }
        )

    marked_equity = cash + unrealized
    return {
        "timestamp": timestamp,
        "label": label,
        "cash": cash,
        "unrealized_pnl": unrealized,
        "marked_equity": marked_equity,
        "open_notional": open_notional,
        "open_stop_loss": open_stop_loss,
        "halted": bool(state.get("halted", False)),
        "position_count": len(positions),
        "positions": positions,
    }


def append_snapshot_csv(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "timestamp",
        "label",
        "cash",
        "unrealized_pnl",
        "marked_equity",
        "open_notional",
        "open_stop_loss",
        "position_count",
        "halted",
    ]
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: snapshot[field] for field in fields})
