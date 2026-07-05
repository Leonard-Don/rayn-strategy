"""Funding-rate helpers for USD-M perpetual futures research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

from .backtest import normalize_timestamp


@dataclass(frozen=True)
class FundingEvent:
    timestamp: str
    funding_rate: float
    mark_price: float


def load_funding_events(path: Path) -> list[FundingEvent]:
    if not path.exists():
        return []
    events: list[FundingEvent] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            events.append(
                FundingEvent(
                    timestamp=normalize_timestamp(row["timestamp"]),
                    funding_rate=float(row["funding_rate"]),
                    mark_price=float(row["mark_price"]),
                )
            )
    events.sort(key=lambda event: event.timestamp)
    return events


def funding_cost_for_trade(trade: object, events: list[FundingEvent]) -> float:
    entry_time = normalize_timestamp(str(getattr(trade, "entry_time")))
    exit_time = normalize_timestamp(str(getattr(trade, "exit_time")))
    notional = float(getattr(trade, "notional"))
    total = 0.0
    for event in events:
        if entry_time < event.timestamp <= exit_time:
            total += notional * event.funding_rate
    return total
