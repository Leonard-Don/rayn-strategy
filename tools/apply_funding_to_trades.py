#!/usr/bin/env python3
"""Apply USD-M perpetual funding costs to an exported trade CSV."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rayn_strategy.funding import funding_cost_for_trade, load_funding_events


@dataclass
class CsvTrade:
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    notional: float
    pnl: float
    entry_reason: str
    exit_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply funding costs to trade CSV.")
    parser.add_argument("--trades", required=True, help="Trade CSV from run_backtest.py or run_grid_backtest.py.")
    parser.add_argument("--funding", required=True, help="Folder with SYMBOL.csv funding files.")
    parser.add_argument("--out", required=True, help="Adjusted trade CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trades = load_trades(Path(args.trades))
    funding_dir = Path(args.funding)
    events_by_symbol = {
        symbol: load_funding_events(funding_dir / f"{symbol}.csv")
        for symbol in sorted({trade.symbol for trade in trades})
    }

    rows = []
    total_raw = 0.0
    total_funding = 0.0
    total_adjusted = 0.0
    for trade in trades:
        funding_cost = funding_cost_for_trade(trade, events_by_symbol.get(trade.symbol, []))
        adjusted_pnl = trade.pnl - funding_cost
        rows.append((trade, funding_cost, adjusted_pnl))
        total_raw += trade.pnl
        total_funding += funding_cost
        total_adjusted += adjusted_pnl

    write_adjusted(Path(args.out), rows)
    print(f"trades:        {len(trades)}")
    print(f"raw_pnl:       {total_raw:.2f}")
    print(f"funding_cost:  {total_funding:.2f}")
    print(f"adjusted_pnl:  {total_adjusted:.2f}")
    print(f"adjusted_out:  {args.out}")
    print_by_symbol(rows)
    return 0


def load_trades(path: Path) -> list[CsvTrade]:
    trades: list[CsvTrade] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trades.append(
                CsvTrade(
                    symbol=row["symbol"],
                    entry_time=row["entry_time"],
                    exit_time=row["exit_time"],
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    notional=float(row["notional"]),
                    pnl=float(row["pnl"]),
                    entry_reason=row["entry_reason"],
                    exit_reason=row["exit_reason"],
                )
            )
    return trades


def write_adjusted(path: Path, rows: list[tuple[CsvTrade, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "notional",
                "raw_pnl",
                "funding_cost",
                "adjusted_pnl",
                "entry_reason",
                "exit_reason",
            ]
        )
        for trade, funding_cost, adjusted_pnl in rows:
            writer.writerow(
                [
                    trade.symbol,
                    trade.entry_time,
                    trade.exit_time,
                    f"{trade.entry_price:.10g}",
                    f"{trade.exit_price:.10g}",
                    f"{trade.notional:.2f}",
                    f"{trade.pnl:.2f}",
                    f"{funding_cost:.4f}",
                    f"{adjusted_pnl:.4f}",
                    trade.entry_reason,
                    trade.exit_reason,
                ]
            )


def print_by_symbol(rows: list[tuple[CsvTrade, float, float]]) -> None:
    grouped: dict[str, list[tuple[CsvTrade, float, float]]] = {}
    for row in rows:
        grouped.setdefault(row[0].symbol, []).append(row)
    print("by_symbol:")
    for symbol, items in sorted(grouped.items()):
        raw = sum(item[0].pnl for item in items)
        funding = sum(item[1] for item in items)
        adjusted = sum(item[2] for item in items)
        print(f"  {symbol}: raw={raw:.2f} funding={funding:.2f} adjusted={adjusted:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
