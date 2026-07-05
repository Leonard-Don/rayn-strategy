#!/usr/bin/env python3
"""Run a bounded Rayn-style grid backtest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.rayn_strategy.exchange_filters import SymbolFilter, symbol_filters_from_exchange_info
from src.rayn_strategy.grid import BoundedGridConfig, BoundedGridEngine
from src.rayn_strategy.config import load_config, load_raw_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded grid backtest.")
    parser.add_argument("--data", required=True, help="Folder with SYMBOL.csv files.")
    parser.add_argument("--config", required=True, help="TOML config path with [grid].")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    parser.add_argument("--entry-start", help="Optional entry start timestamp/date.")
    parser.add_argument("--entry-end", help="Optional exclusive entry end timestamp/date.")
    parser.add_argument("--run-end", help="Optional exclusive run end timestamp/date.")
    parser.add_argument("--exchange-info-json", help="Optional local exchangeInfo JSON file.")
    parser.add_argument(
        "--max-min-order-distortion",
        type=float,
        default=1.5,
        help="Skip orders whose minimum order exceeds target notional by this multiplier.",
    )
    parser.add_argument("--trades-out", help="Optional CSV path for trade details.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    config_path = Path(args.config)
    symbol_filters = load_symbol_filters(Path(args.exchange_info_json), symbols) if args.exchange_info_json else {}
    engine = BoundedGridEngine(
        config=load_config(config_path),
        grid_config=load_grid_config(config_path),
        data_dir=Path(args.data),
        symbols=symbols,
        entry_start=args.entry_start,
        entry_end=args.entry_end,
        run_end=args.run_end,
        symbol_filters=symbol_filters,
        max_min_order_distortion=args.max_min_order_distortion,
    )
    result = engine.run()
    print(result.to_text())
    if args.trades_out:
        write_trades(Path(args.trades_out), result.trades)
        print(f"trades_out:     {args.trades_out}")
    return 0


def load_symbol_filters(path: Path, symbols: list[str]) -> dict[str, SymbolFilter]:
    with path.open() as handle:
        return symbol_filters_from_exchange_info(json.load(handle), symbols)


def load_grid_config(path: Path) -> BoundedGridConfig:
    raw = load_raw_config(path)
    if "grid" not in raw:
        raise ValueError("config must contain a [grid] section")
    return BoundedGridConfig(**raw["grid"])


def write_trades(path: Path, trades: list[object]) -> None:
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
                "pnl",
                "entry_reason",
                "exit_reason",
            ]
        )
        for trade in trades:
            writer.writerow(
                [
                    trade.symbol,
                    trade.entry_time,
                    trade.exit_time,
                    f"{trade.entry_price:.10g}",
                    f"{trade.exit_price:.10g}",
                    f"{trade.notional:.2f}",
                    f"{trade.pnl:.2f}",
                    trade.entry_reason,
                    trade.exit_reason,
                ]
            )


if __name__ == "__main__":
    raise SystemExit(main())
