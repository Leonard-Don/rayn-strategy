#!/usr/bin/env python3
"""Run a Rayn-like strategy backtest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.rayn_strategy.backtest import BacktestEngine
from src.rayn_strategy.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rayn-like backtest.")
    parser.add_argument("--data", required=True, help="Folder with SYMBOL.csv files.")
    parser.add_argument("--config", required=True, help="TOML config path.")
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols, for example BTCUSDT,ETHUSDT,ONDOUSDT.",
    )
    parser.add_argument("--trades-out", help="Optional CSV path for trade details.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    config = load_config(Path(args.config))
    engine = BacktestEngine(config=config, data_dir=Path(args.data), symbols=symbols)
    result = engine.run()
    print(result.to_text())
    if args.trades_out:
        write_trades(Path(args.trades_out), result.trades)
        print(f"trades_out:     {args.trades_out}")
    return 0


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
