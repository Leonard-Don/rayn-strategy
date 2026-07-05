#!/usr/bin/env python3
"""Monthly forward-window validation for the bounded grid engine."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_grid_backtest import load_grid_config, load_symbol_filters
from src.rayn_strategy.backtest import BacktestResult
from src.rayn_strategy.config import load_config
from src.rayn_strategy.grid import BoundedGridEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded grid monthly validation.")
    parser.add_argument("--data", required=True, help="Folder with SYMBOL.csv files.")
    parser.add_argument("--config", required=True, help="TOML config with [grid].")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    parser.add_argument("--first-test", required=True, help="First test month/date.")
    parser.add_argument("--end", required=True, help="Exclusive end timestamp/date.")
    parser.add_argument("--out", required=True, help="Output CSV report path.")
    parser.add_argument("--exchange-info-json", help="Optional local exchangeInfo JSON file.")
    parser.add_argument(
        "--max-min-order-distortion",
        type=float,
        default=1.5,
        help="Skip orders whose minimum order exceeds target notional by this multiplier.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    rows = run_monthly(
        data_dir=Path(args.data),
        config_path=Path(args.config),
        symbols=symbols,
        first_test=to_utc(args.first_test),
        end=to_utc(args.end),
        exchange_info_json=Path(args.exchange_info_json) if args.exchange_info_json else None,
        max_min_order_distortion=args.max_min_order_distortion,
    )
    write_rows(Path(args.out), rows)
    print_rows(rows)
    return 0


def run_monthly(
    *,
    data_dir: Path,
    config_path: Path,
    symbols: list[str],
    first_test: str,
    end: str,
    exchange_info_json: Path | None = None,
    max_min_order_distortion: float = 1.5,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    symbol_filters = load_symbol_filters(exchange_info_json, symbols) if exchange_info_json else {}
    test_start = first_test
    while test_start < end:
        test_end = min(next_month(test_start), end)
        result = BoundedGridEngine(
            config=load_config(config_path),
            grid_config=load_grid_config(config_path),
            data_dir=data_dir,
            symbols=symbols,
            entry_start=test_start,
            entry_end=test_end,
            run_end=test_end,
            symbol_filters=symbol_filters,
            max_min_order_distortion=max_min_order_distortion,
        ).run()
        rows.append(
            {
                "test_month": test_start[:7],
                "test_start": test_start,
                "test_end": test_end,
                "test_roi": roi(result),
                "test_dd": result.max_drawdown_pct,
                "test_trades": len(result.trades),
                "test_halted": result.halted,
            }
        )
        test_start = test_end
    return rows


def roi(result: BacktestResult) -> float:
    return (result.final_equity - result.initial_equity) / result.initial_equity


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, object]]) -> None:
    print("month,test_roi,test_dd,test_trades,halted")
    total_roi = 0.0
    max_dd = 0.0
    for row in rows:
        total_roi += float(row["test_roi"])
        max_dd = max(max_dd, float(row["test_dd"]))
        print(
            ",".join(
                [
                    str(row["test_month"]),
                    f"{float(row['test_roi']):.2%}",
                    f"{float(row['test_dd']):.2%}",
                    str(row["test_trades"]),
                    str(row["test_halted"]),
                ]
            )
        )
    print(f"sum_test_roi: {total_roi:.2%}")
    print(f"max_test_dd:  {max_dd:.2%}")


def next_month(timestamp: str) -> str:
    parsed = datetime.fromisoformat(timestamp)
    year = parsed.year + (1 if parsed.month == 12 else 0)
    month = 1 if parsed.month == 12 else parsed.month + 1
    return datetime(year, month, 1, tzinfo=timezone.utc).isoformat()


def to_utc(value: str) -> str:
    item = value.strip()
    if len(item) == 10:
        item = item + "T00:00:00+00:00"
    if item.endswith("Z"):
        item = item[:-1] + "+00:00"
    parsed = datetime.fromisoformat(item)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
