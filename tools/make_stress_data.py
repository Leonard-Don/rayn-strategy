#!/usr/bin/env python3
"""Create a synthetic flash-crash copy of existing CSV market data."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic flash-crash data.")
    parser.add_argument("--data", required=True, help="Input folder with SYMBOL.csv files.")
    parser.add_argument("--out", required=True, help="Output folder for stressed CSV files.")
    parser.add_argument("--time", required=True, help="UTC shock time, e.g. 2026-04-10T00:00:00Z.")
    parser.add_argument("--benchmark-symbol", default="BTCUSDT")
    parser.add_argument("--benchmark-drop-pct", type=float, default=0.12)
    parser.add_argument("--alt-drop-pct", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shock_time = normalize_timestamp(args.time)

    for path in sorted(data_dir.glob("*.csv")):
        symbol = path.stem.upper()
        drop_pct = (
            args.benchmark_drop_pct
            if symbol == args.benchmark_symbol.upper()
            else args.alt_drop_pct
        )
        rows = read_rows(path)
        index = first_index_at_or_after(rows, shock_time)
        if index is None:
            print(f"{symbol}: skipped, no row at or after {shock_time}")
            continue
        apply_flash_crash(rows[index], drop_pct)
        write_rows(out_dir / path.name, rows)
        print(f"{symbol}: shocked {rows[index]['timestamp']} by {drop_pct:.1%}")
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        writer.writerows(rows)


def first_index_at_or_after(rows: list[dict[str, str]], timestamp: str) -> int | None:
    for index, row in enumerate(rows):
        if normalize_timestamp(row["timestamp"]) >= timestamp:
            return index
    return None


def apply_flash_crash(row: dict[str, str], drop_pct: float) -> None:
    open_price = float(row["open"])
    high_price = float(row["high"])
    low_price = float(row["low"])
    close_price = float(row["close"])

    shocked_close = close_price * (1.0 - drop_pct)
    shocked_low = close_price * max(0.01, 1.0 - (drop_pct * 1.25))
    row["high"] = format_price(max(high_price, open_price, shocked_close))
    row["low"] = format_price(min(low_price, shocked_low, shocked_close))
    row["close"] = format_price(shocked_close)


def format_price(value: float) -> str:
    return f"{value:.10g}"


def normalize_timestamp(value: str) -> str:
    item = value.strip()
    if item.endswith("Z"):
        item = item[:-1] + "+00:00"
    parsed = datetime.fromisoformat(item)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
