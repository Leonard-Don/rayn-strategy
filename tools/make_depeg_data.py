#!/usr/bin/env python3
"""Create a synthetic quote-asset (e.g. USDT) depeg stress copy of CSV data.

A depeg differs from the single-symbol flash crash in three ways, all of which
matter for a long-grid book:

1. Correlation -> 1: every symbol dislocates at once, because the shared quote
   asset is what is stressed (the BTC circuit breaker may not save alts here).
2. Deep intrabar wick: liquidity vanishes, so the bar LOW spikes far below the
   close even if the close partially recovers. This is what the intrabar-low
   liquidation model actually reads.
3. A multi-bar depressed window rather than one instant bar.

Pair this with the engine's --quote-haircut knob (the collateral devaluation
side of a depeg); this tool only produces the price dislocation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic quote-asset depeg data.")
    parser.add_argument("--data", required=True, help="Input folder with SYMBOL.csv files.")
    parser.add_argument("--out", required=True, help="Output folder for stressed CSV files.")
    parser.add_argument("--time", required=True, help="UTC depeg start, e.g. 2025-08-01T00:00:00Z.")
    parser.add_argument("--duration-bars", type=int, default=3, help="Bars the dislocation lasts.")
    parser.add_argument("--alt-wick-pct", type=float, default=0.40, help="Deepest intrabar low drop for alts.")
    parser.add_argument("--alt-close-drop-pct", type=float, default=0.18, help="Close drop for alts at the peak.")
    parser.add_argument("--benchmark-symbol", default="BTCUSDT")
    parser.add_argument(
        "--benchmark-scale",
        type=float,
        default=0.6,
        help="Benchmark severity as a fraction of alt severity (still correlated).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shock_time = normalize_timestamp(args.time)
    duration = max(1, args.duration_bars)

    for path in sorted(data_dir.glob("*.csv")):
        symbol = path.stem.upper()
        scale = args.benchmark_scale if symbol == args.benchmark_symbol.upper() else 1.0
        wick = args.alt_wick_pct * scale
        close_drop = args.alt_close_drop_pct * scale
        rows = read_rows(path)
        index = first_index_at_or_after(rows, shock_time)
        if index is None:
            print(f"{symbol}: skipped, no row at or after {shock_time}")
            continue
        anchor_open = float(rows[index]["open"])
        for k in range(duration):
            if index + k >= len(rows):
                break
            recover = 1.0 - (k / duration)  # 1.0 at the peak bar, easing back
            apply_depeg_bar(rows[index + k], anchor_open, wick * recover, close_drop * recover)
        write_rows(out_dir / path.name, rows)
        print(
            f"{symbol}: depeg from {rows[index]['timestamp']} for {duration} bars"
            f" (wick -{wick:.0%}, close -{close_drop:.0%})"
        )
    return 0


def apply_depeg_bar(row: dict[str, str], anchor_open: float, wick_pct: float, close_drop_pct: float) -> None:
    orig_high = float(row["high"])
    orig_low = float(row["low"])
    target_close = anchor_open * max(0.01, 1.0 - close_drop_pct)
    target_low = anchor_open * max(0.01, 1.0 - wick_pct)
    row["low"] = format_price(min(orig_low, target_low, target_close))
    row["close"] = format_price(target_close)
    row["high"] = format_price(max(orig_high, target_close))


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
