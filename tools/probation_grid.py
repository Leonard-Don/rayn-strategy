#!/usr/bin/env python3
"""Walk-forward bounded grid with symbol probation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_grid_backtest import load_grid_config
from src.rayn_strategy.backtest import BacktestResult
from src.rayn_strategy.config import load_config
from src.rayn_strategy.grid import BoundedGridEngine
from src.rayn_strategy.probation import TradeLike, select_eligible_symbols, symbol_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run symbol-probation grid validation.")
    parser.add_argument("--data", required=True, help="Folder with SYMBOL.csv files.")
    parser.add_argument("--config", required=True, help="TOML config with [grid].")
    parser.add_argument("--symbols", required=True, help="Candidate comma-separated symbols.")
    parser.add_argument("--start", required=True, help="Training start date.")
    parser.add_argument("--first-test", required=True, help="First test month/date.")
    parser.add_argument("--end", required=True, help="Exclusive end timestamp/date.")
    parser.add_argument("--out", required=True, help="Output CSV report path.")
    parser.add_argument("--min-profit-factor", type=float, default=1.1)
    parser.add_argument("--min-pnl", type=float, default=0.0)
    parser.add_argument("--allow-deep-layer-loss", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = run_probation(
        data_dir=Path(args.data),
        config_path=Path(args.config),
        symbols=[item.strip().upper() for item in args.symbols.split(",") if item.strip()],
        start=to_utc(args.start),
        first_test=to_utc(args.first_test),
        end=to_utc(args.end),
        min_profit_factor=args.min_profit_factor,
        min_pnl=args.min_pnl,
        allow_deep_layer_loss=args.allow_deep_layer_loss,
    )
    write_rows(Path(args.out), rows)
    print_rows(rows)
    return 0


def run_probation(
    *,
    data_dir: Path,
    config_path: Path,
    symbols: list[str],
    start: str,
    first_test: str,
    end: str,
    min_profit_factor: float,
    min_pnl: float,
    allow_deep_layer_loss: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    config = load_config(config_path)
    grid_config = load_grid_config(config_path)
    benchmark = config.circuit_breaker.benchmark_symbol
    if benchmark not in symbols:
        symbols = [benchmark] + symbols

    test_start = first_test
    while test_start < end:
        test_end = min(next_month(test_start), end)
        train_result = run_window(
            data_dir=data_dir,
            config_path=config_path,
            symbols=symbols,
            entry_start=start,
            entry_end=test_start,
            run_end=test_start,
        )
        stats = symbol_stats(to_trade_like(train_result.trades))
        eligible = select_eligible_symbols(
            symbols,
            stats,
            benchmark_symbol=benchmark,
            min_profit_factor=min_profit_factor,
            min_pnl=min_pnl,
            allow_deep_layer_loss=allow_deep_layer_loss,
        )
        if eligible == [benchmark] and len(symbols) > 1:
            eligible = symbols[:2] if symbols[0] == benchmark else [benchmark, symbols[0]]

        test_result = BoundedGridEngine(
            config=config,
            grid_config=grid_config,
            data_dir=data_dir,
            symbols=eligible,
            entry_start=test_start,
            entry_end=test_end,
            run_end=test_end,
        ).run()
        rows.append(
            {
                "test_month": test_start[:7],
                "train_start": start,
                "train_end": test_start,
                "test_start": test_start,
                "test_end": test_end,
                "eligible_symbols": ",".join(eligible),
                "train_roi": roi(train_result),
                "train_dd": train_result.max_drawdown_pct,
                "train_trades": len(train_result.trades),
                "test_roi": roi(test_result),
                "test_dd": test_result.max_drawdown_pct,
                "test_trades": len(test_result.trades),
                "test_halted": test_result.halted,
            }
        )
        test_start = test_end
    return rows


def run_window(
    *,
    data_dir: Path,
    config_path: Path,
    symbols: list[str],
    entry_start: str,
    entry_end: str,
    run_end: str,
) -> BacktestResult:
    return BoundedGridEngine(
        config=load_config(config_path),
        grid_config=load_grid_config(config_path),
        data_dir=data_dir,
        symbols=symbols,
        entry_start=entry_start,
        entry_end=entry_end,
        run_end=run_end,
    ).run()


def to_trade_like(trades: list[object]) -> list[TradeLike]:
    return [
        TradeLike(
            symbol=trade.symbol,
            pnl=trade.pnl,
            entry_reason=trade.entry_reason,
            exit_reason=trade.exit_reason,
        )
        for trade in trades
    ]


def roi(result: BacktestResult) -> float:
    return (result.final_equity - result.initial_equity) / result.initial_equity


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, object]]) -> None:
    print("month,eligible,test_roi,test_dd,test_trades,halted")
    total_roi = 0.0
    max_dd = 0.0
    for row in rows:
        total_roi += float(row["test_roi"])
        max_dd = max(max_dd, float(row["test_dd"]))
        print(
            ",".join(
                [
                    str(row["test_month"]),
                    str(row["eligible_symbols"]),
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
