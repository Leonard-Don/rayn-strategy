#!/usr/bin/env python3
"""Walk-forward parameter selection for the Rayn-like strategy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rayn_strategy.backtest import BacktestEngine, BacktestResult, load_candles
from src.rayn_strategy.config import RaynConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monthly walk-forward validation.")
    parser.add_argument("--data", required=True, help="Folder with SYMBOL.csv files.")
    parser.add_argument("--config", required=True, help="Base TOML config path.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    parser.add_argument("--start", required=True, help="First training timestamp/date.")
    parser.add_argument("--first-test", required=True, help="First test month/date.")
    parser.add_argument("--end", required=True, help="Exclusive end timestamp/date.")
    parser.add_argument("--out", required=True, help="Output CSV report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    rows = walk_forward(
        data_dir=Path(args.data),
        base_config=load_config(Path(args.config)),
        symbols=symbols,
        start=to_utc(args.start),
        first_test=to_utc(args.first_test),
        end=to_utc(args.end),
    )
    write_rows(Path(args.out), rows)
    print_rows(rows)
    return 0


def walk_forward(
    *,
    data_dir: Path,
    base_config: RaynConfig,
    symbols: list[str],
    start: str,
    first_test: str,
    end: str,
) -> list[dict[str, object]]:
    validate_data_range(data_dir, symbols, start, end)
    rows: list[dict[str, object]] = []
    test_start = first_test
    while test_start < end:
        test_end = min(next_month(test_start), end)
        best = select_best_config(
            data_dir=data_dir,
            base_config=base_config,
            symbols=symbols,
            train_start=start,
            train_end=test_start,
        )
        test_result = run_window(
            data_dir=data_dir,
            config=best["config"],
            symbols=symbols,
            start=test_start,
            end=test_end,
        )
        train_result = best["result"]
        rows.append(
            {
                "test_month": test_start[:7],
                "train_start": start,
                "train_end": test_start,
                "test_start": test_start,
                "test_end": test_end,
                "train_roi": roi(train_result),
                "train_dd": train_result.max_drawdown_pct,
                "train_trades": len(train_result.trades),
                "test_roi": roi(test_result),
                "test_dd": test_result.max_drawdown_pct,
                "test_trades": len(test_result.trades),
                "test_halted": test_result.halted,
                "tp": best["tp"],
                "sl": best["sl"],
                "trend_ma": best["ma"],
                "breakout_buffer": best["buf"],
                "blocked_hours": "-".join(str(hour) for hour in best["hours"]),
                "max_losses": best["losses"],
                "cooldown": best["cooldown"],
            }
        )
        test_start = test_end
    return rows


def select_best_config(
    *,
    data_dir: Path,
    base_config: RaynConfig,
    symbols: list[str],
    train_start: str,
    train_end: str,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for take_profit in unique_values([base_config.strategy.take_profit_pct, 0.020, 0.024]):
        for stop_loss in unique_values([base_config.strategy.stop_loss_pct, 0.016, 0.020]):
            if take_profit <= stop_loss * 0.8:
                continue
            for trend_ma in unique_values([base_config.strategy.trend_ma_bars, 120, 300]):
                for breakout_buffer in unique_values([base_config.strategy.breakout_buffer_pct, 0.006]):
                    for blocked_hours in [
                        base_config.strategy.blocked_entry_hours_utc,
                        [15],
                        [9, 15],
                    ]:
                        for max_losses, cooldown in [
                            (
                                base_config.risk.max_consecutive_losses,
                                base_config.risk.loss_cooldown_bars,
                            ),
                            (6, 12),
                        ]:
                            config = with_params(
                                base_config,
                                take_profit_pct=take_profit,
                                stop_loss_pct=stop_loss,
                                trend_ma_bars=trend_ma,
                                breakout_buffer_pct=breakout_buffer,
                                blocked_entry_hours_utc=blocked_hours,
                                max_consecutive_losses=max_losses,
                                loss_cooldown_bars=cooldown,
                            )
                            result = run_window(
                                data_dir=data_dir,
                                config=config,
                                symbols=symbols,
                                start=train_start,
                                end=train_end,
                            )
                            score = score_result(result)
                            candidate = {
                                "score": score,
                                "config": config,
                                "result": result,
                                "tp": take_profit,
                                "sl": stop_loss,
                                "ma": trend_ma,
                                "buf": breakout_buffer,
                                "hours": blocked_hours,
                                "losses": max_losses,
                                "cooldown": cooldown,
                            }
                            if best is None or score > float(best["score"]):
                                best = candidate
    if best is None:
        raise RuntimeError("no walk-forward candidate configs were generated")
    return best


def run_window(
    *,
    data_dir: Path,
    config: RaynConfig,
    symbols: list[str],
    start: str,
    end: str,
) -> BacktestResult:
    return BacktestEngine(
        config=config,
        data_dir=data_dir,
        symbols=symbols,
        entry_start=start,
        entry_end=end,
        run_end=end,
    ).run()


def with_params(
    config: RaynConfig,
    *,
    take_profit_pct: float,
    stop_loss_pct: float,
    trend_ma_bars: int,
    breakout_buffer_pct: float,
    blocked_entry_hours_utc: list[int],
    max_consecutive_losses: int,
    loss_cooldown_bars: int,
) -> RaynConfig:
    strategy = replace(
        config.strategy,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        trend_ma_bars=trend_ma_bars,
        breakout_buffer_pct=breakout_buffer_pct,
        blocked_entry_hours_utc=blocked_entry_hours_utc,
    )
    risk = replace(
        config.risk,
        max_consecutive_losses=max_consecutive_losses,
        loss_cooldown_bars=loss_cooldown_bars,
    )
    return replace(config, strategy=strategy, risk=risk)


def validate_data_range(data_dir: Path, symbols: list[str], start: str, end: str) -> None:
    for symbol in symbols:
        candles = load_candles(data_dir / f"{symbol}.csv")
        if not candles:
            raise ValueError(f"{symbol} has no candles")
        if candles[0].timestamp > start:
            raise ValueError(f"{symbol} starts after requested start: {candles[0].timestamp}")
        if candles[-1].timestamp < end:
            raise ValueError(f"{symbol} ends before requested end: {candles[-1].timestamp}")


def score_result(result: BacktestResult) -> float:
    score = roi(result) - (2.0 * result.max_drawdown_pct)
    if result.halted:
        score -= 0.05
    if len(result.trades) < 10:
        score -= 0.02
    return score


def roi(result: BacktestResult) -> float:
    return (result.final_equity - result.initial_equity) / result.initial_equity


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, object]]) -> None:
    print("month,train_roi,train_dd,test_roi,test_dd,test_trades,halted,tp,sl,ma,blocked,cooldown")
    total_roi = 0.0
    max_dd = 0.0
    for row in rows:
        total_roi += float(row["test_roi"])
        max_dd = max(max_dd, float(row["test_dd"]))
        print(
            ",".join(
                [
                    str(row["test_month"]),
                    f"{float(row['train_roi']):.2%}",
                    f"{float(row['train_dd']):.2%}",
                    f"{float(row['test_roi']):.2%}",
                    f"{float(row['test_dd']):.2%}",
                    str(row["test_trades"]),
                    str(row["test_halted"]),
                    f"{float(row['tp']):.4f}",
                    f"{float(row['sl']):.4f}",
                    str(row["trend_ma"]),
                    str(row["blocked_hours"]),
                    str(row["cooldown"]),
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


def unique_values(values: list[float | int]) -> list:
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


if __name__ == "__main__":
    raise SystemExit(main())
