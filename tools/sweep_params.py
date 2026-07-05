#!/usr/bin/env python3
"""Small parameter sweep for the Rayn-like backtester."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rayn_strategy.backtest import BacktestEngine
from src.rayn_strategy.config import RaynConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep a compact strategy/risk grid.")
    parser.add_argument("--data", required=True, help="Folder with SYMBOL.csv files.")
    parser.add_argument("--config", required=True, help="Base TOML config path.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    parser.add_argument("--top", type=int, default=10, help="Number of rows to print.")
    parser.add_argument("--include-filters", action="store_true", help="Sweep risk filters too.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    base = load_config(Path(args.config))

    rows = []
    take_profit_values = unique_values([base.strategy.take_profit_pct, 0.020, 0.024])
    stop_loss_values = unique_values([base.strategy.stop_loss_pct, 0.016, 0.020])
    trend_ma_values = unique_values([base.strategy.trend_ma_bars, 120, 300])
    buffer_values = unique_values([base.strategy.breakout_buffer_pct, 0.006])
    max_range_values = (
        unique_values([base.strategy.max_entry_range_pct, 0.050, 0.070])
        if args.include_filters
        else [base.strategy.max_entry_range_pct]
    )
    blocked_hour_values = (
        [
            base.strategy.blocked_entry_hours_utc,
            [15],
            [9, 15],
        ]
        if args.include_filters
        else [base.strategy.blocked_entry_hours_utc]
    )
    cooldown_values = (
        [
            (base.risk.max_consecutive_losses, base.risk.loss_cooldown_bars),
            (6, 12),
        ]
        if args.include_filters
        else [(base.risk.max_consecutive_losses, base.risk.loss_cooldown_bars)]
    )

    for take_profit in take_profit_values:
        for stop_loss in stop_loss_values:
            if take_profit <= stop_loss * 0.8:
                continue
            for trend_ma in trend_ma_values:
                for breakout_buffer in buffer_values:
                    for max_range in max_range_values:
                        for blocked_hours in blocked_hour_values:
                            for max_losses, cooldown_bars in cooldown_values:
                                config = with_params(
                                    base,
                                    take_profit_pct=take_profit,
                                    stop_loss_pct=stop_loss,
                                    trend_ma_bars=trend_ma,
                                    breakout_buffer_pct=breakout_buffer,
                                    max_entry_range_pct=max_range,
                                    blocked_entry_hours_utc=blocked_hours,
                                    max_consecutive_losses=max_losses,
                                    loss_cooldown_bars=cooldown_bars,
                                )
                                result = BacktestEngine(
                                    config=config,
                                    data_dir=Path(args.data),
                                    symbols=symbols,
                                ).run()
                                roi = (
                                    result.final_equity - result.initial_equity
                                ) / result.initial_equity
                                score = roi - (2.0 * result.max_drawdown_pct)
                                if result.halted:
                                    score -= 0.05
                                rows.append(
                                    {
                                        "score": score,
                                        "roi": roi,
                                        "dd": result.max_drawdown_pct,
                                        "halted": result.halted,
                                        "trades": len(result.trades),
                                        "tp": take_profit,
                                        "sl": stop_loss,
                                        "ma": trend_ma,
                                        "buf": breakout_buffer,
                                        "range": max_range,
                                        "hours": "-".join(str(hour) for hour in blocked_hours),
                                        "losses": max_losses,
                                        "cooldown": cooldown_bars,
                                    }
                                )

    rows.sort(key=lambda item: item["score"], reverse=True)
    print("score,roi,max_dd,halted,trades,tp,sl,trend_ma,breakout_buffer,max_range,blocked_hours,max_losses,cooldown")
    for row in rows[: args.top]:
        print(
            ",".join(
                [
                    f"{row['score']:.4f}",
                    f"{row['roi']:.2%}",
                    f"{row['dd']:.2%}",
                    str(row["halted"]),
                    str(row["trades"]),
                    f"{row['tp']:.4f}",
                    f"{row['sl']:.4f}",
                    str(row["ma"]),
                    f"{row['buf']:.4f}",
                    f"{row['range']:.4f}",
                    row["hours"],
                    str(row["losses"]),
                    str(row["cooldown"]),
                ]
            )
        )
    return 0


def with_params(
    config: RaynConfig,
    *,
    take_profit_pct: float,
    stop_loss_pct: float,
    trend_ma_bars: int,
    breakout_buffer_pct: float,
    max_entry_range_pct: float,
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
        volatility_lookback_bars=24 if max_entry_range_pct > 0 else 0,
        max_entry_range_pct=max_entry_range_pct,
        blocked_entry_hours_utc=blocked_entry_hours_utc,
    )
    risk = replace(
        config.risk,
        max_consecutive_losses=max_consecutive_losses,
        loss_cooldown_bars=loss_cooldown_bars,
    )
    return replace(config, strategy=strategy, risk=risk)


def unique_values(values: list[float | int]) -> list:
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


if __name__ == "__main__":
    raise SystemExit(main())
