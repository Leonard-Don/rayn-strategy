#!/usr/bin/env python3
"""Search Rayn-style bounded grid variants over 90-day windows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_grid_backtest import load_grid_config
from src.rayn_strategy.backtest import BacktestResult, Candle, load_candles
from src.rayn_strategy.config import RaynConfig, load_config
from src.rayn_strategy.grid import BoundedGridConfig, BoundedGridEngine


DEFAULT_UNIVERSES = [
    "BTCUSDT,ETHUSDT,ZECUSDT",
    "BTCUSDT,ETHUSDT,ZECUSDT,ONDOUSDT",
    "BTCUSDT,ZECUSDT,ONDOUSDT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a Rayn-style grid for 90-day windows.")
    parser.add_argument("--data", required=True, help="Folder with SYMBOL.csv kline files.")
    parser.add_argument("--config", required=True, help="Base observed Rayn TOML config.")
    parser.add_argument("--end", default="2025-09-01", help="Exclusive target end date.")
    parser.add_argument("--days", type=int, default=90, help="Window length in days.")
    parser.add_argument("--universes", help="Semicolon-separated comma symbol lists.")
    parser.add_argument("--out", required=True, help="CSV output path.")
    parser.add_argument("--top", type=int, default=20, help="Number of rows to print.")
    parser.add_argument("--target-roi", type=float, default=0.80, help="Target 90-day ROI.")
    parser.add_argument("--target-dd", type=float, default=0.10, help="Target marked max drawdown.")
    parser.add_argument("--scan-september", action="store_true", help="Run windows ending Sep 1, Sep 15, and Sep 30.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_config = load_config(Path(args.config))
    base_grid = load_grid_config(Path(args.config))
    universes = parse_universes(args.universes)
    end_dates = [to_utc(args.end)]
    if args.scan_september:
        end_dates = [to_utc("2025-09-01"), to_utc("2025-09-15"), to_utc("2025-09-30")]

    rows: list[dict[str, object]] = []
    preloaded_data = load_universe_data(Path(args.data), universes)
    for end in end_dates:
        start = subtract_days(end, args.days)
        rows.extend(
            run_sweep(
                data_dir=Path(args.data),
                preloaded_data=preloaded_data,
                base_config=base_config,
                base_grid=base_grid,
                universes=universes,
                start=start,
                end=end,
                target_roi=args.target_roi,
                target_dd=args.target_dd,
            )
        )

    rows.sort(key=lambda row: (bool(row["meets_target"]), float(row["score"]), float(row["roi"])), reverse=True)
    write_rows(Path(args.out), rows)
    print_rows(rows[: args.top])
    print(f"rows: {len(rows)}")
    print(f"out:  {args.out}")
    return 0


def run_sweep(
    *,
    data_dir: Path,
    preloaded_data: dict[str, list[Candle]],
    base_config: RaynConfig,
    base_grid: BoundedGridConfig,
    universes: list[list[str]],
    start: str,
    end: str,
    target_roi: float,
    target_dd: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    window_data = filter_preloaded_data(preloaded_data, start, end, warmup_hours=500)
    for symbols in universes:
        for exposure_scale in [1.0, 2.0, 4.0, 6.0, 8.0]:
            for take_profit in [0.004, 0.006, 0.008]:
                for layer_step in [0.010, 0.016]:
                    for layer_multiplier in [1.05, 1.18]:
                        for max_rescue_layers in [2, 4, 6]:
                            for hard_stop in [0.12, 0.18]:
                                config, grid = build_variant(
                                    base_config,
                                    base_grid,
                                    exposure_scale=exposure_scale,
                                    take_profit=take_profit,
                                    layer_step=layer_step,
                                    layer_multiplier=layer_multiplier,
                                    max_rescue_layers=max_rescue_layers,
                                    hard_stop=hard_stop,
                                )
                                result = BoundedGridEngine(
                                    config=config,
                                    grid_config=grid,
                                    data_dir=data_dir,
                                    symbols=symbols,
                                    entry_start=start,
                                    entry_end=end,
                                    run_end=end,
                                    preloaded_data=window_data,
                                ).run()
                                rows.append(
                                    summarize_result(
                                        result,
                                        symbols=symbols,
                                        start=start,
                                        end=end,
                                        exposure_scale=exposure_scale,
                                        grid=grid,
                                        target_roi=target_roi,
                                        target_dd=target_dd,
                                    )
                                )
    return rows


def build_variant(
    base_config: RaynConfig,
    base_grid: BoundedGridConfig,
    *,
    exposure_scale: float,
    take_profit: float,
    layer_step: float,
    layer_multiplier: float,
    max_rescue_layers: int,
    hard_stop: float,
) -> tuple[RaynConfig, BoundedGridConfig]:
    risk = replace(
        base_config.risk,
        risk_per_trade_pct=base_config.risk.risk_per_trade_pct * exposure_scale,
        max_position_notional_pct=min(0.90, base_config.risk.max_position_notional_pct * exposure_scale),
        max_used_margin_pct=min(0.95, base_config.risk.max_used_margin_pct * exposure_scale),
        max_account_drawdown_pct=0.95,
    )
    config = replace(base_config, risk=risk)
    grid = replace(
        base_grid,
        max_rescue_layers=max_rescue_layers,
        layer_step_pct=layer_step,
        layer_multiplier=layer_multiplier,
        take_profit_pct=take_profit,
        hard_stop_pct=hard_stop,
        max_symbol_notional_pct=min(8.0, base_grid.max_symbol_notional_pct * exposure_scale),
        max_total_notional_pct=min(12.0, base_grid.max_total_notional_pct * exposure_scale),
    )
    return config, grid


def summarize_result(
    result: BacktestResult,
    *,
    symbols: list[str],
    start: str,
    end: str,
    exposure_scale: float,
    grid: BoundedGridConfig,
    target_roi: float,
    target_dd: float,
) -> dict[str, object]:
    roi = (result.final_equity - result.initial_equity) / result.initial_equity
    realized_dd = max_drawdown([point.cash_equity for point in result.equity_curve])
    marked_dd = result.max_drawdown_pct
    max_notional_pct = max_ratio(
        [(point.open_notional, point.marked_equity) for point in result.equity_curve]
    )
    max_margin_pct = max_ratio(
        [(point.open_margin, point.marked_equity) for point in result.equity_curve]
    )
    wins = [trade for trade in result.trades if trade.pnl > 0]
    win_rate = len(wins) / len(result.trades) if result.trades else 0.0
    deep_pnl = sum(trade.pnl for trade in result.trades if layer_count(trade.entry_reason) >= 5)
    hard_stop_pnl = sum(trade.pnl for trade in result.trades if trade.exit_reason == "grid_hard_stop")
    hard_stop_count = sum(1 for trade in result.trades if trade.exit_reason == "grid_hard_stop")
    score = roi
    score -= max(0.0, target_roi - roi) * 0.20
    score -= max(0.0, marked_dd - target_dd) * 4.0
    score -= max(0.0, max_margin_pct - 0.80) * 0.50
    meets_target = roi >= target_roi and marked_dd <= target_dd

    return {
        "meets_target": meets_target,
        "score": score,
        "start": start[:10],
        "end": end[:10],
        "symbols": ",".join(symbols),
        "roi": roi,
        "marked_dd": marked_dd,
        "realized_dd": realized_dd,
        "max_notional_pct": max_notional_pct,
        "max_margin_pct": max_margin_pct,
        "trades": len(result.trades),
        "win_rate": win_rate,
        "deep_pnl": deep_pnl,
        "hard_stop_pnl": hard_stop_pnl,
        "hard_stop_count": hard_stop_count,
        "halted": result.halted,
        "exposure_scale": exposure_scale,
        "take_profit": grid.take_profit_pct,
        "layer_step": grid.layer_step_pct,
        "layer_multiplier": grid.layer_multiplier,
        "max_rescue_layers": grid.max_rescue_layers,
        "hard_stop": grid.hard_stop_pct,
    }


def max_drawdown(values: list[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - (value / peak))
    return max_dd


def max_ratio(values: list[tuple[float, float]]) -> float:
    ratios = [numerator / denominator for numerator, denominator in values if denominator > 0]
    return max(ratios, default=0.0)


def layer_count(entry_reason: str) -> int:
    marker = "layers="
    if marker not in entry_reason:
        return 1
    return int(entry_reason.split(marker)[-1])


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, object]]) -> None:
    print(
        "rank,meets,score,start,end,symbols,roi,marked_dd,realized_dd,max_notional,max_margin,"
        "trades,win_rate,deep_pnl,hard_stop_pnl,scale,tp,step,mult,layers,hard_stop"
    )
    for index, row in enumerate(rows, start=1):
        print(
            ",".join(
                [
                    str(index),
                    str(row["meets_target"]),
                    f"{float(row['score']):.4f}",
                    str(row["start"]),
                    str(row["end"]),
                    str(row["symbols"]),
                    f"{float(row['roi']):.2%}",
                    f"{float(row['marked_dd']):.2%}",
                    f"{float(row['realized_dd']):.2%}",
                    f"{float(row['max_notional_pct']):.2f}",
                    f"{float(row['max_margin_pct']):.2f}",
                    str(row["trades"]),
                    f"{float(row['win_rate']):.2%}",
                    f"{float(row['deep_pnl']):.2f}",
                    f"{float(row['hard_stop_pnl']):.2f}",
                    f"{float(row['exposure_scale']):.1f}",
                    f"{float(row['take_profit']):.3f}",
                    f"{float(row['layer_step']):.3f}",
                    f"{float(row['layer_multiplier']):.2f}",
                    str(row["max_rescue_layers"]),
                    f"{float(row['hard_stop']):.3f}",
                ]
            )
        )


def parse_universes(raw: str | None) -> list[list[str]]:
    items = raw.split(";") if raw else DEFAULT_UNIVERSES
    return [
        [symbol.strip().upper() for symbol in item.split(",") if symbol.strip()]
        for item in items
        if item.strip()
    ]


def load_universe_data(data_dir: Path, universes: list[list[str]]) -> dict[str, list[Candle]]:
    symbols = sorted({symbol for universe in universes for symbol in universe})
    return {symbol: load_candles(data_dir / f"{symbol}.csv") for symbol in symbols}


def filter_preloaded_data(
    data: dict[str, list[Candle]],
    start: str,
    end: str,
    *,
    warmup_hours: int,
) -> dict[str, list[Candle]]:
    start_dt = datetime.fromisoformat(start) - timedelta(hours=warmup_hours)
    warmup_start = start_dt.astimezone(timezone.utc).isoformat()
    return {
        symbol: [row for row in rows if warmup_start <= row.timestamp < end]
        for symbol, rows in data.items()
    }


def to_utc(value: str) -> str:
    item = value.strip()
    if len(item) == 10:
        item += "T00:00:00+00:00"
    if item.endswith("Z"):
        item = item[:-1] + "+00:00"
    parsed = datetime.fromisoformat(item)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def subtract_days(timestamp: str, days: int) -> str:
    parsed = datetime.fromisoformat(timestamp)
    return (parsed - timedelta(days=days)).astimezone(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
