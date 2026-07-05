#!/usr/bin/env python3
"""Compare momentum configs with optional long-side market-breadth filters.

This is a research/backtest tool only. It never places orders.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.backtest import BacktestResult, load_candles
from src.rayn_strategy.config import load_config, load_raw_config
from src.rayn_strategy.momentum import MomentumConfig, MomentumEngine


DEFAULT_PERIODS = [
    ("2026 YTD", "2026-01-01", "2026-06-06"),
    ("2026 MarNow", "2026-03-01", "2026-06-06"),
    ("2026 May14Now", "2026-05-14", "2026-06-06"),
]


def _parse_symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _parse_periods(raw_periods: list[str] | None) -> list[tuple[str, str, str]]:
    if not raw_periods:
        return DEFAULT_PERIODS
    periods = []
    for raw in raw_periods:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"period must be name:start:end, got {raw!r}")
        periods.append((parts[0], parts[1], parts[2]))
    return periods


def _load_momentum_config(path: Path) -> MomentumConfig:
    return MomentumConfig(**load_raw_config(path)["momentum"])


def _stats(result: BacktestResult) -> dict[str, float | int]:
    roi = (result.final_equity / result.initial_equity - 1.0) if result.initial_equity else 0.0
    wins = [t for t in result.trades if t.pnl > 0]
    losses = [t for t in result.trades if t.pnl <= 0]
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
    in_market = (
        sum(1 for p in result.equity_curve if p.open_notional > 0) / len(result.equity_curve)
        if result.equity_curve else 0.0
    )
    return {
        "roi": roi,
        "max_dd": result.max_drawdown_pct,
        "trades": len(result.trades),
        "win_rate": len(wins) / len(result.trades) if result.trades else 0.0,
        "wl": (avg_win / -avg_loss) if avg_loss else 0.0,
        "in_market": in_market,
        "final_equity": result.final_equity,
    }


def run_matrix(
    *,
    config_path: Path,
    data_dir: Path,
    symbols: list[str],
    periods: list[tuple[str, str, str]],
    thresholds: list[float],
    breadth_ma_bars: int,
    disable_account_halt: bool,
) -> list[dict[str, str | float | int]]:
    cfg = load_config(config_path)
    if disable_account_halt:
        cfg = replace(cfg, risk=replace(cfg.risk, max_account_drawdown_pct=0.99))
    base_mom = _load_momentum_config(config_path)
    data = {s: load_candles(data_dir / f"{s}.csv") for s in symbols}
    rows: list[dict[str, str | float | int]] = []

    for threshold in thresholds:
        mom = replace(
            base_mom,
            long_breadth_ma_bars=breadth_ma_bars,
            min_long_breadth_pct=threshold,
        )
        for label, start, end in periods:
            result = MomentumEngine(
                config=cfg,
                mom=mom,
                data_dir=data_dir,
                symbols=symbols,
                entry_start=start,
                run_end=end,
                preloaded_data=data,
            ).run()
            stats = _stats(result)
            rows.append({
                "threshold": threshold,
                "period": label,
                **stats,
            })
    return rows


def _print_rows(rows: list[dict[str, str | float | int]]) -> None:
    print(f"{'breadth':>8} {'period':16} {'ROI':>9} {'maxDD':>8} {'trades':>7} {'win%':>7} {'W/L':>6} {'inMkt':>7}")
    for row in rows:
        print(
            f"{row['threshold']:8.0%} {row['period']:16} "
            f"{row['roi']:>+9.1%} {row['max_dd']:>8.1%} {row['trades']:>7} "
            f"{row['win_rate']:>7.0%} {row['wl']:>6.1f} {row['in_market']:>7.0%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.rayn-momentum-paper.toml")
    parser.add_argument("--data", default="data_multicycle")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    parser.add_argument("--thresholds", default="0,0.25,0.35,0.5,0.65")
    parser.add_argument("--breadth-ma", type=int, default=200)
    parser.add_argument("--period", action="append", help="name:start:end; can be repeated")
    parser.add_argument("--keep-account-halt", action="store_true")
    parser.add_argument("--out", default=None, help="optional CSV output path")
    args = parser.parse_args()

    rows = run_matrix(
        config_path=ROOT / args.config,
        data_dir=Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data,
        symbols=_parse_symbols(args.symbols),
        periods=_parse_periods(args.period),
        thresholds=[float(x) for x in args.thresholds.split(",") if x.strip()],
        breadth_ma_bars=args.breadth_ma,
        disable_account_halt=not args.keep_account_halt,
    )
    _print_rows(rows)

    if args.out:
        out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n# saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
