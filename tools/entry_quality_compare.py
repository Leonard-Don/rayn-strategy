#!/usr/bin/env python3
"""Compare entry-quality gating and risk-tier variants for momentum profiles.

Research/backtest only. This tool never places orders or touches exchange APIs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.backtest import BacktestResult, load_candles
from src.rayn_strategy.config import load_config, load_raw_config
from src.rayn_strategy.momentum import MomentumConfig, MomentumEngine

DEFAULT_PERIODS = [
    ("2022 bear", "2022-01-01", "2023-01-01"),
    ("2024-25", "2024-01-01", "2025-10-13"),
    ("full", "2021-01-01", "2025-10-13"),
]


@dataclass(frozen=True)
class QualityVariant:
    name: str
    min_score: float
    full_score: float
    min_risk_mult: float
    max_risk_mult: float


DEFAULT_VARIANTS = [
    QualityVariant("off", 0.0, 1.0, 1.0, 1.0),
    QualityVariant("gate_30", 0.30, 0.80, 0.50, 1.00),
    QualityVariant("tier_balanced", 0.35, 0.80, 0.50, 1.15),
    QualityVariant("tier_strict", 0.45, 0.85, 0.40, 1.20),
]


def run_quality_comparison(
    *,
    config_path: Path,
    data_dir: Path,
    symbols: list[str],
    periods: list[tuple[str, str, str]],
    variants: list[QualityVariant],
    initial_equity: float | None = 75.0,
) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    if initial_equity is not None:
        cfg = replace(cfg, portfolio=replace(cfg.portfolio, initial_equity=initial_equity))
    base_mom = MomentumConfig(**load_raw_config(config_path)["momentum"])
    data = {symbol: load_candles(data_dir / f"{symbol}.csv") for symbol in symbols}
    rows: list[dict[str, Any]] = []

    for variant in variants:
        mom = replace(
            base_mom,
            entry_quality_min_score=variant.min_score,
            entry_quality_full_score=variant.full_score,
            entry_quality_min_risk_mult=variant.min_risk_mult,
            entry_quality_max_risk_mult=variant.max_risk_mult,
        )
        for period, start, end in periods:
            result = MomentumEngine(
                config=cfg,
                mom=mom,
                data_dir=data_dir,
                symbols=symbols,
                entry_start=start,
                run_end=end,
                preloaded_data=data,
            ).run()
            rows.append(
                {
                    "period": period,
                    "variant": variant.name,
                    "min_score": variant.min_score,
                    "full_score": variant.full_score,
                    "min_risk_mult": variant.min_risk_mult,
                    "max_risk_mult": variant.max_risk_mult,
                    **_stats(result),
                }
            )
    return rows


def compare_quality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    by_period: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_period.setdefault(row["period"], {})[row["variant"]] = row

    for period, variants in by_period.items():
        baseline = variants.get("off")
        if not baseline:
            continue
        for variant, row in variants.items():
            if variant == "off":
                continue
            paired.append(
                {
                    **row,
                    "roi_delta": row["roi"] - baseline["roi"],
                    "max_dd_delta": row["max_dd"] - baseline["max_dd"],
                    "trade_delta": row["trades"] - baseline["trades"],
                }
            )
    return paired


def render_quality_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# 开仓质量分层对比",
        "",
        "| period | variant | ROI | delta | maxDD | DD delta | trades | trades delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {period} | {variant} | {roi:+.1%} | {roi_delta:+.1%} | "
            "{max_dd:.1%} | {max_dd_delta:+.1%} | {trades:d} | {trade_delta:+d} |".format(**row)
        )
    if rows:
        avg_delta = sum(row["roi_delta"] for row in rows) / len(rows)
        avg_dd_delta = sum(row["max_dd_delta"] for row in rows) / len(rows)
        lines.extend(
            [
                "",
                f"平均 ROI 变化: {avg_delta:+.2%}",
                f"平均 maxDD 变化: {avg_dd_delta:+.2%}",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_quality_variants(raw_variants: list[str] | None) -> list[QualityVariant]:
    if not raw_variants:
        return DEFAULT_VARIANTS
    variants = []
    for raw in raw_variants:
        parts = raw.split(":")
        if len(parts) != 5:
            raise ValueError(f"variant must be name:min_score:full_score:min_mult:max_mult, got {raw!r}")
        variants.append(
            QualityVariant(
                name=parts[0],
                min_score=float(parts[1]),
                full_score=float(parts[2]),
                min_risk_mult=float(parts[3]),
                max_risk_mult=float(parts[4]),
            )
        )
    return variants


def _stats(result: BacktestResult) -> dict[str, Any]:
    wins = [trade for trade in result.trades if trade.pnl > 0]
    return {
        "roi": result.final_equity / result.initial_equity - 1.0,
        "max_dd": result.max_drawdown_pct,
        "trades": len(result.trades),
        "short_trades": sum(1 for trade in result.trades if "short" in trade.entry_reason),
        "win_rate": len(wins) / len(result.trades) if result.trades else 0.0,
        "final_equity": result.final_equity,
    }


def _parse_symbols(raw: str) -> list[str]:
    return [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]


def _parse_periods(raw_periods: list[str] | None) -> list[tuple[str, str, str]]:
    if not raw_periods:
        return DEFAULT_PERIODS
    periods = []
    for raw in raw_periods:
        try:
            name, endpoints = raw.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"period must be name:start:end, got {raw!r}") from exc
        parsed = _split_period_endpoints(endpoints)
        if parsed is None:
            raise ValueError(f"period must be name:start:end, got {raw!r}")
        periods.append((name, parsed[0], parsed[1]))
    return periods


def _split_period_endpoints(raw: str) -> tuple[str, str] | None:
    for idx, char in enumerate(raw):
        if char != ":":
            continue
        start = raw[:idx]
        end = raw[idx + 1 :]
        if _is_iso_like_date(start) and _is_iso_like_date(end):
            return start, end
    return None


def _is_iso_like_date(raw: str) -> bool:
    if not raw:
        return False
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return True


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.rayn-short-bounce-paper.toml")
    parser.add_argument("--data", default="data_multicycle")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    parser.add_argument("--period", action="append", help="name:start:end; can be repeated")
    parser.add_argument("--variant", action="append", help="name:min_score:full_score:min_mult:max_mult")
    parser.add_argument("--initial-equity", type=float, default=75.0)
    parser.add_argument("--csv-out", default="reports/evidence/rayn-entry-quality-comparison.csv")
    parser.add_argument("--out", default="reports/decision-records/rayn-entry-quality-comparison.md")
    args = parser.parse_args()

    raw_rows = run_quality_comparison(
        config_path=ROOT / args.config,
        data_dir=Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data,
        symbols=_parse_symbols(args.symbols),
        periods=_parse_periods(args.period),
        variants=parse_quality_variants(args.variant),
        initial_equity=args.initial_equity,
    )
    rows = compare_quality_rows(raw_rows)
    report = render_quality_report(rows)
    csv_out = ROOT / args.csv_out
    md_out = ROOT / args.out
    write_rows(csv_out, rows)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(report, encoding="utf-8")
    print(report)
    print(f"# saved -> {csv_out.relative_to(ROOT)}")
    print(f"# saved -> {md_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
