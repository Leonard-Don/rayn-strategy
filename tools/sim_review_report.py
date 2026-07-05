#!/usr/bin/env python3
"""Review local paper-account equity and order-intent ledger files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.paths import repo_output_path, resolve_report_input_path


def load_equity_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "timestamp": raw["timestamp"],
                    "label": raw.get("label", ""),
                    "cash": _float(raw.get("cash")),
                    "unrealized_pnl": _float(raw.get("unrealized_pnl")),
                    "marked_equity": _float(raw.get("marked_equity")),
                    "open_notional": _float(raw.get("open_notional")),
                    "open_stop_loss": _float(raw.get("open_stop_loss")),
                    "position_count": int(float(raw.get("position_count") or 0)),
                    "halted": str(raw.get("halted", "")).lower() == "true",
                }
            )
    return rows


def load_ledger_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "timestamp": raw["timestamp"],
                    "action": raw.get("action", ""),
                    "symbol": raw.get("symbol", ""),
                    "side": raw.get("side", ""),
                    "notional": _float(raw.get("notional")),
                    "price": _float(raw.get("price")),
                    "stop": _optional_float(raw.get("stop")),
                    "reason": raw.get("reason", ""),
                    "fee_cost": _float(raw.get("fee_cost")),
                    "slippage_cost": _float(raw.get("slippage_cost")),
                    "total_cost": _float(raw.get("total_cost")),
                    "realized_pnl": _optional_float(raw.get("realized_pnl")),
                    "cash_delta": _float(raw.get("cash_delta")),
                    "equity_after": _float(raw.get("equity_after")),
                }
            )
    return rows


def summarize_equity(rows: list[dict[str, Any]], *, initial_equity: float | None = None) -> dict[str, Any]:
    if not rows:
        base = initial_equity or 0.0
        return {
            "points": 0,
            "initial_equity": base,
            "first_timestamp": "",
            "latest_timestamp": "",
            "latest_equity": base,
            "pnl": 0.0,
            "roi_pct": 0.0,
            "peak_equity": base,
            "peak_timestamp": "",
            "trough_equity": base,
            "trough_timestamp": "",
            "max_drawdown_pct": 0.0,
            "current_drawdown_pct": 0.0,
            "cash": base,
            "unrealized_pnl": 0.0,
            "open_notional": 0.0,
            "open_stop_loss": 0.0,
            "position_count": 0,
            "halted": False,
        }

    base = float(initial_equity if initial_equity is not None else rows[0]["marked_equity"])
    latest = rows[-1]
    peak_row = max(rows, key=lambda row: row["marked_equity"])
    trough_row = min(rows, key=lambda row: row["marked_equity"])
    peak = 0.0
    max_drawdown = 0.0
    for row in rows:
        equity = row["marked_equity"]
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    latest_equity = latest["marked_equity"]
    peak_equity = peak_row["marked_equity"]
    return {
        "points": len(rows),
        "initial_equity": base,
        "first_timestamp": rows[0]["timestamp"],
        "latest_timestamp": latest["timestamp"],
        "latest_equity": latest_equity,
        "pnl": latest_equity - base,
        "roi_pct": ((latest_equity / base) - 1.0) * 100 if base else 0.0,
        "peak_equity": peak_equity,
        "peak_timestamp": peak_row["timestamp"],
        "trough_equity": trough_row["marked_equity"],
        "trough_timestamp": trough_row["timestamp"],
        "max_drawdown_pct": max_drawdown * 100,
        "current_drawdown_pct": ((peak_equity - latest_equity) / peak_equity) * 100 if peak_equity else 0.0,
        "cash": latest["cash"],
        "unrealized_pnl": latest["unrealized_pnl"],
        "open_notional": latest["open_notional"],
        "open_stop_loss": latest["open_stop_loss"],
        "position_count": latest["position_count"],
        "halted": latest["halted"],
    }


def summarize_ledger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    opens = [row for row in rows if row["action"] == "OPEN"]
    closes = [row for row in rows if row["action"] == "CLOSE"]
    closed_pnls = [row["realized_pnl"] for row in closes if row["realized_pnl"] is not None]
    by_symbol: dict[str, dict[str, float]] = defaultdict(lambda: {"intents": 0.0, "realized_pnl": 0.0})
    by_reason: dict[str, int] = defaultdict(int)
    for row in rows:
        by_symbol[row["symbol"]]["intents"] += 1
        if row["realized_pnl"] is not None:
            by_symbol[row["symbol"]]["realized_pnl"] += row["realized_pnl"]
        by_reason[row["reason"]] += 1
    return {
        "intents": len(rows),
        "opens": len(opens),
        "closes": len(closes),
        "realized_pnl": sum(closed_pnls),
        "closed_win_rate_pct": (sum(1 for pnl in closed_pnls if pnl > 0) / len(closed_pnls) * 100) if closed_pnls else 0.0,
        "avg_closed_pnl": (sum(closed_pnls) / len(closed_pnls)) if closed_pnls else 0.0,
        "total_cost": sum(row["total_cost"] for row in rows),
        "by_symbol": {key: dict(value) for key, value in sorted(by_symbol.items())},
        "by_reason": dict(sorted(by_reason.items())),
    }


def render_review_report(equity: dict[str, Any], ledger: dict[str, Any]) -> str:
    lines = [
        "# 模拟盘复盘",
        "",
        "## 净值",
        f"- 区间: {equity['first_timestamp']} -> {equity['latest_timestamp']}",
        f"- 初始净值: {equity['initial_equity']:.4f} USDT",
        f"- 当前净值: {equity['latest_equity']:.4f} USDT",
        f"- 当前收益: {equity['pnl']:+.4f} USDT ({equity['roi_pct']:+.2f}%)",
        f"- 峰值净值: {equity['peak_equity']:.4f} USDT @ {equity['peak_timestamp']}",
        f"- 最大回撤: {equity['max_drawdown_pct']:.2f}%",
        f"- 当前持仓数: {equity['position_count']}，未实现盈亏: {equity['unrealized_pnl']:+.4f} USDT",
        f"- 开仓名义本金: {equity['open_notional']:.4f} USDT，止损风险: {equity['open_stop_loss']:.4f} USDT",
        "",
        "## 交易流水",
        f"- 记录条数: {ledger['intents']}，开仓: {ledger['opens']}，平仓: {ledger['closes']}",
        f"- 已实现盈亏: {ledger['realized_pnl']:+.4f} USDT，平仓胜率: {ledger['closed_win_rate_pct']:.1f}%",
        f"- 平均平仓盈亏: {ledger['avg_closed_pnl']:+.4f} USDT，记录成本: {ledger['total_cost']:.4f} USDT",
        "",
        "## 按币种",
    ]
    if ledger["by_symbol"]:
        for symbol, row in ledger["by_symbol"].items():
            lines.append(f"- {symbol}: intents={int(row['intents'])}, realized={row['realized_pnl']:+.4f} USDT")
    else:
        lines.append("- 暂无逐笔流水，从下一次模拟开平仓开始记录。")
    lines.extend(["", "## 按原因"])
    if ledger["by_reason"]:
        for reason, count in ledger["by_reason"].items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- 暂无逐笔原因。")
    return "\n".join(lines) + "\n"


def _float(value: object) -> float:
    return float(value or 0.0)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity", default="reports/runtime/rayn-sim-account-equity.csv")
    parser.add_argument("--ledger", default="reports/runtime/rayn-sim-trade-ledger.csv")
    parser.add_argument("--initial-equity", type=float, default=75.0)
    parser.add_argument("--out", default="reports/runtime/rayn-sim-review-latest.md")
    parser.add_argument("--json-out", default="reports/runtime/rayn-sim-review-latest.json")
    args = parser.parse_args()

    equity_summary = summarize_equity(
        load_equity_rows(resolve_report_input_path(ROOT / args.equity)),
        initial_equity=args.initial_equity,
    )
    ledger_summary = summarize_ledger(load_ledger_rows(resolve_report_input_path(ROOT / args.ledger)))
    report = render_review_report(equity_summary, ledger_summary)
    out = repo_output_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    json_out = repo_output_path(args.json_out)
    json_out.write_text(
        json.dumps({"equity": equity_summary, "ledger": ledger_summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report)
    print(f"# saved -> {out.relative_to(ROOT)}")
    print(f"# saved -> {json_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
