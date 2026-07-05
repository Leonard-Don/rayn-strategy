import csv
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.short_filter_compare import compare_filter_rows, render_filter_report, _parse_periods
from tools.sim_review_report import (
    load_equity_rows,
    load_ledger_rows,
    render_review_report,
    summarize_equity,
    summarize_ledger,
)
from tools.entry_quality_compare import compare_quality_rows, parse_quality_variants, render_quality_report


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_sim_review_report_summarizes_equity_and_ledger(tmp_path):
    equity_path = tmp_path / "equity.csv"
    ledger_path = tmp_path / "ledger.csv"
    _write_csv(
        equity_path,
        [
            "timestamp",
            "label",
            "cash",
            "unrealized_pnl",
            "marked_equity",
            "open_notional",
            "open_stop_loss",
            "position_count",
            "halted",
        ],
        [
            {
                "timestamp": "t1",
                "label": "sim",
                "cash": 75.0,
                "unrealized_pnl": 0.0,
                "marked_equity": 75.0,
                "open_notional": 0.0,
                "open_stop_loss": 0.0,
                "position_count": 0,
                "halted": False,
            },
            {
                "timestamp": "t2",
                "label": "sim",
                "cash": 74.9,
                "unrealized_pnl": 0.6,
                "marked_equity": 75.5,
                "open_notional": 20.0,
                "open_stop_loss": 0.3,
                "position_count": 1,
                "halted": False,
            },
            {
                "timestamp": "t3",
                "label": "sim",
                "cash": 74.5,
                "unrealized_pnl": -0.2,
                "marked_equity": 74.3,
                "open_notional": 12.0,
                "open_stop_loss": 0.4,
                "position_count": 1,
                "halted": False,
            },
        ],
    )
    _write_csv(
        ledger_path,
        [
            "timestamp",
            "action",
            "symbol",
            "side",
            "notional",
            "price",
            "stop",
            "reason",
            "fee_cost",
            "slippage_cost",
            "total_cost",
            "realized_pnl",
            "cash_delta",
            "equity_after",
        ],
        [
            {
                "timestamp": "t2",
                "action": "OPEN",
                "symbol": "ETHUSDT",
                "side": "SELL",
                "notional": 10,
                "price": 100,
                "stop": 105,
                "reason": "momentum_short",
                "fee_cost": 0.01,
                "slippage_cost": 0.02,
                "total_cost": 0.03,
                "realized_pnl": "",
                "cash_delta": -0.03,
                "equity_after": 75.5,
            },
            {
                "timestamp": "t3",
                "action": "CLOSE",
                "symbol": "ETHUSDT",
                "side": "BUY",
                "notional": 10,
                "price": 95,
                "stop": "",
                "reason": "trail_stop",
                "fee_cost": 0.01,
                "slippage_cost": 0.02,
                "total_cost": 0.03,
                "realized_pnl": 0.44,
                "cash_delta": 0.47,
                "equity_after": 74.3,
            },
        ],
    )

    equity = summarize_equity(load_equity_rows(equity_path), initial_equity=75.0)
    ledger = summarize_ledger(load_ledger_rows(ledger_path))
    report = render_review_report(equity, ledger)

    assert equity["latest_equity"] == 74.3
    assert round(equity["roi_pct"], 4) == round((74.3 / 75.0 - 1.0) * 100, 4)
    assert equity["peak_timestamp"] == "t2"
    assert ledger["opens"] == 1
    assert ledger["closes"] == 1
    assert ledger["realized_pnl"] == 0.44
    assert "模拟盘复盘" in report
    assert "ETHUSDT" in report


def test_filter_comparison_reports_before_after_delta():
    rows = compare_filter_rows(
        [
            {
                "period": "sample",
                "variant": "no_rebound_filter",
                "roi": -0.05,
                "max_dd": 0.10,
                "trades": 5,
                "short_trades": 5,
                "win_rate": 0.40,
                "final_equity": 71.25,
            },
            {
                "period": "sample",
                "variant": "rebound_filter",
                "roi": -0.02,
                "max_dd": 0.06,
                "trades": 3,
                "short_trades": 3,
                "win_rate": 0.50,
                "final_equity": 73.50,
            },
        ]
    )
    report = render_filter_report(rows)

    assert rows[0]["roi_delta"] == pytest.approx(0.03)
    assert rows[0]["trade_delta"] == -2
    assert "反弹过滤前后对比" in report
    assert "sample" in report


def test_filter_comparison_parses_iso_period_with_timezone():
    periods = _parse_periods(["current:2026-03-01:2026-06-08T06:00:00+00:00"])

    assert periods == [("current", "2026-03-01", "2026-06-08T06:00:00+00:00")]


def test_entry_quality_compare_parses_variants_and_deltas():
    variants = parse_quality_variants(["balanced:0.35:0.80:0.50:1.15"])
    rows = compare_quality_rows(
        [
            {"period": "sample", "variant": "off", "roi": 0.10, "max_dd": 0.05, "trades": 10},
            {"period": "sample", "variant": "balanced", "roi": 0.14, "max_dd": 0.04, "trades": 8},
        ]
    )
    report = render_quality_report(rows)

    assert variants[0].name == "balanced"
    assert variants[0].min_score == 0.35
    assert rows[0]["roi_delta"] == pytest.approx(0.04)
    assert rows[0]["max_dd_delta"] == pytest.approx(-0.01)
    assert "开仓质量分层对比" in report
    assert "balanced" in report
