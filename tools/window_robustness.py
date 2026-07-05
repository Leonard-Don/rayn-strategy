#!/usr/bin/env python3
"""Overfitting probe: how much of the edge survives if you do NOT get to pick
the 90-day window?

Slides a 90-day window across the full 2024-2025 sample (step 30 days) and runs
each window through the intrabar-liquidation engine. Reports the ROI
distribution and where the hand-picked target window (2025-06-17) ranks. If the
target window sits in the top tail while the median is poor, the headline
numbers were window selection, not edge.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.config import load_config, load_raw_config
from src.rayn_strategy.grid import BoundedGridConfig, BoundedGridEngine

WINDOW_DAYS = 90
STEP_DAYS = 30
DATA = "data_2024_2025"
SYMBOLS = ["BTCUSDT", "ZECUSDT", "ONDOUSDT"]
SAMPLE_START = datetime(2024, 1, 21, tzinfo=timezone.utc)   # after ONDO listing
SAMPLE_END = datetime(2026, 1, 1, tzinfo=timezone.utc)
TARGET_START = "2025-06-17"


def load_grid_config(path: Path) -> BoundedGridConfig:
    return BoundedGridConfig(**load_raw_config(path)["grid"])


def window_roi(config: str, start: datetime) -> dict:
    end = start + timedelta(days=WINDOW_DAYS)
    cfg_path = ROOT / config
    engine = BoundedGridEngine(
        config=load_config(cfg_path),
        grid_config=load_grid_config(cfg_path),
        data_dir=ROOT / DATA,
        symbols=SYMBOLS,
        entry_start=start.date().isoformat(),
        entry_end=end.date().isoformat(),
        run_end=end.date().isoformat(),
        intrabar_liquidation=True,
    )
    r = engine.run()
    roi = (r.final_equity - r.initial_equity) / r.initial_equity
    liq = sum(1 for t in r.trades if t.exit_reason == "liquidation")
    return {"start": start.date().isoformat(), "roi": roi, "dd": r.max_drawdown_pct,
            "halted": r.halted, "trades": len(r.trades), "liq": liq}


def percentile_rank(values: list[float], x: float) -> float:
    below = sum(1 for v in values if v <= x)
    return below / len(values)


def analyze(config: str) -> None:
    starts = []
    cur = SAMPLE_START
    while cur + timedelta(days=WINDOW_DAYS) <= SAMPLE_END:
        starts.append(cur)
        cur += timedelta(days=STEP_DAYS)
    rows = [window_roi(config, s) for s in starts]
    rois = sorted(r["roi"] for r in rows)
    n = len(rows)
    profitable = sum(1 for r in rows if r["roi"] > 0)
    halted = sum(1 for r in rows if r["halted"])
    median = rois[n // 2]
    mean = sum(rois) / n
    # The exact hand-picked target window, plus a few days either side, to show
    # how knife-edge the start date is.
    target = window_roi(config, datetime.fromisoformat(TARGET_START).replace(tzinfo=timezone.utc))
    rank = percentile_rank([r["roi"] for r in rows] + [target["roi"]], target["roi"])
    neighbors = ["2025-06-10", "2025-06-14", "2025-06-17", "2025-06-20", "2025-06-24", "2025-07-01"]

    print(f"\n### {config}")
    print(f"  {n} rolling 90d windows (step {STEP_DAYS}d), intrabar-liquidation ON")
    print(f"  profitable windows : {profitable}/{n} ({profitable/n:.0%})")
    print(f"  halted windows     : {halted}/{n}")
    print(f"  ROI  min/median/mean/max : {rois[0]:+.2%} / {median:+.2%} / {mean:+.2%} / {rois[-1]:+.2%}")
    print(f"  hand-picked target {TARGET_START}: ROI {target['roi']:+.2%}"
          f"  -> {rank:.0%} percentile (near the top)")
    print(f"  knife-edge on the start date (90d windows):")
    for d in neighbors:
        row = window_roi(config, datetime.fromisoformat(d).replace(tzinfo=timezone.utc))
        print(f"    start {d}: ROI {row['roi']:+7.2%}  DD {row['dd']:6.2%}  "
              f"halted={str(row['halted']):>5s}  trades={row['trades']:4d}  liq={row['liq']}")


def _days(date_str: str) -> int:
    return (datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc) - SAMPLE_START).days


def main() -> int:
    for config in ("config.rayn-dynamic-probation.toml", "config.rayn-overheat-filter.toml"):
        analyze(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
