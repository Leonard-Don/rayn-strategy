#!/usr/bin/env python3
"""Run the optimized trend/momentum strategy (config.rayn-momentum.toml) and report
in-sample / out-of-sample / through-10-10 results. Dumps the full equity curve.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.config import load_config, load_raw_config
from src.rayn_strategy.momentum import MomentumConfig, MomentumEngine

CONFIG = ROOT / "config.rayn-momentum.toml"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "ZECUSDT", "ONDOUSDT"]


def load_mom() -> MomentumConfig:
    return MomentumConfig(**load_raw_config(CONFIG)["momentum"])


def run(entry_start=None, run_end=None):
    return MomentumEngine(config=load_config(CONFIG), mom=load_mom(), data_dir=ROOT / "data_2024_2025",
                          symbols=SYMBOLS, entry_start=entry_start, run_end=run_end).run()


def stats(r, label):
    roi = (r.final_equity - r.initial_equity) / r.initial_equity
    w = [t for t in r.trades if t.pnl > 0]
    al = [t for t in r.trades if t.pnl <= 0]
    aw = sum(t.pnl for t in w) / len(w) if w else 0.0
    avl = sum(t.pnl for t in al) / len(al) if al else 0.0
    print(f"{label:30} ROI={roi:+7.1%}  maxDD={r.max_drawdown_pct:5.1%}  trades={len(r.trades):3}  "
          f"win%={(len(w)/len(r.trades) if r.trades else 0):4.0%}  W/L={(aw/-avl if avl else 0):.1f}")


def main() -> int:
    print("Optimized trend/momentum (mirror of Rayn): cut losses, let winners run, low leverage\n")
    stats(run("2024-02-01", "2025-01-01"), "in-sample 2024")
    stats(run("2025-01-01", "2025-10-12"), "out-of-sample 2025 (incl 10-10)")
    full = run("2024-02-01", "2025-10-12")
    stats(full, "full 2024 -> 2025-10-12")
    out = ROOT / "reports" / "evidence" / "rayn-momentum-equity.csv"
    with out.open("w", newline="") as h:
        wr = csv.writer(h)
        wr.writerow(["timestamp", "marked_equity", "roi_pct"])
        for p in full.equity_curve:
            wr.writerow([p.timestamp, f"{p.marked_equity:.2f}",
                         f"{(p.marked_equity / full.initial_equity - 1) * 100:.2f}"])
    print(f"\nequity curve -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
