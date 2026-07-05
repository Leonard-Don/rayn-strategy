#!/usr/bin/env python3
"""Multi-cycle out-of-sample validation of the optimized momentum strategy.

The momentum winner was tuned on BTC/ETH/ZEC/ONDO over 2024-2025 (a trending market).
This re-tests it on a DIFFERENT basket (BTC/ETH/SOL/BNB) over 2021-2025, which spans the
2021 top, the brutal 2022 bear (BTC 47k->16k, LUNA/FTX), the 2023 recovery, and the
2024-25 bull + 10-10. Different symbols AND different regimes = a hard honesty test.

Key question: does the regime filter keep it out of the 2022 bear (flat, not bleeding)?
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.config import load_config, load_raw_config
from src.rayn_strategy.backtest import load_candles
from src.rayn_strategy.momentum import MomentumConfig, MomentumEngine

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DATA = {s: load_candles(ROOT / "data_multicycle" / f"{s}.csv") for s in SYMBOLS}
CFG = load_config(ROOT / "config.rayn-momentum.toml")
# Disable the 20% backstop halt so we measure TRUE peak-to-trough drawdown.
CFG = replace(CFG, risk=replace(CFG.risk, max_account_drawdown_pct=0.99))
MOM = MomentumConfig(**load_raw_config(ROOT / "config.rayn-momentum.toml")["momentum"])

CYCLES = [
    ("2021 bull -> top", "2021-01-01", "2022-01-01"),
    ("2022 BEAR (the test)", "2022-01-01", "2023-01-01"),
    ("2023 recovery", "2023-01-01", "2024-01-01"),
    ("2024-25 bull + 10-10", "2024-01-01", "2025-10-13"),
    ("FULL 2021 -> 2025-10", "2021-01-01", "2025-10-13"),
]


def run(es, re):
    return MomentumEngine(config=CFG, mom=MOM, data_dir=ROOT / "data_multicycle",
                          symbols=SYMBOLS, preloaded_data=DATA, entry_start=es, run_end=re).run()


def row(label, r):
    roi = (r.final_equity - r.initial_equity) / r.initial_equity
    w = [t for t in r.trades if t.pnl > 0]
    los = [t for t in r.trades if t.pnl <= 0]
    aw = sum(t.pnl for t in w) / len(w) if w else 0.0
    al = sum(t.pnl for t in los) / len(los) if los else 0.0
    exposure = (sum(1 for p in r.equity_curve if p.open_notional > 0) / len(r.equity_curve)
                if r.equity_curve else 0.0)
    print(f"{label:24} ROI={roi:+8.1%}  maxDD={r.max_drawdown_pct:5.1%}  trades={len(r.trades):4}  "
          f"win%={(len(w)/len(r.trades) if r.trades else 0):4.0%}  W/L={(aw/-al if al else 0):4.1f}  "
          f"in-market={exposure:4.0%}")


def main() -> int:
    print("Optimized momentum on BTC/ETH/SOL/BNB (NEW basket), 2021-2025 (multi-cycle, all OOS)\n")
    print(f"{'period':24} {'ROI':>11} {'maxDD':>9} {'trades':>9} {'win%':>7} {'W/L':>6} {'in-mkt':>8}")
    for label, es, re in CYCLES:
        row(label, run(es, re))
    print("\nin-market = % of bars holding a position (low in a bear = regime filter standing aside)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
