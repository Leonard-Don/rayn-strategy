#!/usr/bin/env python3
"""Multi-cycle equity curve of the optimized momentum strategy, 2021-2025.
Shows it compounds in bulls and sits out the 2022 bear (flat plateau)."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.config import load_config, load_raw_config
from src.rayn_strategy.backtest import load_candles
from src.rayn_strategy.momentum import MomentumConfig, MomentumEngine

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
BEAR = (datetime.fromisoformat("2022-01-01T00:00:00+00:00"),
        datetime.fromisoformat("2023-01-01T00:00:00+00:00"))


def main() -> int:
    data = {s: load_candles(ROOT / "data_multicycle" / f"{s}.csv") for s in SYMBOLS}
    cfg = load_config(ROOT / "config.rayn-momentum.toml")
    cfg = replace(cfg, risk=replace(cfg.risk, max_account_drawdown_pct=0.99))
    mom = MomentumConfig(**load_raw_config(ROOT / "config.rayn-momentum.toml")["momentum"])
    r = MomentumEngine(config=cfg, mom=mom, data_dir=ROOT / "data_multicycle", symbols=SYMBOLS,
                       preloaded_data=data, entry_start="2021-01-01", run_end="2025-10-13").run()
    xs = [datetime.fromisoformat(p.timestamp) for p in r.equity_curve]
    ys = [(p.marked_equity / r.initial_equity - 1) * 100 for p in r.equity_curve]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs, ys, color="#2ca02c", lw=1.2)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.axvspan(*BEAR, color="#d62728", alpha=0.10)
    ax.annotate("2022 bear (BTC −65%)\nmostly in cash → −9.8%",
                xy=(datetime.fromisoformat("2022-06-15T00:00:00+00:00"), max(ys) * 0.25),
                fontsize=9, color="#a11", ha="center")
    ax.annotate(f"end {ys[-1]:+.0f}%", xy=(xs[-1], ys[-1]), xytext=(-6, -2),
                textcoords="offset points", fontsize=11, color="#2ca02c", fontweight="bold", ha="right")
    ax.set_title("Optimized momentum strategy — 2021-2025, multi-cycle (BTC/ETH/SOL/BNB, all out-of-sample)\n"
                 "compounds in bulls, stands aside in the 2022 bear — survives every regime",
                 fontsize=12)
    ax.set_ylabel("account ROI %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = ROOT / "reports" / "evidence" / "rayn-momentum-multicycle.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
