#!/usr/bin/env python3
"""Backtest the simulated momentum config (config.rayn-momentum-paper.toml, 3x, 500 USDT)
across the 2021-2025 multi-cycle, print a per-cycle table, and save an equity-curve
chart. The 20% account halt is a simulated-monitoring overlay, so it is disabled
here to show true peak-to-trough DD.

If data_recent/ exists, this also reports the recent stretch.
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
CFG = replace(load_config(ROOT / "config.rayn-momentum-paper.toml"),
              risk=replace(load_config(ROOT / "config.rayn-momentum-paper.toml").risk,
                           max_account_drawdown_pct=0.99))
MOM = MomentumConfig(**load_raw_config(ROOT / "config.rayn-momentum-paper.toml")["momentum"])
DATA = {s: load_candles(ROOT / "data_multicycle" / f"{s}.csv") for s in SYMBOLS}

CYCLES = [
    ("2021 bull->top", "2021-01-01", "2022-01-01"),
    ("2022 BEAR", "2022-01-01", "2023-01-01"),
    ("2023 recovery", "2023-01-01", "2024-01-01"),
    ("2024-25 bull+10-10", "2024-01-01", "2025-10-13"),
    ("FULL 2021->2025-10", "2021-01-01", "2025-10-13"),
]


def run(data_dir, data, es, re):
    return MomentumEngine(config=CFG, mom=MOM, data_dir=data_dir, symbols=SYMBOLS,
                          preloaded_data=data, entry_start=es, run_end=re).run()


def stats(r):
    roi = (r.final_equity - r.initial_equity) / r.initial_equity
    w = [t for t in r.trades if t.pnl > 0]
    l = [t for t in r.trades if t.pnl <= 0]
    aw = sum(t.pnl for t in w) / len(w) if w else 0.0
    al = sum(t.pnl for t in l) / len(l) if l else 0.0
    inmkt = (sum(1 for p in r.equity_curve if p.open_notional > 0) / len(r.equity_curve)
             if r.equity_curve else 0.0)
    return roi, r.max_drawdown_pct, len(r.trades), (len(w) / len(r.trades) if r.trades else 0.0), \
        (aw / -al if al else 0.0), inmkt


def chart(full):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available — skipping chart)")
        return
    ec = full.equity_curve
    xs = list(range(len(ec)))
    ys = [(p.marked_equity / full.initial_equity - 1) * 100 for p in ec]
    labels = [p.timestamp[:7] for p in ec]

    def idx_of(pref):
        for i, p in enumerate(ec):
            if p.timestamp[:7] >= pref:
                return i
        return len(ec) - 1
    b0, b1 = idx_of("2022-01"), idx_of("2023-01")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(xs, ys, color="#1a9850", lw=1.0)
    ax.axhline(0, color="#888", ls="--", lw=0.7)
    ax.fill_between(xs, ys, 0, where=[y >= 0 for y in ys], color="#1a9850", alpha=0.12)
    ax.axvspan(b0, b1, color="#d73027", alpha=0.08)
    ax.text((b0 + b1) / 2, max(ys) * 0.15, "2022 bear\n(regime filter sits out)",
            ha="center", color="#b2182b", fontsize=9)
    s = stats(full)
    ax.set_title("Momentum mirror - simulated config (3x, 500 USDT), 2021-2025 multi-cycle\n"
                 "compounds in bulls, preserves capital in the bear", fontsize=12)
    ax.set_ylabel("account ROI %")
    ticks = list(range(0, len(ec), max(1, len(ec) // 12)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[t] for t in ticks], fontsize=8)
    ax.text(0.01, 0.97, f"full: {s[0]:+.0%}  maxDD {s[1]:.0%}  W/L {s[4]:.1f}  win {s[3]:.0%}  "
            f"in-mkt {s[5]:.0%}", transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    plt.tight_layout()
    out = ROOT / "reports" / "evidence" / "momentum-paper-backtest.png"
    plt.savefig(out, dpi=110)
    print(f"\nchart -> {out.relative_to(ROOT)}")


def main() -> int:
    print("=== Momentum mirror (simulated config, 3x, 500 USDT) on BTC/ETH/SOL/BNB ===")
    print(f"{'period':26}{'ROI':>10}{'maxDD':>8}{'trades':>8}{'win%':>7}{'W/L':>6}{'in-mkt':>8}")
    full = None
    for lbl, es, re in CYCLES:
        r = run(ROOT / "data_multicycle", DATA, es, re)
        if lbl.startswith("FULL"):
            full = r
        s = stats(r)
        print(f"{lbl:26}{s[0]:+10.1%}{s[1]:8.1%}{s[2]:8}{s[3]:7.0%}{s[4]:6.1f}{s[5]:8.0%}")
    fresh_dir = ROOT / "data_recent"
    if all((fresh_dir / f"{s}.csv").exists() for s in SYMBOLS):
        fresh = {s: load_candles(fresh_dir / f"{s}.csv") for s in SYMBOLS}
        s = stats(run(fresh_dir, fresh, None, None))
        print(f"{'RECENT (data_recent)':26}{s[0]:+10.1%}{s[1]:8.1%}{s[2]:8}{s[3]:7.0%}{s[4]:6.1f}{s[5]:8.0%}")
    if full:
        chart(full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
