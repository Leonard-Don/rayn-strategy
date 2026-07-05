#!/usr/bin/env python3
"""Is the momentum mirror's +506% real, or overfit? Three honesty tests.

The winner (L=5, 7d breakout, stop 4xATR, trail 7xATR, risk 0.75%, BTC>200MA) was
tuned on BTC/ETH/ZEC/ONDO over 2024-25. validate_multicycle.py already showed it
holds on a different basket (BTC/ETH/SOL/BNB) 2021-2025. This goes further:

  1. Rolling-window distribution -- if you do NOT get to pick the window, what is
     the median? how many windows lose? how many breach the 20% envelope?
  2. Parameter sensitivity -- perturb each key param around the winner. A real edge
     survives nearby params; an overfit one is a knife-edge.
  3. Leave-one-symbol-out -- does the edge depend on one lucky symbol?

All on the OOS majors basket, true peak-to-trough DD (20% halt disabled), honest costs.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.config import load_config, load_raw_config
from src.rayn_strategy.backtest import load_candles
from src.rayn_strategy.momentum import MomentumConfig, MomentumEngine

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DATA = {s: load_candles(ROOT / "data_multicycle" / f"{s}.csv") for s in SYMBOLS}
CFG = load_config(ROOT / "config.rayn-momentum.toml")
CFG = replace(CFG, risk=replace(CFG.risk, max_account_drawdown_pct=0.99))  # measure true DD
BASE_MOM = MomentumConfig(**load_raw_config(ROOT / "config.rayn-momentum.toml")["momentum"])

SAMPLE_START = datetime(2021, 1, 15, tzinfo=timezone.utc)
SAMPLE_END = datetime(2025, 10, 13, tzinfo=timezone.utc)


def run(mom=BASE_MOM, symbols=SYMBOLS, es=None, re=None):
    data = {s: DATA[s] for s in symbols}
    r = MomentumEngine(config=CFG, mom=mom, data_dir=ROOT / "data_multicycle",
                       symbols=symbols, preloaded_data=data, entry_start=es, run_end=re).run()
    roi = (r.final_equity - r.initial_equity) / r.initial_equity
    w = [t for t in r.trades if t.pnl > 0]
    los = [t for t in r.trades if t.pnl <= 0]
    aw = sum(t.pnl for t in w) / len(w) if w else 0.0
    al = sum(t.pnl for t in los) / len(los) if los else 0.0
    return {"roi": roi, "dd": r.max_drawdown_pct, "n": len(r.trades),
            "win": len(w) / len(r.trades) if r.trades else 0.0, "wl": (aw / -al) if al else 0.0}


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def test_rolling(window_days):
    starts, cur = [], SAMPLE_START
    while cur + timedelta(days=window_days) <= SAMPLE_END:
        starts.append(cur)
        cur += timedelta(days=30)
    rows = [run(es=s.date().isoformat(), re=(s + timedelta(days=window_days)).date().isoformat())
            for s in starts]
    rois = [r["roi"] for r in rows]
    n = len(rows)
    prof = sum(1 for r in rows if r["roi"] > 0)
    dd_breach = sum(1 for r in rows if r["dd"] > 0.20)
    print(f"\n## 1. Rolling {window_days}d windows (step 30d), {n} windows, OOS majors")
    print(f"   profitable     : {prof}/{n} ({prof/n:.0%})")
    print(f"   ROI min/median/max : {min(rois):+.1%} / {median(rois):+.1%} / {max(rois):+.1%}")
    print(f"   maxDD median/worst : {median([r['dd'] for r in rows]):.1%} / {max(r['dd'] for r in rows):.1%}")
    print(f"   windows breaching 20% DD : {dd_breach}/{n} ({dd_breach/n:.0%})")


def test_params():
    print("\n## 2. Parameter sensitivity (full 2021-2025, one param off the winner at a time)")
    print(f"   {'variant':28} {'ROI':>9} {'maxDD':>7} {'trades':>7} {'win%':>6} {'W/L':>5}")
    grid = {
        "stop_atr_mult": [3.0, 4.0, 5.0, 6.0],
        "trail_atr_mult": [5.0, 7.0, 9.0, 11.0],
        "breakout_bars": [120, 168, 240, 336],
        "risk_per_trade_pct": [0.005, 0.0075, 0.01, 0.015],
    }
    for field, vals in grid.items():
        for v in vals:
            mom = replace(BASE_MOM, **{field: v})
            r = run(mom=mom, es=SAMPLE_START.date().isoformat(), re=SAMPLE_END.date().isoformat())
            star = "  <- winner" if getattr(BASE_MOM, field) == v else ""
            print(f"   {field}={v:<14} {r['roi']:+9.1%} {r['dd']:7.1%} {r['n']:7} "
                  f"{r['win']:6.0%} {r['wl']:5.1f}{star}")


def test_leave_one_out():
    print("\n## 3. Leave-one-symbol-out (full 2021-2025) -- is it one lucky symbol?")
    print(f"   {'basket':28} {'ROI':>9} {'maxDD':>7} {'trades':>7} {'W/L':>5}")
    full = run(es=SAMPLE_START.date().isoformat(), re=SAMPLE_END.date().isoformat())
    print(f"   {'ALL 4':28} {full['roi']:+9.1%} {full['dd']:7.1%} {full['n']:7} {full['wl']:5.1f}")
    for drop in SYMBOLS:
        if drop == "BTCUSDT":
            continue  # BTC is the regime benchmark; keep it
        syms = [s for s in SYMBOLS if s != drop]
        r = run(symbols=syms, es=SAMPLE_START.date().isoformat(), re=SAMPLE_END.date().isoformat())
        print(f"   {'minus ' + drop:28} {r['roi']:+9.1%} {r['dd']:7.1%} {r['n']:7} {r['wl']:5.1f}")


def main() -> int:
    print("Momentum mirror robustness -- OOS majors BTC/ETH/SOL/BNB, true DD, costs on")
    full = run(es=SAMPLE_START.date().isoformat(), re=SAMPLE_END.date().isoformat())
    print(f"baseline full 2021-2025: ROI {full['roi']:+.1%}, maxDD {full['dd']:.1%}, "
          f"{full['n']} trades, win {full['win']:.0%}, W/L {full['wl']:.1f}")
    test_rolling(180)
    test_rolling(365)
    test_params()
    test_leave_one_out()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
