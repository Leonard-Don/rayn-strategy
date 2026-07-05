#!/usr/bin/env python3
"""How much does realistic slippage erode the momentum mirror? It enters on
BREAKOUTS (buys into upward momentum = chasing) and exits on TRAILING STOPS (sells
into down moves) — the two worst cases for slippage, especially on alts. The
backtest models a flat 5bps; this stress-tests that.

Three views, on the 3x paper config over data_multicycle (2021-2025, $500):
  A. Flat slippage_bps sweep (faithful re-run, compounds) — 5/10/20/30/50 bps.
  B. Volatility-scaled slippage = c x ATR% at the fill bar, per side (post-hoc) —
     realistic because slippage is worst exactly when the strategy trades (volatile
     breakouts / stop flushes). c in {0.25, 0.5, 1.0} of an ATR.
  C. ALL-IN = realistic slippage (c=0.5) + REAL Binance funding (data_funding/).

Post-hoc models (B, C) apply cost on the gross trade log without compounding into
sizing -> a slightly conservative (upper-bound) drag estimate.
"""

from __future__ import annotations

import bisect
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.config import load_config, load_raw_config
from src.rayn_strategy.backtest import load_candles
from src.rayn_strategy.momentum import MomentumConfig, MomentumEngine, atr_pct

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DATA = {s: load_candles(ROOT / "data_multicycle" / f"{s}.csv") for s in SYMBOLS}
BASE = replace(load_config(ROOT / "config.rayn-momentum-paper.toml"),
               risk=replace(load_config(ROOT / "config.rayn-momentum-paper.toml").risk,
                            max_account_drawdown_pct=0.99))
MOM = MomentumConfig(**load_raw_config(ROOT / "config.rayn-momentum-paper.toml")["momentum"])
INIT = BASE.portfolio.initial_equity
CYCLES = [("2021 bull", "2021-01-01", "2022-01-01"), ("2022 BEAR", "2022-01-01", "2023-01-01"),
          ("2023 recovery", "2023-01-01", "2024-01-01"), ("2024-25 bull", "2024-01-01", "2025-10-13"),
          ("FULL", "2021-01-01", "2025-10-13")]


def run(slippage_bps, es=None, re=None):
    cfg = replace(BASE, portfolio=replace(BASE.portfolio, slippage_bps=slippage_bps))
    return MomentumEngine(config=cfg, mom=MOM, data_dir=ROOT / "data_multicycle", symbols=SYMBOLS,
                          preloaded_data=DATA, entry_start=es, run_end=re).run()


def roi(r):
    return (r.final_equity - r.initial_equity) / r.initial_equity


# ---- precompute ATR% per symbol per timestamp (for the volatility-scaled model) ----
def atr_index():
    idx = {}
    for s in SYMBOLS:
        rows = DATA[s]
        highs, lows, closes = [], [], []
        d = {}
        for c in rows:
            highs.append(c.high); lows.append(c.low); closes.append(c.close)
            a = atr_pct(highs, lows, closes, MOM.atr_bars)
            d[c.timestamp] = a if a is not None else 0.0
        idx[s] = d
    return idx


# ---- real funding lookup (8h settlements within a trade window) ----
def funding_index():
    fdir = ROOT / "data_funding"
    if not all((fdir / f"{s}.csv").exists() for s in SYMBOLS):
        return None
    idx = {}
    for s in SYMBOLS:
        ts, rates = [], []
        for line in (fdir / f"{s}.csv").read_text().splitlines()[1:]:
            t, r = line.split(",")
            ts.append(t); rates.append(float(r))
        idx[s] = (ts, rates)
    return idx


def funding_drag(trades, fidx):
    total = 0.0
    for t in trades:
        ts, rates = fidx[t.symbol]
        lo = bisect.bisect_left(ts, t.entry_time)
        hi = bisect.bisect_left(ts, t.exit_time)
        total += t.notional * sum(rates[lo:hi])  # long pays positive funding
    return total


def slip_drag(trades, aidx, c):
    """Extra slippage = c * ATR% at entry bar + c * ATR% at exit bar, on notional."""
    total = 0.0
    for t in trades:
        ae = aidx[t.symbol].get(t.entry_time, 0.0)
        ax = aidx[t.symbol].get(t.exit_time, 0.0)
        total += t.notional * c * (ae + ax)
    return total


def main() -> int:
    print("Momentum mirror — slippage stress (3x paper config, $500, BTC/ETH/SOL/BNB)\n")

    # A. flat slippage sweep (faithful re-run)
    print("## A. Flat slippage_bps sweep (faithful, compounds; includes 5bps taker fee)")
    print(f"   {'slippage/side':16}{'FULL ROI':>11}{'maxDD':>8}")
    for sbps in (5, 10, 20, 30, 50):
        r = run(sbps, "2021-01-01", "2025-10-13")
        tag = "  <- current" if sbps == 5 else ""
        print(f"   {sbps:>6} bps      {roi(r):+11.1%}{r.max_drawdown_pct:8.1%}{tag}")

    # fee-only trade log (slippage_bps=0) for the post-hoc models
    full = run(0.0, "2021-01-01", "2025-10-13")
    gross = sum(t.pnl for t in full.trades)
    print(f"\n   reference (fee-only, no slippage): {gross/INIT:+.1%}, {len(full.trades)} trades")

    aidx = atr_index()
    print("\n## B. Volatility-scaled slippage = c x ATR% per side (realistic; post-hoc, full sample)")
    print(f"   {'c (fraction of ATR)':24}{'slip $':>9}{'net ROI':>10}")
    for c in (0.25, 0.5, 1.0):
        d = slip_drag(full.trades, aidx, c)
        print(f"   c = {c:<20}{d:9.0f}{(gross - d)/INIT:+10.1%}")

    # C. all-in: realistic slippage (c=0.5) + real funding
    fidx = funding_index()
    print("\n## C. ALL-IN bottom line (per cycle): + realistic slippage (c=0.5) + real funding")
    print(f"   {'cycle':16}{'gross':>9}{'+slip':>9}{'+funding':>10}{'= net':>9}")
    for lbl, es, re in CYCLES:
        r = run(0.0, es, re)
        g = sum(t.pnl for t in r.trades)
        sd = slip_drag(r.trades, aidx, 0.5)
        fd = funding_drag(r.trades, fidx) if fidx else 0.0
        net = g - sd - fd
        print(f"   {lbl:16}{g/INIT:+9.1%}{-sd/INIT:+9.1%}{-fd/INIT:+10.1%}{net/INIT:+9.1%}")
    if not fidx:
        print("   (data_funding/ missing — funding column shows 0; run fetch_funding.py)")

    print("\nCaveat: B/C are post-hoc on the gross trade log (no compounding into sizing)")
    print("=> conservative (upper-bound) drag. c=0.5 ATR/side is pessimistic for liquid majors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
