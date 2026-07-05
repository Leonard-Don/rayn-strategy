#!/usr/bin/env python3
"""Estimate how much perpetual-funding cost would have eroded the momentum
strategy's returns.

Runs the simulated momentum config (config.rayn-momentum-paper.toml, 3x, 500 USDT)
over data_multicycle (BTC/ETH/SOL/BNB) across the 2021-2025 cycles — the same
setup as tools/momentum_paper_backtest.py — then applies perpetual funding
POST-HOC to the gross trade log.

Perpetual funding is charged every 8h at 00:00 / 08:00 / 16:00 UTC on the
position's notional. This strategy is long-only (allow_short=false), so every
position is side=+1 and a POSITIVE funding rate is a COST (longs pay).

    funding_cost_for_trade = notional * sum(funding_rate at each 8h settlement
                                            t with entry_time <= t < exit_time)
    net_pnl = trade.pnl - funding_cost_for_trade
    net ROI (period) = sum(net_pnl over trades in period) / initial_equity

Two modes:
  (a) CONSTANT-RATE SENSITIVITY (no external data): assume a constant annualized
      funding rate; per-8h rate = r_annual / (365 * 3). Sweep r_annual in
      {0%, 5%, 10%, 20%, 30%} and report net-of-funding ROI full + per cycle.
  (b) REAL-FUNDING (only if data_funding/ exists): load data_funding/{SYMBOL}.csv
      with header `timestamp,funding_rate` (8h rows) and sum the REAL rates at the
      settlements within each trade's window. Skipped with a note if data missing.

CAVEAT: funding is applied post-hoc on the GROSS trade log; it does NOT compound
into position sizing during the run. Because gross notional grows faster than the
funding-eroded path would, this slightly OVERESTIMATES the dollar drag in later
periods — i.e. it is a conservative (upper-bound-ish) estimate of the drag.

Run:  python3 tools/momentum_funding_cost.py
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

# Same run/setup pattern as tools/momentum_paper_backtest.py:
# simulated 3x config, account halt disabled (set to 0.99) to show the
# true gross trade log without the monitoring halt overlay.
_PAPER_CFG = load_config(ROOT / "config.rayn-momentum-paper.toml")
CFG = replace(_PAPER_CFG, risk=replace(_PAPER_CFG.risk, max_account_drawdown_pct=0.99))
MOM = MomentumConfig(**load_raw_config(ROOT / "config.rayn-momentum-paper.toml")["momentum"])
DATA = {s: load_candles(ROOT / "data_multicycle" / f"{s}.csv") for s in SYMBOLS}

CYCLES = [
    ("2021 bull->top", "2021-01-01", "2022-01-01"),
    ("2022 BEAR", "2022-01-01", "2023-01-01"),
    ("2023 recovery", "2023-01-01", "2024-01-01"),
    ("2024-25 bull+10-10", "2024-01-01", "2025-10-13"),
    ("FULL 2021->2025-10", "2021-01-01", "2025-10-13"),
]

# Constant annualized funding rates to sweep (mode a).
R_ANNUAL_SWEEP = [0.0, 0.05, 0.10, 0.20, 0.30]
# Representative rate for the per-cycle breakdown table.
REPRESENTATIVE_R = 0.20

FUNDING_PER_DAY = 3  # 8h settlements per day at 00:00 / 08:00 / 16:00 UTC


# --------------------------------------------------------------------------- #
# run the strategy (gross trade log)
# --------------------------------------------------------------------------- #
def run(entry_start, run_end):
    return MomentumEngine(
        config=CFG, mom=MOM, data_dir=ROOT / "data_multicycle", symbols=SYMBOLS,
        preloaded_data=DATA, entry_start=entry_start, run_end=run_end,
    ).run()


# --------------------------------------------------------------------------- #
# 8h funding settlement timestamps within a trade window
# --------------------------------------------------------------------------- #
def _parse(ts: str) -> datetime:
    """Parse an ISO8601 timestamp (with trailing Z or +00:00) to UTC datetime."""
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def settlement_times(entry_time: str, exit_time: str) -> list[datetime]:
    """8h funding settlement instants t (00:00/08:00/16:00 UTC) with
    entry_time <= t < exit_time."""
    start = _parse(entry_time)
    end = _parse(exit_time)
    # first settlement at or after start
    base = start.replace(minute=0, second=0, microsecond=0)
    hour_to_next = (8 - base.hour % 8) % 8
    t = base + timedelta(hours=hour_to_next)
    if t < start:  # base was already past start due to sub-hour entry_time
        t += timedelta(hours=8)
    out = []
    while t < end:
        out.append(t)
        t += timedelta(hours=8)
    return out


# --------------------------------------------------------------------------- #
# mode (a): constant-rate funding cost
# --------------------------------------------------------------------------- #
def funding_cost_constant(trade, r_annual: float) -> float:
    """funding_cost = notional * sum over settlements of per-8h rate.
    Long-only: positive rate is a cost, so cost = notional * n_settlements * per8h."""
    per_8h = r_annual / (365 * FUNDING_PER_DAY)
    n = len(settlement_times(trade.entry_time, trade.exit_time))
    return trade.notional * n * per_8h


# --------------------------------------------------------------------------- #
# mode (b): real funding from data_funding/{SYMBOL}.csv
# --------------------------------------------------------------------------- #
def load_funding(path: Path) -> list[tuple[datetime, float]]:
    import csv
    rows: list[tuple[datetime, float]] = []
    with path.open("r", newline="") as h:
        for raw in csv.DictReader(h):
            rows.append((_parse(raw["timestamp"]), float(raw["funding_rate"])))
    rows.sort(key=lambda r: r[0])
    return rows


def funding_cost_real(trade, funding_by_symbol: dict[str, dict[datetime, float]]) -> float:
    table = funding_by_symbol.get(trade.symbol, {})
    total_rate = 0.0
    for t in settlement_times(trade.entry_time, trade.exit_time):
        rate = table.get(t)
        if rate is not None:
            total_rate += rate
    return trade.notional * total_rate


# --------------------------------------------------------------------------- #
# ROI helpers
# --------------------------------------------------------------------------- #
def gross_roi(result) -> float:
    return sum(t.pnl for t in result.trades) / result.initial_equity


def net_roi_constant(result, r_annual: float) -> float:
    init = result.initial_equity
    net = sum(t.pnl - funding_cost_constant(t, r_annual) for t in result.trades)
    return net / init


def net_roi_real(result, funding_by_symbol) -> float:
    init = result.initial_equity
    net = sum(t.pnl - funding_cost_real(t, funding_by_symbol) for t in result.trades)
    return net / init


def total_funding_dollars(result, r_annual: float) -> float:
    return sum(funding_cost_constant(t, r_annual) for t in result.trades)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=== Momentum funding-cost drag (PAPER config, 3x, $500, long-only) ===")
    print("Funding charged every 8h (00:00/08:00/16:00 UTC) on position notional.")
    print("Long-only: a POSITIVE funding rate is a COST. net_pnl = pnl - funding_cost.\n")

    # run every cycle once, keep the gross result
    results = {lbl: run(es, re) for lbl, es, re in CYCLES}

    # ---- mode (a): constant-rate sensitivity -------------------------------
    print("--- Mode (a): CONSTANT-RATE SENSITIVITY (net-of-funding ROI) ---")
    cols = "".join(f"{f'{int(r*100)}%/yr':>9}" for r in R_ANNUAL_SWEEP)
    print(f"{'period':22}{cols}")
    for lbl, _, _ in CYCLES:
        r = results[lbl]
        cells = "".join(f"{net_roi_constant(r, ra):>+9.1%}" for ra in R_ANNUAL_SWEEP)
        print(f"{lbl:22}{cells}")

    full = results["FULL 2021->2025-10"]
    g = gross_roi(full)
    print()
    print("Full-period 2021->2025-10 summary:")
    print(f"  trades: {len(full.trades)}")
    print(f"  gross ROI (no funding):                {g:>+8.1%}")
    for ra in R_ANNUAL_SWEEP:
        net = net_roi_constant(full, ra)
        drag = total_funding_dollars(full, ra)
        print(f"  net ROI @ {int(ra*100):>2d}%/yr funding:  {net:>+8.1%}   "
              f"(funding drag ${drag:,.0f} on ${full.initial_equity:,.0f})")

    # ---- per-cycle breakdown at representative rate ------------------------
    print(f"\n--- Per-cycle breakdown @ {int(REPRESENTATIVE_R*100)}%/yr funding "
          f"(representative) ---")
    print(f"{'period':22}{'gross ROI':>12}{'funding $':>12}{'net ROI':>12}")
    for lbl, _, _ in CYCLES:
        r = results[lbl]
        print(f"{lbl:22}{gross_roi(r):>+12.1%}"
              f"{total_funding_dollars(r, REPRESENTATIVE_R):>12,.0f}"
              f"{net_roi_constant(r, REPRESENTATIVE_R):>+12.1%}")

    # ---- mode (b): real funding (only if data present) ---------------------
    print("\n--- Mode (b): REAL-FUNDING (data_funding/) ---")
    funding_dir = ROOT / "data_funding"
    have_all = funding_dir.is_dir() and all(
        (funding_dir / f"{s}.csv").exists() for s in SYMBOLS
    )
    if not have_all:
        missing = []
        if not funding_dir.is_dir():
            missing.append("data_funding/ directory")
        else:
            missing = [f"data_funding/{s}.csv" for s in SYMBOLS
                       if not (funding_dir / f"{s}.csv").exists()]
        print("SKIPPED — real-funding mode needs the funding CSVs.")
        print(f"  Missing: {', '.join(missing)}")
        print("  Provide data_funding/{SYMBOL}.csv with header `timestamp,funding_rate`")
        print("  (8h rows, funding_rate a float fraction like 0.0001, timestamp ISO8601 UTC).")
    else:
        funding_by_symbol = {
            s: dict(load_funding(funding_dir / f"{s}.csv")) for s in SYMBOLS
        }
        print(f"{'period':22}{'gross ROI':>12}{'net-of-real':>14}")
        for lbl, _, _ in CYCLES:
            r = results[lbl]
            print(f"{lbl:22}{gross_roi(r):>+12.1%}"
                  f"{net_roi_real(r, funding_by_symbol):>+14.1%}")

    # ---- caveat ------------------------------------------------------------
    print("\nCAVEAT: funding is applied POST-HOC on the gross trade log; it does NOT")
    print("compound into position sizing during the run. This slightly OVERESTIMATES")
    print("the dollar drag in later periods — i.e. a conservative (upper-bound-ish)")
    print("estimate of the funding drag.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
