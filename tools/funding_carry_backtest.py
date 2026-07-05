#!/usr/bin/env python3
"""Funding-rate carry (cash-and-carry / delta-neutral) backtest.

The simplest real crypto "arb": hold spot LONG + perp SHORT in equal size, so
the position is delta-neutral (no directional exposure to price). The SHORT perp
leg COLLECTS funding whenever funding is positive (longs pay shorts) and PAYS it
when funding is negative. Because the position is delta-neutral, price drift on
the two legs cancels (ignoring basis/tracking error), and the realized P&L of the
strategy is just the stream of funding cashflows on the perp-leg notional:

    per-settlement carry return (on perp notional) = +funding_rate
    cumulative carry          = sum of funding_rate over the held period

This script computes, using REAL Binance perpetual funding history
(data_funding/{SYMBOL}.csv, header `timestamp,funding_rate`, 8h rows where
funding_rate is a fraction, e.g. 0.0001 = 0.01% per 8h):

  PER SYMBOL
    - annualized carry yield, overall and per calendar year (2021-2025)
    - worst drawdown of the cumulative-carry curve (the stretch where funding
      went persistently negative, e.g. 2022 for SOL/BNB)

  EQUAL-WEIGHT 4-SYMBOL "FUNDING HARVEST" PORTFOLIO
    - annualized yield, worst drawdown, and a rough Sharpe
      (annualized mean / annualized std of the per-8h carry series)

GROSS, IDEALIZED carry. This is the yield on the PERP-LEG NOTIONAL and ignores
real-world frictions (capital efficiency ~half because you fund BOTH legs;
spot+perp entry/exit fees; basis/tracking error; having to pay or unwind when
funding flips negative; liquidation risk on the under-margined short perp leg;
exchange/counterparty risk). Realized net is BELOW these numbers. See the report
reports/decision-records/2026-06-03-arb-feasibility.md for the honest haircut discussion.

Annualization: funding settles ~3x/day (every 8h at 00:00/08:00/16:00 UTC), so
there are ~365*3 = 1095 settlements per year. A symbol's annualized yield over a
window = (sum of funding rates) * (1095 / n_settlements_in_window), i.e. mean
per-8h rate * 1095. (Simple/linear annualization, NOT compounded — carry is a
small additive yield, and linear keeps it comparable to the funding-rate
literature and to the momentum reports in this repo.)

Run:  python3 tools/funding_carry_backtest.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FUNDING_DIR = ROOT / "data_funding"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

# Funding settles every 8h -> 3 settlements/day -> ~1095 per year.
SETTLEMENTS_PER_YEAR = 365 * 3  # 1095


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def _parse(ts: str) -> datetime:
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_funding(path: Path) -> list[tuple[datetime, float]]:
    rows: list[tuple[datetime, float]] = []
    with path.open("r", newline="") as h:
        for raw in csv.DictReader(h):
            rows.append((_parse(raw["timestamp"]), float(raw["funding_rate"])))
    rows.sort(key=lambda r: r[0])
    return rows


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
def annualized_yield(rates: list[float]) -> float:
    """Simple (linear, non-compounded) annualized carry yield on perp notional.

    = mean per-settlement rate * settlements/year. A delta-neutral short-perp
    position earns +rate each settlement, so the annualized yield is the average
    8h rate scaled to a year."""
    if not rates:
        return 0.0
    return (sum(rates) / len(rates)) * SETTLEMENTS_PER_YEAR


def worst_drawdown(rates: list[float]) -> tuple[float, int]:
    """Worst peak-to-trough drawdown of the CUMULATIVE carry curve.

    Carry compounds additively (a small yield stream), so the cumulative curve is
    the running sum of funding rates. Drawdown is measured in the same units as
    cumulative carry (a fraction of perp notional) -- it is the total funding GIVEN
    BACK during the worst negative-funding stretch. Returns (drawdown, length)
    where drawdown is a NEGATIVE fraction (0 if the curve only ever rose) and
    length is the number of settlements in that underwater stretch."""
    cum = 0.0
    peak = 0.0
    peak_idx = 0
    worst = 0.0
    worst_len = 0
    for i, r in enumerate(rates):
        cum += r
        if cum > peak:
            peak = cum
            peak_idx = i
        dd = cum - peak
        if dd < worst:
            worst = dd
            worst_len = i - peak_idx
    return worst, worst_len


def sharpe(rates: list[float]) -> float:
    """Rough annualized Sharpe of the per-8h carry series.

    = (annualized mean) / (annualized std). Annualized std = per-settlement std *
    sqrt(settlements/year). Risk-free assumed 0 (this is an excess-style carry
    yield already). No autocorrelation adjustment -- funding is persistent, so the
    true risk-adjusted number is somewhat worse than this i.i.d. estimate."""
    n = len(rates)
    if n < 2:
        return 0.0
    mean = sum(rates) / n
    var = sum((r - mean) ** 2 for r in rates) / (n - 1)
    std = var ** 0.5
    if std == 0:
        return 0.0
    ann_mean = mean * SETTLEMENTS_PER_YEAR
    ann_std = std * (SETTLEMENTS_PER_YEAR ** 0.5)
    return ann_mean / ann_std


def pct_positive(rates: list[float]) -> float:
    if not rates:
        return 0.0
    return sum(1 for r in rates if r > 0) / len(rates)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    if not FUNDING_DIR.is_dir() or not all(
        (FUNDING_DIR / f"{s}.csv").exists() for s in SYMBOLS
    ):
        print("ERROR: missing data_funding/{SYMBOL}.csv for", SYMBOLS)
        return 1

    funding = {s: load_funding(FUNDING_DIR / f"{s}.csv") for s in SYMBOLS}

    print("=" * 74)
    print(" FUNDING-RATE CARRY (delta-neutral: spot LONG + perp SHORT)")
    print(" GROSS yield on perp-leg notional from REAL Binance funding 2021-2025.")
    print(" Short perp COLLECTS funding when funding>0, PAYS when funding<0.")
    print("=" * 74)

    years = ["2021", "2022", "2023", "2024", "2025"]

    # ---- per-symbol -------------------------------------------------------- #
    print("\n--- PER-SYMBOL annualized carry yield (gross, on perp notional) ---")
    header = f"{'symbol':9}{'overall':>10}" + "".join(f"{y:>9}" for y in years)
    header += f"{'%pos':>8}{'worstDD':>10}{'DDlen':>7}"
    print(header)
    for s in SYMBOLS:
        rows = funding[s]
        all_rates = [r for _, r in rows]
        overall = annualized_yield(all_rates)
        by_year = {}
        for ts, r in rows:
            by_year.setdefault(str(ts.year), []).append(r)
        yr_cells = "".join(
            f"{annualized_yield(by_year.get(y, [])):>+9.1%}" for y in years
        )
        dd, dd_len = worst_drawdown(all_rates)
        print(
            f"{s:9}{overall:>+10.2%}{yr_cells}"
            f"{pct_positive(all_rates):>8.0%}{dd:>+10.2%}{dd_len:>7d}"
        )
    print("  worstDD = worst peak-to-trough of the cumulative-carry curve")
    print("            (total funding given back during the worst negative stretch,")
    print("            as a fraction of perp notional); DDlen = its length in 8h steps.")

    # ---- per-symbol settlement counts (transparency) ----------------------- #
    print("\n--- settlement counts (8h funding rows) per symbol/year ---")
    print(f"{'symbol':9}{'total':>8}" + "".join(f"{y:>7}" for y in years))
    for s in SYMBOLS:
        rows = funding[s]
        counts = {}
        for ts, _ in rows:
            counts[str(ts.year)] = counts.get(str(ts.year), 0) + 1
        print(f"{s:9}{len(rows):>8}" + "".join(f"{counts.get(y, 0):>7}" for y in years))
    print("  (SOL 2022 has extra rows: Binance used <8h funding intervals during")
    print("   high-vol stretches; total realized funding is still captured by summing.)")

    # ---- equal-weight portfolio on the common 8h grid ---------------------- #
    # Align all four symbols on the shared 00/08/16 UTC grid. Each symbol's carry
    # for a grid slot = sum of its funding settlements falling in that 8h slot
    # (so off-grid SOL settlements aggregate into the enclosing slot -- no funding
    # lost). Portfolio per-slot carry = mean across the symbols PRESENT in that
    # slot (equal weight). Only slots where all 4 are present are used for the
    # headline so weights are stable.
    def slot_key(ts: datetime) -> datetime:
        h = (ts.hour // 8) * 8
        return ts.replace(hour=h, minute=0, second=0, microsecond=0)

    per_symbol_slot: dict[str, dict[datetime, float]] = {}
    for s in SYMBOLS:
        acc: dict[datetime, float] = {}
        for ts, r in funding[s]:
            k = slot_key(ts)
            acc[k] = acc.get(k, 0.0) + r
        per_symbol_slot[s] = acc

    all_slots = sorted(set().union(*[set(d) for d in per_symbol_slot.values()]))
    common_slots = [
        k for k in all_slots if all(k in per_symbol_slot[s] for s in SYMBOLS)
    ]

    port_rates: list[float] = []
    port_by_year: dict[str, list[float]] = {}
    for k in common_slots:
        carry = sum(per_symbol_slot[s][k] for s in SYMBOLS) / len(SYMBOLS)
        port_rates.append(carry)
        port_by_year.setdefault(str(k.year), []).append(carry)

    print("\n" + "=" * 74)
    print(" EQUAL-WEIGHT 4-SYMBOL DELTA-NEUTRAL 'FUNDING HARVEST' PORTFOLIO")
    print("=" * 74)
    print(f"  common 8h slots (all 4 symbols present): {len(common_slots)}")
    span_days = (common_slots[-1] - common_slots[0]).days if common_slots else 0
    print(f"  span: {common_slots[0].date()} -> {common_slots[-1].date()}  "
          f"(~{span_days/365.25:.1f} yr)")
    print()
    print(f"  annualized yield (gross, on perp notional): "
          f"{annualized_yield(port_rates):>+8.2%}")
    pdd, pdd_len = worst_drawdown(port_rates)
    print(f"  worst drawdown of cumulative carry:         {pdd:>+8.2%}  "
          f"({pdd_len} settlements ~= {pdd_len/3:.0f} days)")
    print(f"  rough Sharpe (ann mean / ann std of 8h carry): {sharpe(port_rates):>6.2f}")
    print(f"  fraction of 8h slots with positive carry:   {pct_positive(port_rates):>8.0%}")

    print("\n  --- portfolio annualized yield per year ---")
    print(f"  {'year':6}{'ann yield':>12}{'worstDD':>11}{'Sharpe':>9}{'%pos':>7}")
    for y in years:
        yr = port_by_year.get(y, [])
        ydd, _ = worst_drawdown(yr)
        print(f"  {y:6}{annualized_yield(yr):>+12.2%}{ydd:>+11.2%}"
              f"{sharpe(yr):>9.2f}{pct_positive(yr):>7.0%}")

    # ---- honesty footer ---------------------------------------------------- #
    print("\n" + "-" * 74)
    print("REAL-WORLD HAIRCUT (these gross numbers IGNORE):")
    print("  * Capital efficiency ~half: you must fund BOTH legs (spot notional +")
    print("    perp margin), so yield on TOTAL deployed capital is roughly half the")
    print("    above perp-notional yield.")
    print("  * Entry/exit fees on spot AND perp (taker ~0.04-0.10% each leg, each way).")
    print("  * Basis / spot-perp tracking error; perp may not converge to spot exactly.")
    print("  * Funding flips negative in bears -- you then PAY (see worst DD / 2022),")
    print("    or must unwind (and pay fees + basis to exit).")
    print("  * Liquidation risk on the under-margined SHORT perp leg in a sharp rally.")
    print("  * Exchange / counterparty / stablecoin risk on the whole position.")
    print("  => realized NET yield is materially below the gross carry shown here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
