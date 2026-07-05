#!/usr/bin/env python3
"""Correctness audit probes for the Rayn backtest and grid engines.

This does not change strategy behavior. It runs the existing engines and checks
invariants that, if violated, would mean published report numbers are unreliable:

1. PnL accounting reconciliation: sum(trade.pnl) == final_equity - initial_equity.
2. Timestamp-intersection truncation: does adding a late-listed symbol silently
   shorten the common backtest window for every symbol?
3. Sizing semantics: nominal risk_per_trade vs the true notional / true stop risk
   once rescue layers are stacked.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rayn_strategy.backtest import BacktestEngine, load_candles, normalize_timestamp
from src.rayn_strategy.config import load_config, load_raw_config
from src.rayn_strategy.grid import BoundedGridConfig, BoundedGridEngine


ROOT = Path(__file__).resolve().parent.parent


def load_grid_config(path: Path) -> BoundedGridConfig:
    return BoundedGridConfig(**load_raw_config(path)["grid"])


def common_window(data_dir: Path, symbols: list[str]) -> tuple[str, str, int]:
    sets = []
    for sym in symbols:
        rows = load_candles(data_dir / f"{sym}.csv")
        sets.append({r.timestamp for r in rows})
    common = sorted(set.intersection(*sets))
    return common[0], common[-1], len(common)


def check_reconciliation(label: str, result) -> None:
    pnl_sum = sum(t.pnl for t in result.trades)
    equity_delta = result.final_equity - result.initial_equity
    diff = pnl_sum - equity_delta
    ok = abs(diff) < 1e-6
    print(f"  [{label}]")
    print(f"    trades={len(result.trades)} halted={result.halted}")
    print(f"    sum(trade.pnl)      = {pnl_sum:14.6f}")
    print(f"    final - initial     = {equity_delta:14.6f}")
    print(f"    reconciliation diff = {diff:14.2e}  -> {'OK' if ok else 'MISMATCH'}")


def probe_1_intersection() -> None:
    print("\n=== PROBE 1: timestamp-intersection truncation (data_2024_2025) ===")
    data_dir = ROOT / "data_2024_2025"
    for symbols in (
        ["BTCUSDT", "ZECUSDT"],
        ["BTCUSDT", "ZECUSDT", "ONDOUSDT"],
        ["BTCUSDT", "ZECUSDT", "ONDOUSDT", "HYPEUSDT"],
        ["BTCUSDT", "ETHUSDT", "ZECUSDT", "ONDOUSDT", "HYPEUSDT"],
    ):
        first, last, n = common_window(data_dir, symbols)
        print(f"  {'+'.join(symbols):42s} window {first[:10]} -> {last[:10]}  bars={n}")
    print("  NOTE: if these windows differ, 'full 2024-2025' comparisons across")
    print("  symbol sets are NOT same-period comparisons.")


def probe_2_reconciliation() -> None:
    print("\n=== PROBE 2: PnL accounting reconciliation ===")
    # Non-grid engine on 2026 data.
    cfg = load_config(ROOT / "config.optimized.toml")
    eng = BacktestEngine(cfg, ROOT / "data", ["BTCUSDT", "ETHUSDT", "ZECUSDT"])
    check_reconciliation("backtest / optimized / BTC,ETH,ZEC / 2026", eng.run())

    # Grid engine, target window.
    gcfg = ROOT / "config.rayn-regime-filter.toml"
    grid = BoundedGridEngine(
        config=load_config(gcfg),
        grid_config=load_grid_config(gcfg),
        data_dir=ROOT / "data_2024_2025",
        symbols=["BTCUSDT", "ZECUSDT", "ONDOUSDT"],
        entry_start="2025-06-17",
        entry_end="2025-09-15",
        run_end="2025-09-15",
    )
    check_reconciliation("grid / regime-filter / BTC,ZEC,ONDO / target window", grid.run())

    # Grid engine, full 2024-2025 (the run reported as Halted).
    grid_full = BoundedGridEngine(
        config=load_config(gcfg),
        grid_config=load_grid_config(gcfg),
        data_dir=ROOT / "data_2024_2025",
        symbols=["BTCUSDT", "ZECUSDT", "ONDOUSDT"],
    )
    check_reconciliation("grid / regime-filter / BTC,ZEC,ONDO / full 2024-2025", grid_full.run())


def probe_3_sizing() -> None:
    print("\n=== PROBE 3: sizing semantics (nominal vs true risk) ===")
    gcfg_path = ROOT / "config.rayn-regime-filter.toml"
    cfg = load_config(gcfg_path)
    gcfg = load_grid_config(gcfg_path)
    equity = cfg.portfolio.initial_equity
    rpt = cfg.risk.risk_per_trade_pct
    strat_stop = cfg.strategy.stop_loss_pct
    raw = equity * rpt / strat_stop
    cap = equity * cfg.risk.max_position_notional_pct
    base = max(0.0, min(raw, cap))
    print(f"  initial_equity            = {equity:,.0f}")
    print(f"  risk_per_trade_pct        = {rpt:.6f}")
    print(f"  strategy.stop_loss_pct    = {strat_stop:.4f}  (used by position_notional)")
    print(f"  grid.hard_stop_pct        = {gcfg.hard_stop_pct:.4f}  (actual grid stop)")
    print(f"  raw notional = eq*rpt/stop= {raw:,.2f}  ({raw/equity:.2%} of equity)")
    print(f"  cap = eq*max_pos_notional = {cap:,.2f}  ({cap/equity:.2%} of equity)")
    print(f"  base layer notional       = {base:,.2f}  ({base/equity:.2%} of equity)")
    # Stack all rescue layers.
    total = sum(base * (gcfg.layer_multiplier ** k) for k in range(gcfg.max_rescue_layers + 1))
    print(f"  full-stack notional (all {gcfg.max_rescue_layers} rescue layers)"
          f" = {total:,.2f}  ({total/equity:.2f}x equity)")
    nominal_risk = base * strat_stop
    true_risk_full = total * gcfg.hard_stop_pct
    print(f"  nominal risk/trade (base*strat_stop)      = {nominal_risk:,.2f}"
          f"  ({nominal_risk/equity:.2%} of equity)")
    print(f"  true risk if full stack hits hard_stop    = {true_risk_full:,.2f}"
          f"  ({true_risk_full/equity:.2%} of equity)")
    print(f"  true/nominal ratio                        = {true_risk_full/nominal_risk:,.1f}x")


def _roi(result) -> float:
    return (result.final_equity - result.initial_equity) / result.initial_equity


def probe_4_window_vs_symbol() -> None:
    print("\n=== PROBE 4: is 'adding HYPE hurts' a window effect or a symbol effect? ===")
    gcfg = ROOT / "config.rayn-regime-filter.toml"
    cfg = load_config(gcfg)
    grid_cfg = load_grid_config(gcfg)
    data_dir = ROOT / "data_2024_2025"

    def run(symbols, entry_start=None):
        return BoundedGridEngine(
            config=cfg, grid_config=grid_cfg, data_dir=data_dir,
            symbols=symbols, entry_start=entry_start,
        ).run()

    a = run(["BTCUSDT", "ZECUSDT"])                              # full window
    b = run(["BTCUSDT", "ZECUSDT"], entry_start="2025-05-30")    # HYPE's window, same symbols
    c = run(["BTCUSDT", "ZECUSDT", "HYPEUSDT"])                  # auto-truncated to HYPE's window
    print(f"  (a) BTC,ZEC        full 2024-2025          ROI={_roi(a):+7.2%}  DD={a.max_drawdown_pct:.2%}  halted={a.halted}  trades={len(a.trades)}")
    print(f"  (b) BTC,ZEC        from 2025-05-30 (HYPE win) ROI={_roi(b):+7.2%}  DD={b.max_drawdown_pct:.2%}  halted={b.halted}  trades={len(b.trades)}")
    print(f"  (c) BTC,ZEC,HYPE   full -> truncated 2025-05-30 ROI={_roi(c):+7.2%}  DD={c.max_drawdown_pct:.2%}  halted={c.halted}  trades={len(c.trades)}")
    print("  (a)->(b): pure WINDOW effect (same symbols, different period).")
    print("  (b)->(c): pure SYMBOL effect (same period, +HYPE).")


def main() -> int:
    probe_1_intersection()
    probe_2_reconciliation()
    probe_3_sizing()
    probe_4_window_vs_symbol()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
