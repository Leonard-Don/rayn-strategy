#!/usr/bin/env python3
"""Simulated momentum-monitoring flow.

Produces order intents only; never places an order or moves money.

It reuses MomentumEngine's exact decision primitives (_entry_long/_entry_short/
_regime/_liq + atr_pct) and replicates the engine's per-bar exit/trailing/sizing,
so a paper step is bit-identical to one iteration of the backtest loop (verified
by tests/test_momentum_paper.py). State persists to a JSON file between runs.

Usage:
  # dry-run replay: step the last N bars of a data dir through a fresh paper book
  python3 tools/momentum_paper.py --config config.rayn-momentum-paper.toml \
      --data data_multicycle --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT --demo 720

  # recent-data tick: step the latest bar and persist local state
  python3 tools/momentum_paper.py --config ... --data <fresh> --symbols ... --state reports/runtime/momentum-paper-state.json --tick

For a monitoring loop: refresh klines into --data, then run with --tick.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.backtest import load_candles
from src.rayn_strategy.config import load_config, load_raw_config
from src.rayn_strategy.momentum import MomentumConfig, MomentumEngine, MomPosition, atr_pct


@dataclass
class Intent:
    action: str       # "OPEN" / "CLOSE"
    symbol: str
    side: str         # "BUY" / "SELL"
    notional: float
    price: float
    stop: float | None
    reason: str
    fee_cost: float = 0.0
    slippage_cost: float = 0.0
    realized_pnl: float | None = None
    cash_delta: float = 0.0
    quality_score: float | None = None
    risk_multiplier: float | None = None


class MomentumPaper:
    """Stateful paper book. step() processes ONE bar (the latest) per symbol."""

    def __init__(self, config, mom: MomentumConfig):
        self.config = config
        self.mom = mom
        self.lev = config.risk.leverage
        self.benchmark = config.circuit_breaker.benchmark_symbol
        # decision primitives are reused from the engine (no preloaded data needed for helpers)
        self._eng = MomentumEngine(config=config, mom=mom, data_dir=ROOT, symbols=[self.benchmark])
        self.cash = config.portfolio.initial_equity
        self.peak = self.cash
        self.halted = False
        self.last_timestamp: str | None = None
        self.positions: dict[str, MomPosition] = {}

    # ---- state persistence ----
    def to_dict(self) -> dict:
        return {"cash": self.cash, "peak": self.peak, "halted": self.halted,
                "last_timestamp": self.last_timestamp,
                "positions": {s: asdict(p) for s, p in self.positions.items()}}

    def load(self, d: dict) -> None:
        self.cash = d["cash"]; self.peak = d["peak"]; self.halted = d.get("halted", False)
        self.last_timestamp = d.get("last_timestamp")
        self.positions = {s: MomPosition(**p) for s, p in d.get("positions", {}).items()}

    def _marked(self, prices: dict[str, float]) -> float:
        u = sum(p.side * p.notional * ((prices[s] / p.entry_price) - 1.0)
                for s, p in self.positions.items())
        return self.cash + u

    def _close(self, p: MomPosition, price: float, reason: str) -> Intent:
        gross = p.side * p.notional * ((price / p.entry_price) - 1.0)
        fee, slippage, cost = _cost_components(p.notional, self.config)
        cash_delta = gross - cost
        self.cash += cash_delta
        return Intent("CLOSE", p.symbol, "SELL" if p.side > 0 else "BUY",
                      p.notional, price, None, reason,
                      fee_cost=fee,
                      slippage_cost=slippage,
                      realized_pnl=gross - 2 * cost,
                      cash_delta=cash_delta)

    def _short_rebound_filter_blocks(self, hist: dict[str, dict]) -> bool:
        bars = self.mom.short_rebound_filter_bars
        if bars <= 0 or self.mom.short_rebound_filter_min_pct <= 0.0:
            return False
        symbols = tuple(self.mom.short_rebound_filter_symbols)
        min_count = self.mom.short_rebound_filter_min_count or len(symbols)
        rebounding = 0

        for symbol in symbols:
            closes = hist.get(symbol, {}).get("closes", [])
            if len(closes) <= bars or closes[-(bars + 1)] <= 0:
                continue
            change = closes[-1] / closes[-(bars + 1)] - 1.0
            if change >= self.mom.short_rebound_filter_min_pct:
                rebounding += 1

        bench = hist.get(self.benchmark, {}).get("closes", [])
        if len(bench) > bars and bench[-(bars + 1)] > 0:
            bench_change = bench[-1] / bench[-(bars + 1)] - 1.0
            if bench_change < self.mom.short_rebound_filter_benchmark_min_pct:
                return False

        return rebounding >= min_count

    def step(self, symbols: list[str], hist: dict[str, dict]) -> list[Intent]:
        """hist[s] = {'closes':[...], 'highs':[...], 'lows':[...], 'bar': Candle-like dict
        with open/high/low/close} — all up to & including the current (latest) bar."""
        intents: list[Intent] = []
        timestamp = hist[symbols[0]]["bar"]["ts"]
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return intents
        prices = {s: hist[s]["bar"]["close"] for s in symbols}

        # 1. manage open positions (exits + trailing) on the latest bar
        for s, p in list(self.positions.items()):
            p.bars_held += 1
            bar = hist[s]["bar"]
            liq = self._eng._liq(p.entry_price, p.side, self.lev)
            reason = px = None
            if p.side > 0:
                if bar["low"] <= liq and liq >= p.stop_price:
                    reason, px = "liquidation", liq
                elif bar["low"] <= p.stop_price:
                    reason, px = "trail_stop", min(p.stop_price, bar["open"])
            else:
                if bar["high"] >= liq and liq <= p.stop_price:
                    reason, px = "liquidation", liq
                elif bar["high"] >= p.stop_price:
                    reason, px = "trail_stop", max(p.stop_price, bar["open"])
            if reason is None and p.bars_held >= self.mom.max_hold_bars:
                reason, px = "time_stop", bar["close"]
            if reason:
                intents.append(self._close(p, px, reason))
                del self.positions[s]
            else:
                a = atr_pct(hist[s]["highs"], hist[s]["lows"], hist[s]["closes"], self.mom.atr_bars)
                if p.side > 0:
                    p.water = max(p.water, bar["close"])
                    if a is not None:
                        p.stop_price = max(p.stop_price, p.water * (1.0 - self.mom.trail_atr_mult * a))
                else:
                    p.water = min(p.water, bar["close"])
                    if a is not None:
                        p.stop_price = min(p.stop_price, p.water * (1.0 + self.mom.trail_atr_mult * a))

        # 2. account-level halt: flatten everything at 20% account drawdown, then stop
        marked = self._marked(prices)
        self.peak = max(self.peak, marked)
        dd = (self.peak - marked) / self.peak if self.peak > 0 else 0.0
        if not self.halted and dd >= self.config.risk.max_account_drawdown_pct:
            for s, p in list(self.positions.items()):
                intents.append(self._close(p, prices[s], "account_halt"))
                del self.positions[s]
            self.halted = True
            self.last_timestamp = timestamp
            return intents
        if self.halted:
            self.last_timestamp = timestamp
            return intents

        # 3. entries
        regime = self._eng._regime(hist[self.benchmark]["closes"])
        long_breadth_ok = self._eng._long_breadth_ok({s: hist[s]["closes"] for s in symbols})
        short_rebound_blocked = self._short_rebound_filter_blocks(hist)
        for s in symbols:
            if s in self.positions or len(self.positions) >= self.config.risk.max_open_positions:
                continue
            a = atr_pct(hist[s]["highs"], hist[s]["lows"], hist[s]["closes"], self.mom.atr_bars)
            if a is None or a <= 0:
                continue
            side = 0
            if regime > 0 and self.mom.allow_long and long_breadth_ok and self._eng._entry_long(hist[s]["closes"]):
                side = 1
            elif (
                regime < 0
                and self.mom.allow_short
                and not short_rebound_blocked
                and self._eng._entry_short(hist[s]["closes"])
            ):
                side = -1
            if side == 0:
                continue
            quality = self._eng._entry_quality(side, hist[s]["closes"])
            if quality.risk_multiplier <= 0.0:
                continue
            price = prices[s]
            stop_dist = self.mom.stop_atr_mult * a
            notional = (marked * self.mom.risk_per_trade_pct * quality.risk_multiplier) / stop_dist
            notional = min(notional, marked * self.config.risk.max_position_notional_pct)
            total = sum(q.notional for q in self.positions.values())
            cap = marked * self.config.risk.max_total_notional_pct
            if total + notional > cap:
                notional = max(0.0, cap - total)
            if notional <= 0:
                continue
            fee, slippage, cost = _cost_components(notional, self.config)
            self.cash -= cost
            stop = price * (1.0 - stop_dist) if side > 0 else price * (1.0 + stop_dist)
            self.positions[s] = MomPosition(s, hist[s]["bar"]["ts"], price, notional, side, stop, price)
            intents.append(Intent("OPEN", s, "BUY" if side > 0 else "SELL", notional, price, stop,
                                  "momentum_long" if side > 0 else "momentum_short",
                                  fee_cost=fee,
                                  slippage_cost=slippage,
                                  cash_delta=-cost,
                                  quality_score=quality.score,
                                  risk_multiplier=quality.risk_multiplier))
        self.last_timestamp = timestamp
        return intents


def _cost_components(notional: float, config) -> tuple[float, float, float]:
    fee = notional * config.portfolio.taker_fee_bps / 10_000.0
    slippage = notional * config.portfolio.slippage_bps / 10_000.0
    return fee, slippage, fee + slippage


def _hist_upto(rows_by_sym, symbols, i, tail: int | None = None):
    # Indicators look back <= ~201 bars, so a bounded tail gives identical decisions
    # (and avoids O(n^2) full-history copies on long replays). tail=None => full history.
    lo = max(0, i + 1 - tail) if tail else 0
    h = {}
    for s in symbols:
        rows = rows_by_sym[s][lo: i + 1]
        bar = rows[-1]
        h[s] = {"closes": [r.close for r in rows], "highs": [r.high for r in rows],
                "lows": [r.low for r in rows],
                "bar": {"ts": bar.timestamp, "open": bar.open, "high": bar.high,
                        "low": bar.low, "close": bar.close}}
    return h


def _catch_up_start_index(ts: list[str], last_timestamp: str | None, warm: int) -> int:
    if not ts:
        return 0
    if last_timestamp is None:
        return max(warm, len(ts) - 1)
    for i, timestamp in enumerate(ts):
        if timestamp > last_timestamp:
            return max(warm, i)
    return len(ts)


def _print_intent(timestamp: str, intent: Intent) -> None:
    stop = f" stop={intent.stop}" if intent.stop is not None else ""
    print(f"  {timestamp[:16]} {intent.action} {intent.side} {intent.symbol} "
          f"${intent.notional:.2f} @ {intent.price:.4g}{stop} [{intent.reason}]")


def append_intent_ledger(
    path: Path,
    timestamp: str,
    intents: list[Intent],
    *,
    equity_after: float,
) -> None:
    if not intents:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "action",
        "symbol",
        "side",
        "notional",
        "price",
        "stop",
        "reason",
        "fee_cost",
        "slippage_cost",
        "total_cost",
        "realized_pnl",
        "cash_delta",
        "quality_score",
        "risk_multiplier",
        "equity_after",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for intent in intents:
            total_cost = intent.fee_cost + intent.slippage_cost
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "action": intent.action,
                    "symbol": intent.symbol,
                    "side": intent.side,
                    "notional": _fmt_float(intent.notional),
                    "price": _fmt_float(intent.price),
                    "stop": _fmt_float(intent.stop),
                    "reason": intent.reason,
                    "fee_cost": _fmt_float(intent.fee_cost),
                    "slippage_cost": _fmt_float(intent.slippage_cost),
                    "total_cost": _fmt_float(total_cost),
                    "realized_pnl": _fmt_float(intent.realized_pnl),
                    "cash_delta": _fmt_float(intent.cash_delta),
                    "quality_score": _fmt_float(intent.quality_score),
                    "risk_multiplier": _fmt_float(intent.risk_multiplier),
                    "equity_after": _fmt_float(equity_after),
                }
            )


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.rayn-momentum-paper.toml")
    ap.add_argument("--data", default="data_multicycle")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    ap.add_argument("--state", default=None, help="JSON state file (persist between runs)")
    ap.add_argument("--ledger", default=None, help="CSV ledger for persisted paper order intents")
    ap.add_argument("--demo", type=int, default=0, help="replay the last N bars from a FRESH book (dry run)")
    ap.add_argument("--tick", action="store_true", help="step ONLY the latest bar using persisted state")
    ap.add_argument(
        "--catch-up",
        action="store_true",
        help="step every bar newer than persisted last_timestamp, in order",
    )
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    mom = MomentumConfig(**load_raw_config(ROOT / args.config)["momentum"])
    symbols = args.symbols.split(",")
    rows = {s: load_candles(ROOT / args.data / f"{s}.csv") for s in symbols}
    ts = sorted(set.intersection(*(set(r.timestamp for r in v) for v in rows.values())))
    rows = {s: [r for r in rows[s] if r.timestamp in set(ts)] for s in symbols}
    n = len(ts)

    paper = MomentumPaper(cfg, mom)
    state_path = ROOT / args.state if args.state else None
    ledger_path = ROOT / args.ledger if args.ledger else None
    if state_path and state_path.exists():
        paper.load(json.loads(state_path.read_text()))

    if args.demo:
        warm = max(mom.trend_ma_bars, mom.breakout_bars, mom.momentum_bars) + 1
        start = max(warm, n - args.demo)
        print(f"# PAPER demo replay — {symbols}, bars {start}..{n-1} ({n-1-start} bars), config {args.config}")
        opened = closed = 0
        for i in range(start, n):
            for it in paper.step(symbols, _hist_upto(rows, symbols, i, tail=300)):
                tag = "OPEN " if it.action == "OPEN" else "CLOSE"
                stop = f" stop {it.stop:.4g}" if it.stop else ""
                print(f"  {ts[i][:16]} {tag} {it.side:4} {it.symbol:8} ${it.notional:8.2f} @ {it.price:.4g}"
                      f"{stop}  [{it.reason}]")
                opened += it.action == "OPEN"; closed += it.action == "CLOSE"
        marks = {s: rows[s][n - 1].close for s in symbols}
        eq = paper._marked(marks)
        print(f"\n# paper book now: equity ${eq:.2f} ({(eq/cfg.portfolio.initial_equity-1):+.1%}), "
              f"open {len(paper.positions)}, opened {opened}, closed {closed}, halted={paper.halted}")
        for s, p in paper.positions.items():
            upnl = p.side * p.notional * ((marks[s] / p.entry_price) - 1.0)
            print(f"    holding {s}: ${p.notional:.2f} @ {p.entry_price:.4g}, stop {p.stop_price:.4g}, "
                  f"uPnL ${upnl:+.2f}")
        return 0

    if args.tick:
        intents = paper.step(symbols, _hist_upto(rows, symbols, n - 1, tail=300))
        print(f"# PAPER tick {ts[-1][:16]} — {len(intents)} intent(s):")
        for it in intents:
            _print_intent(ts[-1], it)
        if not intents:
            print("  (no action — hold)")
        if ledger_path and intents:
            marks = {s: rows[s][n - 1].close for s in symbols}
            append_intent_ledger(ledger_path, ts[-1], intents, equity_after=paper._marked(marks))
        if state_path:
            state_path.write_text(json.dumps(paper.to_dict(), indent=2))
            print(f"# state saved -> {args.state}")
        return 0

    if args.catch_up:
        warm = max(mom.trend_ma_bars, mom.breakout_bars, mom.momentum_bars) + 1
        start = _catch_up_start_index(ts, paper.last_timestamp, warm)
        all_intents: list[Intent] = []
        processed = 0
        first_ts = ts[start] if start < n else None
        for i in range(start, n):
            processed += 1
            intents = paper.step(symbols, _hist_upto(rows, symbols, i, tail=300))
            for it in intents:
                all_intents.append(it)
                _print_intent(ts[i], it)
            if ledger_path and intents:
                marks = {s: rows[s][i].close for s in symbols}
                append_intent_ledger(ledger_path, ts[i], intents, equity_after=paper._marked(marks))
        if first_ts is None:
            print(f"# PAPER catch-up — 0 bar(s), 0 intent(s); already current at {paper.last_timestamp}")
        else:
            print(f"# PAPER catch-up {first_ts[:16]}..{ts[-1][:16]} — "
                  f"{processed} bar(s), {len(all_intents)} intent(s)")
        if not all_intents:
            print("  (no action — hold)")
        if state_path:
            state_path.write_text(json.dumps(paper.to_dict(), indent=2))
            print(f"# state saved -> {args.state}")
        return 0

    ap.error("pass --demo N, --tick, or --catch-up")


if __name__ == "__main__":
    raise SystemExit(main())
