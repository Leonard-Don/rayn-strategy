#!/usr/bin/env python3
"""Backtest the current long momentum module plus the short-regime observer.

Research/backtest only. This tool never places orders.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.backtest import (
    BacktestResult,
    EquityPoint,
    Trade,
    current_drawdown,
    load_candles,
    normalize_timestamp,
    trading_cost,
)
from src.rayn_strategy.config import RaynConfig, load_config, load_raw_config
from src.rayn_strategy.market_filter import ShortCandidateFilter, short_candidate_allows_entry
from src.rayn_strategy.momentum import MomentumConfig, MomentumEngine, MomPosition, atr_pct
from src.rayn_strategy.indicators import sma


DEFAULT_PERIODS = [
    ("2026 YTD", "2026-01-01", "2026-06-06"),
    ("2026 MarNow", "2026-03-01", "2026-06-06"),
    ("2026 May14Now", "2026-05-14", "2026-06-06"),
]


@dataclass
class ActivePosition:
    position: MomPosition
    config: RaynConfig
    mom: MomentumConfig
    entry_reason: str


@dataclass(frozen=True)
class DynamicShortRiskConfig:
    enabled: bool = False
    max_multiplier: float = 1.0
    min_below_ma_pct: float = 0.0
    max_drawdown_pct: float = 1.0
    ma_bars: int = 200


def short_risk_multiplier(
    benchmark_closes: list[float],
    *,
    peak: float,
    marked: float,
    config: DynamicShortRiskConfig,
) -> float:
    if not config.enabled or config.max_multiplier <= 1.0:
        return 1.0
    ma = sma(benchmark_closes, config.ma_bars)
    if ma is None or ma <= 0 or not benchmark_closes:
        return 1.0
    below_ma = (ma - benchmark_closes[-1]) / ma
    drawdown = (peak - marked) / peak if peak > 0 else 0.0
    if below_ma >= config.min_below_ma_pct and drawdown <= config.max_drawdown_pct:
        return config.max_multiplier
    return 1.0


class CombinedMomentumEngine:
    """Single-account, one-direction-at-a-time blend of long and short modules."""

    def __init__(
        self,
        *,
        long_config: RaynConfig,
        long_mom: MomentumConfig,
        short_config: RaynConfig,
        short_mom: MomentumConfig,
        data_dir: Path,
        symbols: list[str],
        entry_start: str | None = None,
        run_end: str | None = None,
        preloaded_data=None,
        short_candidate_filter: ShortCandidateFilter | None = None,
        dynamic_short_risk: DynamicShortRiskConfig | None = None,
    ) -> None:
        self.long_config = long_config
        self.long_mom = long_mom
        self.short_config = short_config
        self.short_mom = short_mom
        self.data_dir = data_dir
        self.symbols = symbols
        self.entry_start = normalize_timestamp(entry_start) if entry_start else None
        self.run_end = normalize_timestamp(run_end) if run_end else None
        self.preloaded_data = preloaded_data
        self.short_candidate_filter = short_candidate_filter
        self.dynamic_short_risk = dynamic_short_risk or DynamicShortRiskConfig()
        self.long_helper = MomentumEngine(
            config=long_config, mom=long_mom, data_dir=data_dir, symbols=symbols
        )
        self.short_helper = MomentumEngine(
            config=short_config, mom=short_mom, data_dir=data_dir, symbols=symbols
        )

    def run(self) -> BacktestResult:
        data = self.preloaded_data or {s: load_candles(self.data_dir / f"{s}.csv") for s in self.symbols}
        benchmark = self.long_config.circuit_breaker.benchmark_symbol
        if benchmark not in data:
            raise ValueError(f"benchmark {benchmark} must be in symbols")
        timestamps = sorted(set.intersection(*(set(r.timestamp for r in rows) for rows in data.values())))
        by_time = {s: {r.timestamp: r for r in rows} for s, rows in data.items()}
        closes = {s: [] for s in self.symbols}
        highs = {s: [] for s in self.symbols}
        lows = {s: [] for s in self.symbols}
        candles = {s: [] for s in self.symbols}
        positions: dict[str, ActivePosition] = {}
        trades: list[Trade] = []
        curve: list[EquityPoint] = []
        cash = self.long_config.portfolio.initial_equity
        peak = cash
        max_dd = 0.0
        last_ts = timestamps[0]

        for ts in timestamps:
            if self.run_end and ts >= self.run_end:
                break
            last_ts = ts
            for symbol in self.symbols:
                candle = by_time[symbol][ts]
                closes[symbol].append(candle.close)
                highs[symbol].append(candle.high)
                lows[symbol].append(candle.low)
                candles[symbol].append(candle)
            marks = {s: by_time[s][ts].close for s in self.symbols}

            for symbol, active in list(positions.items()):
                pos = active.position
                pos.bars_held += 1
                candle = by_time[symbol][ts]
                leverage = active.config.risk.leverage
                helper = self.long_helper if pos.side > 0 else self.short_helper
                liq = helper._liq(pos.entry_price, pos.side, leverage)
                reason = exit_price = None
                if pos.side > 0:
                    if candle.low <= liq and liq >= pos.stop_price:
                        reason, exit_price = "liquidation", liq
                    elif candle.low <= pos.stop_price:
                        reason, exit_price = "trail_stop", min(pos.stop_price, candle.open)
                else:
                    if candle.high >= liq and liq <= pos.stop_price:
                        reason, exit_price = "liquidation", liq
                    elif candle.high >= pos.stop_price:
                        reason, exit_price = "trail_stop", max(pos.stop_price, candle.open)
                if reason is None and pos.bars_held >= active.mom.max_hold_bars:
                    reason, exit_price = "time_stop", candle.close
                if reason:
                    trade, cash = self._close(active, ts, exit_price, reason, cash)
                    trades.append(trade)
                    del positions[symbol]
                else:
                    atr = atr_pct(highs[symbol], lows[symbol], closes[symbol], active.mom.atr_bars)
                    if pos.side > 0:
                        pos.water = max(pos.water, candle.close)
                        if atr is not None:
                            pos.stop_price = max(pos.stop_price, pos.water * (1.0 - active.mom.trail_atr_mult * atr))
                    else:
                        pos.water = min(pos.water, candle.close)
                        if atr is not None:
                            pos.stop_price = min(pos.stop_price, pos.water * (1.0 + active.mom.trail_atr_mult * atr))

            marked = self._marked(cash, positions, marks)
            peak = max(peak, marked)
            max_dd = max(max_dd, current_drawdown(marked, peak))

            if self._window_open(ts):
                open_side = next((p.position.side for p in positions.values()), 0)
                if open_side >= 0:
                    cash = self._try_entries(
                        ts=ts,
                        side=1,
                        helper=self.long_helper,
                        config=self.long_config,
                        mom=self.long_mom,
                        label="combined_long",
                        closes=closes,
                        highs=highs,
                        lows=lows,
                        candles=candles,
                        marks=marks,
                        positions=positions,
                        cash=cash,
                    )
                if open_side <= 0 and not any(p.position.side > 0 for p in positions.values()):
                    short_multiplier = short_risk_multiplier(
                        closes[benchmark],
                        peak=peak,
                        marked=marked,
                        config=self.dynamic_short_risk,
                    )
                    cash = self._try_entries(
                        ts=ts,
                        side=-1,
                        helper=self.short_helper,
                        config=self.short_config,
                        mom=self.short_mom,
                        label="combined_short",
                        closes=closes,
                        highs=highs,
                        lows=lows,
                        candles=candles,
                        marks=marks,
                        positions=positions,
                        cash=cash,
                        risk_multiplier=short_multiplier,
                    )

            marked = self._marked(cash, positions, marks)
            peak = max(peak, marked)
            max_dd = max(max_dd, current_drawdown(marked, peak))
            curve.append(EquityPoint(ts, cash, marked, sum(p.position.notional for p in positions.values()),
                                     sum(p.position.notional / p.config.risk.leverage for p in positions.values())))

        finals = {s: by_time[s][last_ts].close for s in self.symbols}
        for symbol, active in list(positions.items()):
            trade, cash = self._close(active, last_ts, finals[symbol], "end_of_test", cash)
            trades.append(trade)
            del positions[symbol]
        peak = max(peak, cash)
        max_dd = max(max_dd, current_drawdown(cash, peak))
        return BacktestResult(self.long_config.portfolio.initial_equity, cash, peak, max_dd, trades=trades,
                              equity_curve=curve)

    def _try_entries(
        self,
        *,
        ts: str,
        side: int,
        helper: MomentumEngine,
        config: RaynConfig,
        mom: MomentumConfig,
        label: str,
        closes: dict[str, list[float]],
        highs: dict[str, list[float]],
        lows: dict[str, list[float]],
        candles: dict[str, list[object]],
        marks: dict[str, float],
        positions: dict[str, ActivePosition],
        cash: float,
        risk_multiplier: float = 1.0,
    ) -> float:
        benchmark = config.circuit_breaker.benchmark_symbol
        regime = helper._regime(closes[benchmark])
        if side > 0 and (regime <= 0 or not mom.allow_long or not helper._long_breadth_ok(closes)):
            return cash
        if side < 0 and (regime >= 0 or not mom.allow_short or helper._short_rebound_filter_blocks(closes)):
            return cash

        marked = self._marked(cash, positions, marks)
        side_positions = [p for p in positions.values() if p.position.side == side]
        for symbol in self.symbols:
            if symbol in positions or len(side_positions) >= config.risk.max_open_positions:
                continue
            atr = atr_pct(highs[symbol], lows[symbol], closes[symbol], mom.atr_bars)
            if atr is None or atr <= 0:
                continue
            if side > 0:
                should_enter = helper._entry_long(closes[symbol])
            else:
                should_enter = helper._entry_short(closes[symbol])
            if not should_enter:
                continue
            if side < 0 and self.short_candidate_filter is not None:
                if not short_candidate_allows_entry(candles[symbol], self.short_candidate_filter):
                    continue
            quality = helper._entry_quality(side, closes[symbol])
            if quality.risk_multiplier <= 0.0:
                continue
            price = marks[symbol]
            stop_dist = mom.stop_atr_mult * atr
            notional = (
                marked
                * mom.risk_per_trade_pct
                * risk_multiplier
                * quality.risk_multiplier
            ) / stop_dist
            notional = min(notional, marked * config.risk.max_position_notional_pct)
            total = sum(p.position.notional for p in positions.values())
            cap = marked * config.risk.max_total_notional_pct
            if total + notional > cap:
                notional = max(0.0, cap - total)
            if notional <= 0:
                continue
            cash -= trading_cost(notional, config)
            stop = price * (1.0 - stop_dist) if side > 0 else price * (1.0 + stop_dist)
            pos = MomPosition(symbol, ts, price, notional, side, stop, price)
            positions[symbol] = ActivePosition(pos, config, mom, label)
            side_positions.append(positions[symbol])
        return cash

    def _window_open(self, ts: str) -> bool:
        if self.entry_start and ts < self.entry_start:
            return False
        return True

    def _marked(self, cash: float, positions: dict[str, ActivePosition], prices: dict[str, float]) -> float:
        unrealized = sum(
            active.position.side
            * active.position.notional
            * ((prices[symbol] / active.position.entry_price) - 1.0)
            for symbol, active in positions.items()
        )
        return cash + unrealized

    def _close(self, active: ActivePosition, ts: str, price: float, reason: str, cash: float):
        pos = active.position
        gross = pos.side * pos.notional * ((price / pos.entry_price) - 1.0)
        one_side = trading_cost(pos.notional, active.config)
        pnl = gross - 2 * one_side
        cash = cash + gross - one_side
        trade = Trade(pos.symbol, pos.entry_time, ts, pos.entry_price, price, pos.notional, pnl,
                      active.entry_reason, reason)
        return trade, cash

def _parse_symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _parse_periods(raw_periods: list[str] | None) -> list[tuple[str, str, str]]:
    if not raw_periods:
        return DEFAULT_PERIODS
    periods = []
    for raw in raw_periods:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"period must be name:start:end, got {raw!r}")
        periods.append((parts[0], parts[1], parts[2]))
    return periods


def _load_momentum(path: Path) -> MomentumConfig:
    return MomentumConfig(**load_raw_config(path)["momentum"])


def _load_account_config(path: Path, initial_equity: float | None) -> RaynConfig:
    cfg = load_config(path)
    if initial_equity is not None:
        cfg = replace(cfg, portfolio=replace(cfg.portfolio, initial_equity=initial_equity))
    return cfg


def _stats(result: BacktestResult) -> dict[str, float | int]:
    roi = result.final_equity / result.initial_equity - 1.0
    wins = [t for t in result.trades if t.pnl > 0]
    losses = [t for t in result.trades if t.pnl <= 0]
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
    return {
        "roi": roi,
        "max_dd": result.max_drawdown_pct,
        "trades": len(result.trades),
        "win_rate": len(wins) / len(result.trades) if result.trades else 0.0,
        "wl": avg_win / -avg_loss if avg_loss else 0.0,
        "long_trades": sum(1 for t in result.trades if t.entry_reason == "combined_long"),
        "short_trades": sum(1 for t in result.trades if t.entry_reason == "combined_short"),
        "final_equity": result.final_equity,
    }


def run_matrix(
    *,
    long_config_path: Path,
    short_config_path: Path,
    data_dir: Path,
    symbols: list[str],
    periods: list[tuple[str, str, str]],
    thresholds: list[float],
    breadth_ma_bars: int,
    initial_equity: float | None,
    short_candidate_filter: ShortCandidateFilter | None = None,
    dynamic_short_risk: DynamicShortRiskConfig | None = None,
) -> list[dict[str, str | float | int]]:
    long_config = _load_account_config(long_config_path, initial_equity)
    short_config = _load_account_config(short_config_path, initial_equity)
    base_long_mom = _load_momentum(long_config_path)
    short_mom = _load_momentum(short_config_path)
    data = {s: load_candles(data_dir / f"{s}.csv") for s in symbols}
    rows: list[dict[str, str | float | int]] = []

    for threshold in thresholds:
        long_mom = replace(
            base_long_mom,
            long_breadth_ma_bars=breadth_ma_bars,
            min_long_breadth_pct=threshold,
        )
        for label, start, end in periods:
            result = CombinedMomentumEngine(
                long_config=long_config,
                long_mom=long_mom,
                short_config=short_config,
                short_mom=short_mom,
                data_dir=data_dir,
                symbols=symbols,
                entry_start=start,
                run_end=end,
                preloaded_data=data,
                short_candidate_filter=short_candidate_filter,
                dynamic_short_risk=dynamic_short_risk,
            ).run()
            rows.append({"threshold": threshold, "period": label, **_stats(result)})
    return rows


def _print_rows(rows: list[dict[str, str | float | int]]) -> None:
    print(f"{'breadth':>8} {'period':16} {'ROI':>9} {'maxDD':>8} {'trades':>7} {'long':>6} {'short':>6} {'win%':>7} {'W/L':>6}")
    for row in rows:
        print(
            f"{row['threshold']:8.0%} {row['period']:16} "
            f"{row['roi']:>+9.1%} {row['max_dd']:>8.1%} {row['trades']:>7} "
            f"{row['long_trades']:>6} {row['short_trades']:>6} {row['win_rate']:>7.0%} {row['wl']:>6.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-config", default="config.rayn-momentum-paper.toml")
    parser.add_argument("--short-config", default="config.rayn-short-paper.toml")
    parser.add_argument("--data", default="data_multicycle")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    parser.add_argument("--period", action="append", help="name:start:end; can be repeated")
    parser.add_argument("--thresholds", default="0,0.35,0.5,0.65")
    parser.add_argument("--breadth-ma", type=int, default=200)
    parser.add_argument("--initial-equity", type=float, default=75.0)
    parser.add_argument("--dynamic-short-risk", action="store_true")
    parser.add_argument("--dynamic-short-max-mult", type=float, default=1.5)
    parser.add_argument("--dynamic-short-min-below-ma-pct", type=float, default=0.05)
    parser.add_argument("--dynamic-short-max-dd-pct", type=float, default=0.10)
    parser.add_argument("--dynamic-short-ma", type=int, default=200)
    parser.add_argument("--short-min-24h-quote-volume", type=float, default=0.0)
    parser.add_argument("--short-max-24h-range-pct", type=float, default=0.0)
    parser.add_argument("--short-max-below-ma-pct", type=float, default=0.0)
    parser.add_argument("--short-candidate-ma", type=int, default=200)
    parser.add_argument("--short-candidate-lookback", type=int, default=24)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    candidate_filter = None
    if (
        args.short_min_24h_quote_volume > 0.0
        or args.short_max_24h_range_pct > 0.0
        or args.short_max_below_ma_pct > 0.0
    ):
        candidate_filter = ShortCandidateFilter(
            min_24h_quote_volume=args.short_min_24h_quote_volume,
            max_24h_range_pct=args.short_max_24h_range_pct,
            max_below_ma_pct=args.short_max_below_ma_pct,
            ma_bars=args.short_candidate_ma,
            lookback_bars=args.short_candidate_lookback,
        )
    dynamic_short_risk = DynamicShortRiskConfig(
        enabled=args.dynamic_short_risk,
        max_multiplier=args.dynamic_short_max_mult,
        min_below_ma_pct=args.dynamic_short_min_below_ma_pct,
        max_drawdown_pct=args.dynamic_short_max_dd_pct,
        ma_bars=args.dynamic_short_ma,
    )

    rows = run_matrix(
        long_config_path=ROOT / args.long_config,
        short_config_path=ROOT / args.short_config,
        data_dir=Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data,
        symbols=_parse_symbols(args.symbols),
        periods=_parse_periods(args.period),
        thresholds=[float(x) for x in args.thresholds.split(",") if x.strip()],
        breadth_ma_bars=args.breadth_ma,
        initial_equity=args.initial_equity,
        short_candidate_filter=candidate_filter,
        dynamic_short_risk=dynamic_short_risk,
    )
    _print_rows(rows)
    if args.out:
        out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n# saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
