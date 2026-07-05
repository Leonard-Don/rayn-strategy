"""A trend/momentum strategy class — the opposite of the Rayn martingale.

Cut losses, let winners run: momentum entries, an initial ATR stop plus a ratcheting
trailing stop, risk-based (inverse-volatility) sizing, low leverage, no averaging down.
Long when the BTC regime is up, SHORT when it is down (symmetric trend-following), so it
can profit in bears instead of only sitting them out. Funding is not modeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backtest import (
    BacktestResult,
    Candle,
    EquityPoint,
    Trade,
    current_drawdown,
    load_candles,
    normalize_timestamp,
    trading_cost,
)
from .config import RaynConfig
from .indicators import pct_change, rolling_high, rsi, sma
from .market_filter import breadth_above_ma, long_breadth_allows_entry

MAINTENANCE_MARGIN_PCT = 0.005


def atr_pct(highs, lows, closes, length: int) -> float | None:
    """Average true range as a fraction of price, over the last `length` bars."""
    if len(closes) < length + 1:
        return None
    total = 0.0
    n = 0
    for i in range(len(closes) - length, len(closes)):
        if i <= 0 or closes[i] <= 0:
            continue
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        total += tr / closes[i]
        n += 1
    return total / n if n else None


def rolling_low(values, length: int) -> float | None:
    if length <= 0 or len(values) < length:
        return None
    return min(values[-length:])


@dataclass(frozen=True)
class MomentumConfig:
    trend_ma_bars: int = 200
    breakout_bars: int = 96
    momentum_bars: int = 168
    min_momentum_pct: float = 0.0
    rsi_bars: int = 14
    max_entry_rsi: float = 85.0
    atr_bars: int = 48
    stop_atr_mult: float = 3.0
    trail_atr_mult: float = 4.0
    risk_per_trade_pct: float = 0.0075
    max_hold_bars: int = 4320
    regime_ma_bars: int = 200
    regime_band_pct: float = 0.0   # neutral band around the regime MA (reduces whipsaw / squeeze)
    long_breadth_ma_bars: int = 0
    min_long_breadth_pct: float = 0.0
    short_entry_mode: str = "breakdown"  # breakdown / bounce_failure / either
    short_bounce_lookback_bars: int = 24
    short_bounce_min_rebound_pct: float = 0.025
    short_bounce_failure_ma_bars: int = 24
    short_bounce_max_extension_below_ma_pct: float = 0.08
    short_rebound_filter_bars: int = 0
    short_rebound_filter_min_pct: float = 0.0
    short_rebound_filter_min_count: int = 0
    short_rebound_filter_benchmark_min_pct: float = -0.005
    short_rebound_filter_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    entry_quality_min_score: float = 0.0
    entry_quality_full_score: float = 1.0
    entry_quality_min_risk_mult: float = 1.0
    entry_quality_max_risk_mult: float = 1.0
    entry_quality_momentum_weight: float = 0.45
    entry_quality_trend_weight: float = 0.35
    entry_quality_trigger_weight: float = 0.20
    entry_quality_momentum_cap_pct: float = 0.08
    entry_quality_trend_cap_pct: float = 0.08
    entry_quality_breakout_cap_pct: float = 0.04
    allow_long: bool = True
    allow_short: bool = True


@dataclass(frozen=True)
class EntryQuality:
    score: float
    momentum_score: float
    trend_score: float
    trigger_score: float
    risk_multiplier: float


@dataclass
class MomPosition:
    symbol: str
    entry_time: str
    entry_price: float
    notional: float
    side: int                # +1 long, -1 short
    stop_price: float
    water: float             # highest close (long) / lowest close (short) since entry
    bars_held: int = 0


def entry_quality_risk_multiplier(score: float, mom: MomentumConfig) -> float:
    min_score = max(0.0, mom.entry_quality_min_score)
    full_score = max(min_score, mom.entry_quality_full_score)
    min_mult = max(0.0, mom.entry_quality_min_risk_mult)
    max_mult = max(0.0, mom.entry_quality_max_risk_mult)

    if score < min_score:
        return 0.0
    if full_score <= min_score:
        return max_mult
    progress = _clamp01((score - min_score) / (full_score - min_score))
    return min_mult + progress * (max_mult - min_mult)


class MomentumEngine:
    def __init__(
        self,
        config: RaynConfig,
        mom: MomentumConfig,
        data_dir: Path,
        symbols: list[str],
        entry_start: str | None = None,
        entry_end: str | None = None,
        run_end: str | None = None,
        preloaded_data: dict[str, list[Candle]] | None = None,
    ) -> None:
        self.config = config
        self.mom = mom
        self.data_dir = data_dir
        self.symbols = symbols
        self.entry_start = normalize_timestamp(entry_start) if entry_start else None
        self.entry_end = normalize_timestamp(entry_end) if entry_end else None
        self.run_end = normalize_timestamp(run_end) if run_end else None
        self.preloaded_data = preloaded_data

    def run(self) -> BacktestResult:
        data = self.preloaded_data or {s: load_candles(self.data_dir / f"{s}.csv") for s in self.symbols}
        benchmark = self.config.circuit_breaker.benchmark_symbol
        if benchmark not in data:
            raise ValueError(f"benchmark {benchmark} must be in symbols")
        timestamps = sorted(set.intersection(*(set(r.timestamp for r in rows) for rows in data.values())))
        by_time = {s: {r.timestamp: r for r in rows} for s, rows in data.items()}
        closes = {s: [] for s in self.symbols}
        highs = {s: [] for s in self.symbols}
        lows = {s: [] for s in self.symbols}
        positions: dict[str, MomPosition] = {}
        trades: list[Trade] = []
        curve: list[EquityPoint] = []
        cash = self.config.portfolio.initial_equity
        peak = cash
        max_dd = 0.0
        leverage = self.config.risk.leverage
        last_ts = timestamps[0]

        for ts in timestamps:
            if self.run_end and ts >= self.run_end:
                break
            last_ts = ts
            for s in self.symbols:
                c = by_time[s][ts]
                closes[s].append(c.close)
                highs[s].append(c.high)
                lows[s].append(c.low)
            marks = {s: by_time[s][ts].close for s in self.symbols}

            # manage open positions (exits)
            for s, p in list(positions.items()):
                p.bars_held += 1
                c = by_time[s][ts]
                liq = self._liq(p.entry_price, p.side, leverage)
                reason = exit_price = None
                if p.side > 0:
                    if c.low <= liq and liq >= p.stop_price:
                        reason, exit_price = "liquidation", liq
                    elif c.low <= p.stop_price:
                        reason, exit_price = "trail_stop", min(p.stop_price, c.open)
                else:
                    if c.high >= liq and liq <= p.stop_price:
                        reason, exit_price = "liquidation", liq
                    elif c.high >= p.stop_price:
                        reason, exit_price = "trail_stop", max(p.stop_price, c.open)
                if reason is None and p.bars_held >= self.mom.max_hold_bars:
                    reason, exit_price = "time_stop", c.close
                if reason:
                    trade, cash = self._close(p, ts, exit_price, reason, cash)
                    trades.append(trade)
                    del positions[s]
                else:
                    a = atr_pct(highs[s], lows[s], closes[s], self.mom.atr_bars)
                    if p.side > 0:
                        p.water = max(p.water, c.close)
                        if a is not None:
                            p.stop_price = max(p.stop_price, p.water * (1.0 - self.mom.trail_atr_mult * a))
                    else:
                        p.water = min(p.water, c.close)
                        if a is not None:
                            p.stop_price = min(p.stop_price, p.water * (1.0 + self.mom.trail_atr_mult * a))

            marked = self._marked(cash, positions, marks)
            peak = max(peak, marked)
            max_dd = max(max_dd, current_drawdown(marked, peak))

            # entries
            if self._window_open(ts):
                regime = self._regime(closes[benchmark])
                long_breadth_ok = self._long_breadth_ok(closes)
                short_rebound_blocked = self._short_rebound_filter_blocks(closes)
                for s in self.symbols:
                    if s in positions or len(positions) >= self.config.risk.max_open_positions:
                        continue
                    a = atr_pct(highs[s], lows[s], closes[s], self.mom.atr_bars)
                    if a is None or a <= 0:
                        continue
                    side = 0
                    if regime > 0 and self.mom.allow_long and long_breadth_ok and self._entry_long(closes[s]):
                        side = 1
                    elif (
                        regime < 0
                        and self.mom.allow_short
                        and not short_rebound_blocked
                        and self._entry_short(closes[s])
                    ):
                        side = -1
                    if side == 0:
                        continue
                    quality = self._entry_quality(side, closes[s])
                    if quality.risk_multiplier <= 0.0:
                        continue
                    price = marks[s]
                    stop_dist = self.mom.stop_atr_mult * a
                    notional = (marked * self.mom.risk_per_trade_pct * quality.risk_multiplier) / stop_dist
                    notional = min(notional, marked * self.config.risk.max_position_notional_pct)
                    total = sum(q.notional for q in positions.values())
                    cap = marked * self.config.risk.max_total_notional_pct
                    if total + notional > cap:
                        notional = max(0.0, cap - total)
                    if notional <= 0:
                        continue
                    cash -= trading_cost(notional, self.config)
                    stop = price * (1.0 - stop_dist) if side > 0 else price * (1.0 + stop_dist)
                    positions[s] = MomPosition(s, ts, price, notional, side, stop, price)

            marked = self._marked(cash, positions, marks)
            peak = max(peak, marked)
            max_dd = max(max_dd, current_drawdown(marked, peak))
            curve.append(EquityPoint(ts, cash, marked, sum(p.notional for p in positions.values()),
                                     sum(p.notional / leverage for p in positions.values())))

        finals = {s: by_time[s][last_ts].close for s in self.symbols}
        for s, p in list(positions.items()):
            trade, cash = self._close(p, last_ts, finals[s], "end_of_test", cash)
            trades.append(trade)
            del positions[s]
        peak = max(peak, cash)
        max_dd = max(max_dd, current_drawdown(cash, peak))
        return BacktestResult(self.config.portfolio.initial_equity, cash, peak, max_dd,
                              trades=trades, equity_curve=curve)

    def _liq(self, entry: float, side: int, leverage: float) -> float:
        if side > 0:
            return entry * (1.0 - 1.0 / leverage + MAINTENANCE_MARGIN_PCT)
        return entry * (1.0 + 1.0 / leverage - MAINTENANCE_MARGIN_PCT)

    def _entry_long(self, closes: list[float]) -> bool:
        need = max(self.mom.trend_ma_bars, self.mom.breakout_bars, self.mom.momentum_bars) + 1
        if len(closes) < need:
            return False
        ma = sma(closes, self.mom.trend_ma_bars)
        hh = rolling_high(closes[:-1], self.mom.breakout_bars)
        r = rsi(closes, self.mom.rsi_bars)
        if ma is None or hh is None or r is None:
            return False
        mom = pct_change(closes[-(self.mom.momentum_bars + 1)], closes[-1])
        return (closes[-1] > ma and closes[-1] > hh and mom >= self.mom.min_momentum_pct
                and r <= self.mom.max_entry_rsi)

    def _entry_short(self, closes: list[float]) -> bool:
        if self.mom.short_entry_mode == "bounce_failure":
            return self._entry_short_bounce_failure(closes)
        if self.mom.short_entry_mode == "either":
            return self._entry_short_breakdown(closes) or self._entry_short_bounce_failure(closes)
        return self._entry_short_breakdown(closes)

    def _entry_short_breakdown(self, closes: list[float]) -> bool:
        need = max(self.mom.trend_ma_bars, self.mom.breakout_bars, self.mom.momentum_bars) + 1
        if len(closes) < need:
            return False
        ma = sma(closes, self.mom.trend_ma_bars)
        ll = rolling_low(closes[:-1], self.mom.breakout_bars)
        r = rsi(closes, self.mom.rsi_bars)
        if ma is None or ll is None or r is None:
            return False
        mom = pct_change(closes[-(self.mom.momentum_bars + 1)], closes[-1])
        # mirror of long: downtrend, fresh breakdown, downward momentum, not already washed out
        return (closes[-1] < ma and closes[-1] < ll and mom <= -self.mom.min_momentum_pct
                and r >= (100.0 - self.mom.max_entry_rsi))

    def _entry_short_bounce_failure(self, closes: list[float]) -> bool:
        need = max(
            self.mom.trend_ma_bars,
            self.mom.short_bounce_failure_ma_bars,
            self.mom.short_bounce_lookback_bars + 1,
            self.mom.rsi_bars + 1,
        )
        if len(closes) < need:
            return False
        trend_ma = sma(closes, self.mom.trend_ma_bars)
        fail_ma = sma(closes, self.mom.short_bounce_failure_ma_bars)
        r = rsi(closes, self.mom.rsi_bars)
        if trend_ma is None or fail_ma is None or fail_ma <= 0 or r is None:
            return False
        recent = closes[-(self.mom.short_bounce_lookback_bars + 1):-1]
        recent_low = min(recent)
        recent_high = max(recent)
        if recent_low <= 0:
            return False
        rebound = recent_high / recent_low - 1.0
        extension = (fail_ma - closes[-1]) / fail_ma
        return (
            closes[-1] < trend_ma
            and closes[-1] < fail_ma
            and closes[-1] < closes[-2]
            and rebound >= self.mom.short_bounce_min_rebound_pct
            and extension <= self.mom.short_bounce_max_extension_below_ma_pct
            and r >= (100.0 - self.mom.max_entry_rsi)
        )

    def _regime(self, bench_closes: list[float]) -> int:
        if self.mom.regime_ma_bars <= 0:
            return 1
        ma = sma(bench_closes, self.mom.regime_ma_bars)
        if ma is None:
            return 0
        band = self.mom.regime_band_pct
        if bench_closes[-1] > ma * (1.0 + band):
            return 1
        if bench_closes[-1] < ma * (1.0 - band):
            return -1
        return 0

    def _long_breadth_ok(self, closes_by_symbol: dict[str, list[float]]) -> bool:
        if self.mom.min_long_breadth_pct <= 0.0:
            return True
        ma_bars = self.mom.long_breadth_ma_bars or self.mom.trend_ma_bars
        breadth = breadth_above_ma(closes_by_symbol, ma_bars)
        return long_breadth_allows_entry(breadth, self.mom.min_long_breadth_pct)

    def _short_rebound_filter_blocks(self, closes_by_symbol: dict[str, list[float]]) -> bool:
        bars = self.mom.short_rebound_filter_bars
        if bars <= 0 or self.mom.short_rebound_filter_min_pct <= 0.0:
            return False
        symbols = tuple(self.mom.short_rebound_filter_symbols)
        min_count = self.mom.short_rebound_filter_min_count or len(symbols)
        rebounding = 0

        for symbol in symbols:
            closes = closes_by_symbol.get(symbol, [])
            if len(closes) <= bars or closes[-(bars + 1)] <= 0:
                continue
            change = closes[-1] / closes[-(bars + 1)] - 1.0
            if change >= self.mom.short_rebound_filter_min_pct:
                rebounding += 1

        bench = closes_by_symbol.get(self.config.circuit_breaker.benchmark_symbol, [])
        if len(bench) > bars and bench[-(bars + 1)] > 0:
            bench_change = bench[-1] / bench[-(bars + 1)] - 1.0
            if bench_change < self.mom.short_rebound_filter_benchmark_min_pct:
                return False

        return rebounding >= min_count

    def _entry_quality(self, side: int, closes: list[float]) -> EntryQuality:
        momentum_score = self._entry_momentum_quality(side, closes)
        trend_score = self._entry_trend_quality(side, closes)
        trigger_score = self._entry_trigger_quality(side, closes)
        total_weight = (
            max(0.0, self.mom.entry_quality_momentum_weight)
            + max(0.0, self.mom.entry_quality_trend_weight)
            + max(0.0, self.mom.entry_quality_trigger_weight)
        )
        if total_weight <= 0.0:
            score = 1.0
        else:
            score = (
                max(0.0, self.mom.entry_quality_momentum_weight) * momentum_score
                + max(0.0, self.mom.entry_quality_trend_weight) * trend_score
                + max(0.0, self.mom.entry_quality_trigger_weight) * trigger_score
            ) / total_weight
        score = _clamp01(score)
        return EntryQuality(
            score=score,
            momentum_score=momentum_score,
            trend_score=trend_score,
            trigger_score=trigger_score,
            risk_multiplier=entry_quality_risk_multiplier(score, self.mom),
        )

    def _entry_momentum_quality(self, side: int, closes: list[float]) -> float:
        if self.mom.momentum_bars <= 0 or len(closes) <= self.mom.momentum_bars:
            return 0.0
        momentum = side * pct_change(closes[-(self.mom.momentum_bars + 1)], closes[-1])
        edge = momentum - max(0.0, self.mom.min_momentum_pct)
        return _score_edge(edge, self.mom.entry_quality_momentum_cap_pct)

    def _entry_trend_quality(self, side: int, closes: list[float]) -> float:
        ma = sma(closes, self.mom.trend_ma_bars)
        if ma is None or ma <= 0.0 or not closes:
            return 0.0
        edge = side * (closes[-1] / ma - 1.0)
        return _score_edge(edge, self.mom.entry_quality_trend_cap_pct)

    def _entry_trigger_quality(self, side: int, closes: list[float]) -> float:
        if side > 0:
            high = rolling_high(closes[:-1], self.mom.breakout_bars)
            if high is None or high <= 0.0:
                return 0.0
            return _score_edge(closes[-1] / high - 1.0, self.mom.entry_quality_breakout_cap_pct)

        if self.mom.short_entry_mode == "bounce_failure":
            recent = closes[-(self.mom.short_bounce_lookback_bars + 1):-1]
            if not recent:
                return 0.0
            recent_low = min(recent)
            recent_high = max(recent)
            if recent_low <= 0.0:
                return 0.0
            rebound = recent_high / recent_low - 1.0
            rebound_score = _score_edge(
                rebound - self.mom.short_bounce_min_rebound_pct,
                self.mom.entry_quality_breakout_cap_pct,
            )
            fail_ma = sma(closes, self.mom.short_bounce_failure_ma_bars)
            if fail_ma is None or fail_ma <= 0.0:
                return rebound_score
            extension = (fail_ma - closes[-1]) / fail_ma
            extension_room = self.mom.short_bounce_max_extension_below_ma_pct - extension
            extension_score = _score_edge(extension_room, self.mom.entry_quality_breakout_cap_pct)
            return (rebound_score + extension_score) / 2.0

        low = rolling_low(closes[:-1], self.mom.breakout_bars)
        if low is None or closes[-1] <= 0.0:
            return 0.0
        return _score_edge(low / closes[-1] - 1.0, self.mom.entry_quality_breakout_cap_pct)

    def _window_open(self, ts: str) -> bool:
        if self.entry_start and ts < self.entry_start:
            return False
        if self.entry_end and ts >= self.entry_end:
            return False
        return True

    def _marked(self, cash, positions, prices) -> float:
        u = sum(p.side * p.notional * ((prices[s] / p.entry_price) - 1.0) for s, p in positions.items())
        return cash + u

    def _close(self, p: MomPosition, ts: str, price: float, reason: str, cash: float):
        gross = p.side * p.notional * ((price / p.entry_price) - 1.0)
        one_side = trading_cost(p.notional, self.config)
        pnl = gross - 2 * one_side
        cash = cash + gross - one_side
        label = "momentum_long" if p.side > 0 else "momentum_short"
        return Trade(p.symbol, p.entry_time, ts, p.entry_price, price, p.notional, pnl, label, reason), cash


def _score_edge(edge: float, cap: float) -> float:
    if cap <= 0.0:
        return 0.0
    return _clamp01(edge / cap)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
