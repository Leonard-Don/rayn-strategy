"""Read-only signal checks for Rayn-style strategy research."""

from __future__ import annotations

from dataclasses import dataclass, field

from .backtest import Candle
from .config import RaynConfig
from .indicators import pct_change
from .signals import entry_signal, market_regime_filter_reason


@dataclass(frozen=True)
class SymbolDecision:
    symbol: str
    action: str
    reason: str
    price: float


@dataclass(frozen=True)
class SignalCheckResult:
    verdict: str
    primary_reason: str
    timestamp: str
    benchmark_symbol: str
    decisions: list[SymbolDecision] = field(default_factory=list)
    equity: float | None = None
    stop_equity: float | None = None

    @property
    def buy_symbols(self) -> list[str]:
        return [decision.symbol for decision in self.decisions if decision.action == "buy"]

    def to_text(self) -> str:
        lines = [
            "Rayn Signal Check",
            f"verdict:          {self.verdict}",
            f"primary_reason:   {self.primary_reason}",
            f"timestamp:        {self.timestamp}",
            f"benchmark:        {self.benchmark_symbol}",
        ]
        if self.equity is not None:
            lines.append(f"equity:           {self.equity:.2f}")
        if self.stop_equity is not None:
            lines.append(f"stage_stop:       {self.stop_equity:.2f}")
        if self.buy_symbols:
            lines.append(f"buy_symbols:      {','.join(self.buy_symbols)}")
        lines.append("symbol_decisions:")
        for decision in self.decisions:
            lines.append(
                f"  {decision.symbol}: {decision.action} "
                f"reason={decision.reason} price={decision.price:.10g}"
            )
        return "\n".join(lines)


def evaluate_signal_check(
    config: RaynConfig,
    candles_by_symbol: dict[str, list[Candle]],
    *,
    equity: float | None = None,
    stop_equity: float | None = None,
) -> SignalCheckResult:
    benchmark = config.circuit_breaker.benchmark_symbol
    if benchmark not in candles_by_symbol:
        raise ValueError(f"benchmark symbol {benchmark} must be included")
    if not candles_by_symbol:
        raise ValueError("candles_by_symbol must not be empty")

    timestamp = latest_common_timestamp(candles_by_symbol)
    aligned = {
        symbol: [candle for candle in candles if candle.timestamp <= timestamp]
        for symbol, candles in candles_by_symbol.items()
    }

    decisions = build_symbol_decisions(config, aligned)
    if equity is not None and stop_equity is not None and equity <= stop_equity:
        return SignalCheckResult(
            verdict="STOP",
            primary_reason="stage_stop",
            timestamp=timestamp,
            benchmark_symbol=benchmark,
            decisions=decisions,
            equity=equity,
            stop_equity=stop_equity,
        )

    benchmark_reason = benchmark_circuit_reason(
        [candle.close for candle in aligned[benchmark]],
        config,
    )
    if benchmark_reason:
        return SignalCheckResult(
            verdict="STOP",
            primary_reason=benchmark_reason,
            timestamp=timestamp,
            benchmark_symbol=benchmark,
            decisions=decisions,
            equity=equity,
            stop_equity=stop_equity,
        )

    market_reason = market_regime_filter_reason(
        [candle.close for candle in aligned[benchmark]],
        config.strategy,
    )
    if market_reason:
        return SignalCheckResult(
            verdict="WAIT",
            primary_reason=market_reason,
            timestamp=timestamp,
            benchmark_symbol=benchmark,
            decisions=decisions,
            equity=equity,
            stop_equity=stop_equity,
        )

    if any(decision.action == "buy" for decision in decisions):
        return SignalCheckResult(
            verdict="GO",
            primary_reason="strategy_entry",
            timestamp=timestamp,
            benchmark_symbol=benchmark,
            decisions=decisions,
            equity=equity,
            stop_equity=stop_equity,
        )

    return SignalCheckResult(
        verdict="WAIT",
        primary_reason="no_entry_signal",
        timestamp=timestamp,
        benchmark_symbol=benchmark,
        decisions=decisions,
        equity=equity,
        stop_equity=stop_equity,
    )


def latest_common_timestamp(candles_by_symbol: dict[str, list[Candle]]) -> str:
    latest = []
    for symbol, candles in candles_by_symbol.items():
        if not candles:
            raise ValueError(f"{symbol} has no candles")
        latest.append(candles[-1].timestamp)
    return min(latest)


def build_symbol_decisions(
    config: RaynConfig,
    candles_by_symbol: dict[str, list[Candle]],
) -> list[SymbolDecision]:
    decisions: list[SymbolDecision] = []
    for symbol, candles in candles_by_symbol.items():
        closes = [candle.close for candle in candles]
        highs = [candle.high for candle in candles]
        lows = [candle.low for candle in candles]
        volumes = [candle.volume for candle in candles]
        latest = candles[-1]
        signal = entry_signal(
            closes,
            config.strategy,
            highs=highs,
            lows=lows,
            volumes=volumes,
            timestamp=latest.timestamp,
        )
        decisions.append(
            SymbolDecision(
                symbol=symbol,
                action=signal.action,
                reason=signal.reason,
                price=latest.close,
            )
        )
    return decisions


def benchmark_circuit_reason(closes: list[float], config: RaynConfig) -> str:
    lookback = config.circuit_breaker.benchmark_drop_lookback_bars
    if len(closes) <= lookback:
        return ""
    old = closes[-(lookback + 1)]
    new = closes[-1]
    drop = -pct_change(old, new)
    if drop >= config.circuit_breaker.benchmark_drop_pct:
        return "benchmark_drop"
    return ""
