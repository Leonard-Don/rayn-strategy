"""Portfolio risk controls."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RaynConfig
from .indicators import pct_change


@dataclass
class Position:
    symbol: str
    entry_time: str
    entry_price: float
    notional: float
    margin: float
    bars_held: int = 0
    entry_reason: str = ""


@dataclass
class RiskState:
    peak_equity: float
    current_day: str | None = None
    day_start_equity: float = 0.0
    pause_remaining_bars: int = 0
    consecutive_losses: int = 0
    loss_cooldown_remaining_bars: int = 0
    halted: bool = False


class RiskManager:
    def __init__(self, config: RaynConfig) -> None:
        self.config = config

    def initial_state(self) -> RiskState:
        equity = self.config.portfolio.initial_equity
        return RiskState(peak_equity=equity, day_start_equity=equity)

    def mark_day(self, state: RiskState, timestamp: str, equity: float) -> None:
        day = timestamp[:10]
        if state.current_day != day:
            state.current_day = day
            state.day_start_equity = equity

    def update_account_limits(self, state: RiskState, equity: float) -> None:
        state.peak_equity = max(state.peak_equity, equity)
        drawdown = 1.0 - (equity / state.peak_equity)
        if drawdown >= self.config.risk.max_account_drawdown_pct:
            state.halted = True

        if state.day_start_equity > 0:
            day_loss = 1.0 - (equity / state.day_start_equity)
            if day_loss >= self.config.risk.max_daily_loss_pct:
                state.pause_remaining_bars = max(
                    state.pause_remaining_bars,
                    self.config.circuit_breaker.pause_bars,
                )

    def update_circuit_breaker(
        self,
        state: RiskState,
        benchmark_closes: list[float],
    ) -> bool:
        if state.pause_remaining_bars > 0:
            state.pause_remaining_bars -= 1
            return True

        lookback = self.config.circuit_breaker.benchmark_drop_lookback_bars
        if len(benchmark_closes) <= lookback:
            return False

        old = benchmark_closes[-(lookback + 1)]
        new = benchmark_closes[-1]
        drop = -pct_change(old, new)
        if drop >= self.config.circuit_breaker.benchmark_drop_pct:
            state.pause_remaining_bars = self.config.circuit_breaker.pause_bars
            return True
        return False

    def can_open(
        self,
        symbol: str,
        equity: float,
        positions: dict[str, Position],
    ) -> tuple[bool, str]:
        if equity <= 0:
            return False, "no_equity"
        if len(positions) >= self.config.risk.max_open_positions:
            return False, "max_open_positions"
        if symbol in positions and not self.config.risk.allow_scale_in:
            return False, "scale_in_disabled"

        total_notional = sum(position.notional for position in positions.values())
        total_margin = sum(position.margin for position in positions.values())
        if total_notional / equity >= self.config.risk.max_total_notional_pct:
            return False, "max_total_notional"
        if total_margin / equity >= self.config.risk.max_used_margin_pct:
            return False, "max_used_margin"
        return True, "ok"

    def record_trade_result(self, state: RiskState, pnl: float) -> None:
        if pnl < 0:
            state.consecutive_losses += 1
            if (
                self.config.risk.max_consecutive_losses > 0
                and state.consecutive_losses >= self.config.risk.max_consecutive_losses
            ):
                state.loss_cooldown_remaining_bars = max(
                    state.loss_cooldown_remaining_bars,
                    self.config.risk.loss_cooldown_bars,
                )
            return

        if pnl > 0:
            state.consecutive_losses = 0

    def loss_cooldown_active(self, state: RiskState) -> bool:
        return state.loss_cooldown_remaining_bars > 0

    def advance_loss_cooldown(self, state: RiskState) -> None:
        if state.loss_cooldown_remaining_bars > 0:
            state.loss_cooldown_remaining_bars -= 1

    def position_notional(self, equity: float) -> float:
        stop = self.config.strategy.stop_loss_pct
        raw = equity * self.config.risk.risk_per_trade_pct / stop
        cap = equity * self.config.risk.max_position_notional_pct
        return max(0.0, min(raw, cap))

    def build_position(
        self,
        symbol: str,
        timestamp: str,
        price: float,
        equity: float,
        reason: str,
    ) -> Position:
        notional = self.position_notional(equity)
        margin = notional / self.config.risk.leverage
        return Position(
            symbol=symbol,
            entry_time=timestamp,
            entry_price=price,
            notional=notional,
            margin=margin,
            entry_reason=reason,
        )
