"""Entry and exit signal logic."""

from __future__ import annotations

from dataclasses import dataclass

from .config import StrategyConfig
from .indicators import average_range_pct, pct_change, rolling_high, rsi, sma, volume_spike


@dataclass(frozen=True)
class Signal:
    action: str
    reason: str


NO_SIGNAL = Signal(action="hold", reason="no_signal")


def entry_signal(
    closes: list[float],
    config: StrategyConfig,
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
    timestamp: str | None = None,
) -> Signal:
    if len(closes) < max(
        config.trend_ma_bars,
        config.breakout_lookback_bars,
        config.panic_lookback_bars,
        config.panic_rsi_bars + 1,
    ):
        return NO_SIGNAL

    current = closes[-1]
    previous = closes[-2]
    trend_ma = sma(closes, config.trend_ma_bars)
    recent_high = rolling_high(closes[:-1], config.breakout_lookback_bars)
    panic_high = rolling_high(closes[:-1], config.panic_lookback_bars)
    current_rsi = rsi(closes, config.panic_rsi_bars)

    if trend_ma is None or recent_high is None or panic_high is None or current_rsi is None:
        return NO_SIGNAL

    above_trend = current > trend_ma
    broke_out = current > recent_high * (1.0 + config.breakout_buffer_pct)
    candidate_reason = ""
    if above_trend and broke_out:
        if _volume_confirmed(volumes, config.breakout_volume_lookback, config.breakout_volume_multiplier):
            candidate_reason = "trend_breakout"
        else:
            return Signal(action="hold", reason="low_volume_breakout")
    else:
        drawdown = pct_change(panic_high, current)
        rebounded = current > previous * (1.0 + config.rebound_confirm_pct)
        panic_setup = drawdown <= -config.panic_drop_pct and current_rsi <= config.panic_rsi_max
        if panic_setup and rebounded:
            if _volume_confirmed(volumes, config.panic_volume_lookback, config.panic_volume_multiplier):
                candidate_reason = "panic_rebound"
            else:
                return Signal(action="hold", reason="low_volume_rebound")

    if not candidate_reason:
        return NO_SIGNAL

    blocked_reason = entry_filter_reason(
        closes=closes,
        highs=highs,
        lows=lows,
        timestamp=timestamp,
        config=config,
    )
    if blocked_reason:
        return Signal(action="hold", reason=blocked_reason)

    return Signal(action="buy", reason=candidate_reason)


def _volume_confirmed(volumes: list[float] | None, lookback: int, multiplier: float) -> bool:
    """Return True if volume confirmation passes or is disabled."""
    if not volumes or lookback <= 0 or multiplier <= 0:
        return True  # disabled — allow entry
    return volume_spike(volumes, lookback, multiplier)


def entry_filter_reason(
    *,
    closes: list[float],
    highs: list[float] | None,
    lows: list[float] | None,
    timestamp: str | None,
    config: StrategyConfig,
) -> str:
    if timestamp and entry_hour(timestamp) in config.blocked_entry_hours_utc:
        return "blocked_hour"

    if config.max_entry_range_pct > 0 and config.volatility_lookback_bars > 0:
        if highs is None or lows is None:
            return ""
        range_pct = average_range_pct(
            highs,
            lows,
            closes,
            config.volatility_lookback_bars,
        )
        if range_pct is not None and range_pct > config.max_entry_range_pct:
            return "volatility_filter"

    if config.undertrend_lookback_bars > 0:
        moving_average = sma(closes, config.undertrend_lookback_bars)
        if moving_average is not None and closes[-1] < moving_average * (1.0 + config.min_entry_ma_ratio):
            return "weakness_filter"

    if config.entry_momentum_lookback_bars > 0:
        lookback = config.entry_momentum_lookback_bars
        if len(closes) > lookback:
            momentum = pct_change(closes[-(lookback + 1)], closes[-1])
            if momentum < config.min_entry_momentum_pct:
                return "weakness_filter"

    if config.max_entry_rsi > 0:
        current_rsi = rsi(closes, config.panic_rsi_bars)
        if current_rsi is not None and current_rsi > config.max_entry_rsi:
            return "overbought_filter"

    if config.max_entry_ma_extension_pct > 0 and config.overextension_lookback_bars > 0:
        moving_average = sma(closes, config.overextension_lookback_bars)
        if moving_average is not None and closes[-1] > moving_average * (1.0 + config.max_entry_ma_extension_pct):
            return "overextension_filter"
    return ""


def market_regime_filter_reason(closes: list[float], config: StrategyConfig) -> str:
    if config.regime_filter_ma_bars > 0:
        moving_average = sma(closes, config.regime_filter_ma_bars)
        if moving_average is None:
            return ""
        if closes[-1] < moving_average * (1.0 + config.regime_filter_min_ma_ratio):
            return "market_regime_filter"

    if config.regime_filter_momentum_bars > 0:
        lookback = config.regime_filter_momentum_bars
        if len(closes) <= lookback:
            return ""
        momentum = pct_change(closes[-(lookback + 1)], closes[-1])
        if momentum < config.regime_filter_min_momentum_pct:
            return "market_regime_filter"

    return ""


def entry_hour(timestamp: str) -> int:
    try:
        return int(timestamp[11:13])
    except (ValueError, IndexError):
        return -1


def exit_signal(
    entry_price: float,
    current_price: float,
    high_price: float,
    low_price: float,
    bars_held: int,
    circuit_breaker_active: bool,
    config: StrategyConfig,
) -> Signal:
    move = pct_change(entry_price, current_price)
    if circuit_breaker_active:
        return Signal(action="sell", reason="circuit_breaker")
    if low_price <= entry_price * (1.0 - config.stop_loss_pct):
        return Signal(action="sell", reason="stop_loss")
    if high_price >= entry_price * (1.0 + config.take_profit_pct):
        return Signal(action="sell", reason="take_profit")
    if move >= config.take_profit_pct:
        return Signal(action="sell", reason="take_profit")
    if move <= -config.stop_loss_pct:
        return Signal(action="sell", reason="stop_loss")
    if bars_held >= config.max_hold_bars:
        return Signal(action="sell", reason="time_stop")
    return NO_SIGNAL
