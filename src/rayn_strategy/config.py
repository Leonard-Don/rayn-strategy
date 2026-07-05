"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from .paths import resolve_config_path


@dataclass(frozen=True)
class PortfolioConfig:
    initial_equity: float
    quote_asset: str
    taker_fee_bps: float
    slippage_bps: float


@dataclass(frozen=True)
class StrategyConfig:
    timeframe: str
    trend_ma_bars: int
    breakout_lookback_bars: int
    breakout_buffer_pct: float
    panic_lookback_bars: int
    panic_drop_pct: float
    panic_rsi_bars: int
    panic_rsi_max: float
    rebound_confirm_pct: float
    take_profit_pct: float
    stop_loss_pct: float
    max_hold_bars: int
    volatility_lookback_bars: int = 0
    max_entry_range_pct: float = 0.0
    blocked_entry_hours_utc: list[int] = field(default_factory=list)
    regime_filter_ma_bars: int = 0
    regime_filter_min_ma_ratio: float = 0.0
    regime_filter_momentum_bars: int = 0
    regime_filter_min_momentum_pct: float = 0.0
    max_entry_rsi: float = 0.0
    overextension_lookback_bars: int = 0
    max_entry_ma_extension_pct: float = 0.0
    undertrend_lookback_bars: int = 0
    min_entry_ma_ratio: float = 0.0
    entry_momentum_lookback_bars: int = 0
    min_entry_momentum_pct: float = 0.0
    # Volume confirmation for entries (0 = disabled)
    breakout_volume_lookback: int = 0
    breakout_volume_multiplier: float = 0.0
    panic_volume_lookback: int = 0
    panic_volume_multiplier: float = 0.0
    # Multi-dimensional regime detection (0 = disabled, falls back to simple filter)
    regime_short_ma_bars: int = 0
    regime_long_ma_bars: int = 0
    regime_atr_short_bars: int = 0
    regime_atr_long_bars: int = 0
    regime_volume_short_bars: int = 0
    regime_volume_long_bars: int = 0
    regime_bearish_threshold: float = -0.3
    regime_cautious_threshold: float = 0.0
    regime_cautious_position_scale: float = 0.5


@dataclass(frozen=True)
class RiskConfig:
    leverage: float
    risk_per_trade_pct: float
    max_position_notional_pct: float
    max_total_notional_pct: float
    max_used_margin_pct: float
    max_open_positions: int
    max_daily_loss_pct: float
    max_account_drawdown_pct: float
    allow_scale_in: bool
    max_consecutive_losses: int = 0
    loss_cooldown_bars: int = 0


@dataclass(frozen=True)
class CircuitBreakerConfig:
    benchmark_symbol: str
    benchmark_drop_lookback_bars: int
    benchmark_drop_pct: float
    pause_bars: int
    close_positions_on_trigger: bool


@dataclass(frozen=True)
class RaynConfig:
    portfolio: PortfolioConfig
    strategy: StrategyConfig
    risk: RiskConfig
    circuit_breaker: CircuitBreakerConfig


def load_raw_config(path: str | Path) -> dict:
    with resolve_config_path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_config(path: str | Path) -> RaynConfig:
    raw = load_raw_config(path)

    config = RaynConfig(
        portfolio=PortfolioConfig(**raw["portfolio"]),
        strategy=StrategyConfig(**raw["strategy"]),
        risk=RiskConfig(**raw["risk"]),
        circuit_breaker=CircuitBreakerConfig(**raw["circuit_breaker"]),
    )
    validate_config(config)
    return config


def validate_config(config: RaynConfig) -> None:
    if config.portfolio.initial_equity <= 0:
        raise ValueError("initial_equity must be positive")
    if config.risk.leverage <= 0:
        raise ValueError("leverage must be positive")
    if config.risk.leverage > 10:
        raise ValueError("leverage cap is intentionally limited to 10x")
    if config.strategy.stop_loss_pct <= 0:
        raise ValueError("stop_loss_pct must be positive")
    if config.strategy.take_profit_pct <= 0:
        raise ValueError("take_profit_pct must be positive")
    if config.risk.risk_per_trade_pct <= 0:
        raise ValueError("risk_per_trade_pct must be positive")
    if config.risk.max_total_notional_pct > 1:
        raise ValueError("max_total_notional_pct should not exceed 1.0")
    for hour in config.strategy.blocked_entry_hours_utc:
        if hour < 0 or hour > 23:
            raise ValueError("blocked_entry_hours_utc values must be between 0 and 23")
    if config.strategy.max_entry_range_pct < 0:
        raise ValueError("max_entry_range_pct must be non-negative")
    if config.strategy.volatility_lookback_bars < 0:
        raise ValueError("volatility_lookback_bars must be non-negative")
    if config.strategy.regime_filter_ma_bars < 0:
        raise ValueError("regime_filter_ma_bars must be non-negative")
    if config.strategy.regime_filter_momentum_bars < 0:
        raise ValueError("regime_filter_momentum_bars must be non-negative")
    if config.strategy.max_entry_rsi < 0:
        raise ValueError("max_entry_rsi must be non-negative")
    if config.strategy.overextension_lookback_bars < 0:
        raise ValueError("overextension_lookback_bars must be non-negative")
    if config.strategy.max_entry_ma_extension_pct < 0:
        raise ValueError("max_entry_ma_extension_pct must be non-negative")
    if config.strategy.undertrend_lookback_bars < 0:
        raise ValueError("undertrend_lookback_bars must be non-negative")
    if config.strategy.entry_momentum_lookback_bars < 0:
        raise ValueError("entry_momentum_lookback_bars must be non-negative")
    if config.risk.max_consecutive_losses < 0:
        raise ValueError("max_consecutive_losses must be non-negative")
    if config.risk.loss_cooldown_bars < 0:
        raise ValueError("loss_cooldown_bars must be non-negative")
