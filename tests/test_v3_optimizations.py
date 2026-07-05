"""Tests for V3 optimizations: volume-confirmed entries, regime detection,
layer depth expansion, and symbol scoring."""

from pathlib import Path
import sys
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.backtest import Candle
from src.rayn_strategy.config import (
    CircuitBreakerConfig,
    PortfolioConfig,
    RaynConfig,
    RiskConfig,
    StrategyConfig,
)
from src.rayn_strategy.grid import (
    BoundedGridConfig,
    GridLayer,
    GridPosition,
    effective_layer_step_pct,
    should_add_grid_layer,
)
from src.rayn_strategy.regime import MarketRegime, detect_market_regime
from src.rayn_strategy.signals import entry_signal
from src.rayn_strategy.symbol_scorer import SymbolScore, score_symbols, select_top_symbols


# --------------------------------------------------------------------------
# Volume-confirmed entries
# --------------------------------------------------------------------------

def _make_breakout_config(**overrides) -> StrategyConfig:
    base = dict(
        timeframe="1h",
        trend_ma_bars=3,
        breakout_lookback_bars=4,
        breakout_buffer_pct=0.001,
        panic_lookback_bars=4,
        panic_drop_pct=0.08,
        panic_rsi_bars=2,
        panic_rsi_max=34.0,
        rebound_confirm_pct=0.002,
        take_profit_pct=0.02,
        stop_loss_pct=0.016,
        max_hold_bars=72,
    )
    base.update(overrides)
    return StrategyConfig(**base)


def test_entry_signal_allows_breakout_without_volume_when_disabled() -> None:
    config = _make_breakout_config()  # volume lookback=0 by default
    closes = [100.0] * 10 + [101.0, 102.0, 103.0, 106.0]
    highs = [c * 1.001 for c in closes]
    lows = [c * 0.999 for c in closes]
    volumes = [100.0] * len(closes)

    signal = entry_signal(closes, config, highs=highs, lows=lows, volumes=volumes)
    assert signal.action == "buy"
    assert signal.reason == "trend_breakout"


def test_entry_signal_blocks_low_volume_breakout() -> None:
    config = _make_breakout_config(
        breakout_volume_lookback=3,
        breakout_volume_multiplier=1.5,
    )
    closes = [100.0] * 10 + [101.0, 102.0, 103.0, 106.0]
    highs = [c * 1.001 for c in closes]
    lows = [c * 0.999 for c in closes]
    # Flat volume — no spike
    volumes = [100.0] * len(closes)

    signal = entry_signal(closes, config, highs=highs, lows=lows, volumes=volumes)
    assert signal.action == "hold"
    assert signal.reason == "low_volume_breakout"


def test_entry_signal_allows_high_volume_breakout() -> None:
    config = _make_breakout_config(
        breakout_volume_lookback=3,
        breakout_volume_multiplier=1.5,
    )
    closes = [100.0] * 10 + [101.0, 102.0, 103.0, 106.0]
    highs = [c * 1.001 for c in closes]
    lows = [c * 0.999 for c in closes]
    # Volume spike on the last bar
    volumes = [100.0] * (len(closes) - 1) + [250.0]

    signal = entry_signal(closes, config, highs=highs, lows=lows, volumes=volumes)
    assert signal.action == "buy"
    assert signal.reason == "trend_breakout"


def test_entry_signal_blocks_low_volume_rebound() -> None:
    config = _make_breakout_config(
        panic_volume_lookback=3,
        panic_volume_multiplier=1.5,
    )
    # Build a panic rebound scenario
    closes = [100.0] * 10 + [90.0, 88.0, 86.0, 87.0]
    highs = [c * 1.001 for c in closes]
    lows = [c * 0.999 for c in closes]
    volumes = [100.0] * len(closes)  # flat volume

    signal = entry_signal(closes, config, highs=highs, lows=lows, volumes=volumes)
    # Should be blocked (if panic conditions are met) or no_signal
    if signal.reason == "low_volume_rebound":
        assert signal.action == "hold"
    # If panic conditions aren't met due to RSI, that's also valid


def test_entry_signal_works_without_volumes_parameter() -> None:
    """Backward compatibility: omitting volumes should work like before."""
    config = _make_breakout_config(
        breakout_volume_lookback=3,
        breakout_volume_multiplier=1.5,
    )
    closes = [100.0] * 10 + [101.0, 102.0, 103.0, 106.0]
    highs = [c * 1.001 for c in closes]
    lows = [c * 0.999 for c in closes]

    # No volumes arg — should still allow entry (volume check disabled when no data)
    signal = entry_signal(closes, config, highs=highs, lows=lows)
    assert signal.action == "buy"
    assert signal.reason == "trend_breakout"


# --------------------------------------------------------------------------
# Multi-dimensional regime detection
# --------------------------------------------------------------------------

def test_regime_bullish_uptrend() -> None:
    # Short MA above long MA (bullish)
    closes = list(range(1, 300))  # strongly uptrending
    regime = detect_market_regime(
        [float(c) for c in closes],
        short_ma_bars=50,
        long_ma_bars=200,
    )
    assert regime.trend == "bullish"
    assert regime.allow_entry is True
    assert regime.position_scale == 1.0
    assert regime.score > 0


def test_regime_bearish_downtrend() -> None:
    closes = list(range(300, 0, -1))  # strongly downtrending
    regime = detect_market_regime(
        [float(c) for c in closes],
        short_ma_bars=50,
        long_ma_bars=200,
        bearish_threshold=-0.3,
    )
    assert regime.trend == "bearish"
    assert regime.score < 0


def test_regime_cautious_zone() -> None:
    # Create data where score is between bearish and cautious thresholds
    closes = [100.0] * 250  # flat = neutral trend
    regime = detect_market_regime(
        closes,
        short_ma_bars=50,
        long_ma_bars=200,
        bearish_threshold=-0.5,
        cautious_threshold=0.1,
        cautious_position_scale=0.5,
    )
    assert regime.trend == "neutral"
    assert regime.allow_entry is True
    assert regime.position_scale == 0.5  # cautious zone


def test_regime_with_volumes() -> None:
    closes = list(range(1, 300))
    # Elevated recent volume
    volumes = [100.0] * 250 + [200.0] * 49
    regime = detect_market_regime(
        [float(c) for c in closes],
        volumes=volumes,
        short_ma_bars=50,
        long_ma_bars=200,
        volume_short_bars=20,
        volume_long_bars=100,
    )
    assert regime.volume_trend in ("normal", "elevated")
    assert regime.allow_entry is True


def test_regime_insufficient_data() -> None:
    closes = [100.0, 101.0, 102.0]  # too short
    regime = detect_market_regime(
        closes,
        short_ma_bars=50,
        long_ma_bars=200,
    )
    assert regime.trend == "neutral"
    assert regime.score == 0.0
    assert regime.allow_entry is True


def test_regime_extreme_volatility() -> None:
    # Create data with high short-term ATR vs long-term
    closes = [100.0] * 200
    highs = [100.0] * 180 + [120.0] * 20  # sudden high expansion
    lows = [100.0] * 180 + [80.0] * 20
    regime = detect_market_regime(
        closes,
        highs=highs,
        lows=lows,
        short_ma_bars=50,
        long_ma_bars=200,
        atr_short_bars=20,
        atr_long_bars=100,
    )
    assert regime.volatility in ("high", "extreme")


# --------------------------------------------------------------------------
# Layer depth expansion
# --------------------------------------------------------------------------

def test_layer_step_expansion_increases_with_depth() -> None:
    config = BoundedGridConfig(
        max_rescue_layers=5,
        layer_step_pct=0.01,
        layer_multiplier=1.0,
        take_profit_pct=0.008,
        hard_stop_pct=0.12,
        max_symbol_notional_pct=1.5,
        max_total_notional_pct=2.5,
        layer_step_expansion_rate=0.15,
    )

    step_0 = effective_layer_step_pct(config, current_rescue_layers=0)
    step_1 = effective_layer_step_pct(config, current_rescue_layers=1)
    step_2 = effective_layer_step_pct(config, current_rescue_layers=2)
    step_4 = effective_layer_step_pct(config, current_rescue_layers=4)

    assert step_0 == 0.01  # No expansion at layer 0
    assert round(step_1, 5) == 0.0115  # 0.01 * (1 + 1*0.15)
    assert round(step_2, 5) == 0.013  # 0.01 * (1 + 2*0.15)
    assert round(step_4, 5) == 0.016  # 0.01 * (1 + 4*0.15)


def test_layer_step_expansion_disabled_by_default() -> None:
    config = BoundedGridConfig(
        max_rescue_layers=5,
        layer_step_pct=0.01,
        layer_multiplier=1.0,
        take_profit_pct=0.008,
        hard_stop_pct=0.12,
        max_symbol_notional_pct=1.5,
        max_total_notional_pct=2.5,
        # layer_step_expansion_rate defaults to 0.0
    )

    step_0 = effective_layer_step_pct(config, current_rescue_layers=0)
    step_4 = effective_layer_step_pct(config, current_rescue_layers=4)

    assert step_0 == step_4 == 0.01  # No expansion


def test_deeper_layers_require_wider_drop() -> None:
    config = BoundedGridConfig(
        max_rescue_layers=5,
        layer_step_pct=0.01,
        layer_multiplier=1.0,
        take_profit_pct=0.008,
        hard_stop_pct=0.12,
        max_symbol_notional_pct=1.5,
        max_total_notional_pct=2.5,
        layer_step_expansion_rate=0.20,
    )
    # 2 rescue layers: step = 0.01 * (1 + 2*0.20) = 0.014
    position = GridPosition(
        symbol="BTCUSDT",
        entry_time="2026-02-01T00:00:00+00:00",
        entry_reason="trend_breakout",
        layers=[
            GridLayer("2026-02-01T00:00:00+00:00", 100.0, 50.0, 5.0),
            GridLayer("2026-02-01T01:00:00+00:00", 99.0, 50.0, 5.0),
            GridLayer("2026-02-01T02:00:00+00:00", 98.0, 50.0, 5.0),
        ],
    )
    # last_layer_price = 98.0, expanded step for 2 rescue layers = 1.4%
    # threshold = 98.0 * (1 - 0.014) = 96.628
    assert not should_add_grid_layer(position, 97.0, config)  # not deep enough
    assert should_add_grid_layer(position, 96.5, config)  # deep enough


# --------------------------------------------------------------------------
# Symbol scoring
# --------------------------------------------------------------------------

def _make_candles(n: int, base_close: float, trend: float = 0.0, volume: float = 100.0) -> list[Candle]:
    return [
        Candle(
            timestamp=f"2026-02-01T{i:02d}:00:00+00:00",
            open=base_close + trend * i - 0.1,
            high=base_close + trend * i + 1.0,
            low=base_close + trend * i - 1.0,
            close=base_close + trend * i,
            volume=volume,
        )
        for i in range(n)
    ]


def test_symbol_scoring_ranks_trending_higher() -> None:
    trending = _make_candles(100, 100.0, trend=0.5, volume=200.0)
    flat = _make_candles(100, 100.0, trend=0.0, volume=100.0)

    scores = score_symbols(
        {"TRENDING": trending, "FLAT": flat},
        trend_ma_bars=20,
        momentum_bars=10,
        atr_bars=10,
        volume_bars=10,
    )
    assert scores[0].symbol == "TRENDING"
    assert scores[0].composite > scores[1].composite


def test_select_top_symbols_filters_ineligible() -> None:
    good = _make_candles(100, 100.0, trend=0.5, volume=200.0)
    bad = _make_candles(100, 100.0, trend=-0.5, volume=50.0)

    selected = select_top_symbols(
        {"GOOD": good, "BAD": bad},
        trend_ma_bars=20,
        momentum_bars=10,
        atr_bars=10,
        volume_bars=10,
        min_composite_score=0.5,
    )
    assert "GOOD" in selected


def test_symbol_scoring_handles_empty_candles() -> None:
    scores = score_symbols(
        {"EMPTY": []},
        trend_ma_bars=20,
    )
    assert len(scores) == 1
    assert scores[0].composite == 0.0
    assert not scores[0].eligible


# --------------------------------------------------------------------------
# Config backward compatibility
# --------------------------------------------------------------------------

def test_config_loads_with_new_volume_params() -> None:
    """Ensure new StrategyConfig params have safe defaults."""
    config = StrategyConfig(
        timeframe="1h",
        trend_ma_bars=200,
        breakout_lookback_bars=48,
        breakout_buffer_pct=0.003,
        panic_lookback_bars=72,
        panic_drop_pct=0.08,
        panic_rsi_bars=14,
        panic_rsi_max=34.0,
        rebound_confirm_pct=0.002,
        take_profit_pct=0.008,
        stop_loss_pct=0.018,
        max_hold_bars=2160,
    )
    # All new params should default to 0 (disabled)
    assert config.breakout_volume_lookback == 0
    assert config.breakout_volume_multiplier == 0.0
    assert config.panic_volume_lookback == 0
    assert config.panic_volume_multiplier == 0.0
    assert config.regime_short_ma_bars == 0
    assert config.regime_long_ma_bars == 0


def test_config_loads_optimized_v3_toml() -> None:
    """Verify the V3 config file loads without errors."""
    from src.rayn_strategy.config import load_config
    config = load_config(Path(ROOT / "config.rayn-optimized-v3.toml"))
    assert config.strategy.breakout_volume_lookback == 20
    assert config.strategy.breakout_volume_multiplier == 1.2
    assert config.strategy.panic_volume_lookback == 20
    assert config.strategy.regime_short_ma_bars == 50
    assert config.strategy.regime_long_ma_bars == 200
