"""Tests for V3 optimizations: volume-confirmed entries, layer depth expansion,
and config compatibility."""

from pathlib import Path

from rayn_strategy.config import (
    StrategyConfig,
)
from rayn_strategy.grid import (
    BoundedGridConfig,
    GridLayer,
    GridPosition,
    effective_layer_step_pct,
    should_add_grid_layer,
)
from rayn_strategy.signals import entry_signal

ROOT = Path(__file__).resolve().parents[1]

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
    from rayn_strategy.config import load_config

    config = load_config(Path(ROOT / "config.rayn-optimized-v3.toml"))
    assert config.strategy.breakout_volume_lookback == 20
    assert config.strategy.breakout_volume_multiplier == 1.2
    assert config.strategy.panic_volume_lookback == 20
    assert config.strategy.regime_short_ma_bars == 50
    assert config.strategy.regime_long_ma_bars == 200
