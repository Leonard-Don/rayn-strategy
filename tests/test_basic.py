from pathlib import Path
import csv
from dataclasses import replace
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.config import (
    CircuitBreakerConfig,
    PortfolioConfig,
    RaynConfig,
    RiskConfig,
    StrategyConfig,
)
from src.rayn_strategy.backtest import BacktestEngine, Candle
from src.rayn_strategy.grid import (
    BoundedGridConfig,
    BoundedGridEngine,
    GridLayer,
    GridPosition,
    effective_layer_step_pct,
    effective_take_profit_pct,
    estimated_long_liquidation_price,
    grid_exit_reason,
    is_panic_layer_candle,
    next_grid_notional,
    projected_average_entry,
    should_add_grid_layer,
    should_cooldown_symbol,
    should_trim_grid_layers,
)
from src.rayn_strategy.funding import FundingEvent, funding_cost_for_trade
from src.rayn_strategy.exchange_filters import SymbolFilter
from src.rayn_strategy.indicators import atr_pct, average_range_pct, lower_wick_pct, pct_change, rolling_high, rsi, sma, volume_spike
from src.rayn_strategy.signal_check import evaluate_signal_check
from src.rayn_strategy.risk import RiskManager
from src.rayn_strategy.signals import entry_signal
from src.rayn_strategy.signals import market_regime_filter_reason
from src.rayn_strategy.probation import TradeLike, select_eligible_symbols, symbol_stats


def test_indicators() -> None:
    values = [1, 2, 3, 4, 5]
    assert sma(values, 3) == 4
    assert rolling_high(values, 4) == 5
    assert round(pct_change(100, 105), 4) == 0.05
    assert rsi([1, 2, 3, 2, 4, 5], 5) is not None
    assert round(average_range_pct([12, 12], [8, 8], [10, 10], 2), 4) == 0.4
    assert round(atr_pct([12, 13, 14], [10, 11, 12], [11, 12, 13], 2), 4) == 0.1603
    assert round(lower_wick_pct(open_price=100, low=90, close=98), 4) == 0.0816
    assert volume_spike([100, 120, 110, 250], 3, 2.0)


def test_entry_signal_blocks_high_volatility() -> None:
    config = make_config().strategy
    closes = [100.0] * 220 + [101.0, 102.0, 103.0, 106.0]
    highs = [close * 1.08 for close in closes]
    lows = [close * 0.92 for close in closes]

    signal = entry_signal(
        closes,
        config,
        highs=highs,
        lows=lows,
        timestamp="2026-05-30T08:00:00+00:00",
    )

    assert signal.action == "hold"
    assert signal.reason == "volatility_filter"


def test_entry_signal_blocks_configured_utc_hour() -> None:
    config = make_config().strategy
    closes = [100.0] * 220 + [101.0, 102.0, 103.0, 106.0]
    highs = [close * 1.002 for close in closes]
    lows = [close * 0.998 for close in closes]

    signal = entry_signal(
        closes,
        config,
        highs=highs,
        lows=lows,
        timestamp="2026-05-30T15:00:00+00:00",
    )

    assert signal.action == "hold"
    assert signal.reason == "blocked_hour"


def test_entry_signal_blocks_overbought_breakout() -> None:
    config = replace(
        make_config().strategy,
        max_entry_rsi=70.0,
        overextension_lookback_bars=3,
        max_entry_ma_extension_pct=0.04,
    )
    closes = [100.0] * 220 + [103.0, 106.0, 112.0, 118.0]
    highs = [close * 1.002 for close in closes]
    lows = [close * 0.998 for close in closes]

    signal = entry_signal(
        closes,
        config,
        highs=highs,
        lows=lows,
        timestamp="2026-05-30T08:00:00+00:00",
    )

    assert signal.action == "hold"
    assert signal.reason == "overbought_filter"


def test_entry_signal_blocks_weak_symbol_rebound() -> None:
    config = replace(
        make_config().strategy,
        undertrend_lookback_bars=5,
        min_entry_ma_ratio=-0.04,
        entry_momentum_lookback_bars=4,
        min_entry_momentum_pct=-0.08,
    )
    closes = [100.0] * 220 + [90.0, 85.0, 80.0, 78.0, 75.0, 76.0]
    highs = [close * 1.002 for close in closes]
    lows = [close * 0.998 for close in closes]

    signal = entry_signal(
        closes,
        config,
        highs=highs,
        lows=lows,
        timestamp="2026-05-30T08:00:00+00:00",
    )

    assert signal.action == "hold"
    assert signal.reason == "weakness_filter"


def test_market_regime_filter_blocks_weak_benchmark_momentum() -> None:
    config = replace(
        make_config().strategy,
        regime_filter_ma_bars=5,
        regime_filter_min_ma_ratio=-0.01,
        regime_filter_momentum_bars=4,
        regime_filter_min_momentum_pct=-0.03,
    )
    closes = [100.0, 101.0, 100.0, 98.0, 95.0, 94.0, 93.0]

    assert market_regime_filter_reason(closes, config) == "market_regime_filter"


def test_market_regime_filter_allows_stable_benchmark() -> None:
    config = replace(
        make_config().strategy,
        regime_filter_ma_bars=5,
        regime_filter_min_ma_ratio=-0.01,
        regime_filter_momentum_bars=4,
        regime_filter_min_momentum_pct=-0.03,
    )
    closes = [100.0, 101.0, 102.0, 101.5, 102.5, 103.0, 104.0]

    assert market_regime_filter_reason(closes, config) == ""


def test_risk_manager_enters_loss_cooldown_after_streak() -> None:
    config = make_config()
    manager = RiskManager(config)
    state = manager.initial_state()

    manager.record_trade_result(state, -1.0)
    assert not manager.loss_cooldown_active(state)

    manager.record_trade_result(state, -1.0)
    assert manager.loss_cooldown_active(state)

    manager.advance_loss_cooldown(state)
    assert manager.loss_cooldown_active(state)
    manager.advance_loss_cooldown(state)
    assert not manager.loss_cooldown_active(state)


def test_backtest_only_opens_entries_inside_entry_window() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        data_dir = Path(raw_tmp)
        write_test_candles(data_dir / "BTCUSDT.csv")
        config = make_config()

        result = BacktestEngine(
            config=config,
            data_dir=data_dir,
            symbols=["BTCUSDT"],
            entry_start="2026-02-01T05:00:00+00:00",
            entry_end="2026-02-01T08:00:00+00:00",
        ).run()

        assert result.trades
        assert all("2026-02-01T05:00:00+00:00" <= trade.entry_time < "2026-02-01T08:00:00+00:00" for trade in result.trades)


def test_backtest_run_end_closes_without_future_prices() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        data_dir = Path(raw_tmp)
        write_test_candles(data_dir / "BTCUSDT.csv")
        config = make_config()

        result = BacktestEngine(
            config=config,
            data_dir=data_dir,
            symbols=["BTCUSDT"],
            entry_start="2026-02-01T05:00:00+00:00",
            entry_end="2026-02-01T08:00:00+00:00",
            run_end="2026-02-01T08:00:00+00:00",
        ).run()

        assert result.trades
        assert all(trade.exit_time < "2026-02-01T08:00:00+00:00" for trade in result.trades)


def test_bounded_grid_respects_rescue_layer_cap() -> None:
    config = BoundedGridConfig(
        max_rescue_layers=2,
        layer_step_pct=0.01,
        layer_multiplier=1.25,
        take_profit_pct=0.006,
        hard_stop_pct=0.05,
        max_symbol_notional_pct=0.20,
        max_total_notional_pct=0.40,
    )
    position = GridPosition(
        symbol="BTCUSDT",
        entry_time="2026-02-01T00:00:00+00:00",
        entry_reason="trend_breakout",
        layers=[
            GridLayer("2026-02-01T00:00:00+00:00", 100.0, 100.0, 0.08),
            GridLayer("2026-02-01T01:00:00+00:00", 99.0, 125.0, 0.10),
            GridLayer("2026-02-01T02:00:00+00:00", 98.0, 156.25, 0.125),
        ],
    )

    assert not should_add_grid_layer(position, 96.0, config)


def test_bounded_grid_adds_next_layer_after_step() -> None:
    config = BoundedGridConfig(
        max_rescue_layers=2,
        layer_step_pct=0.01,
        layer_multiplier=1.25,
        take_profit_pct=0.006,
        hard_stop_pct=0.05,
        max_symbol_notional_pct=0.20,
        max_total_notional_pct=0.40,
    )
    position = GridPosition(
        symbol="BTCUSDT",
        entry_time="2026-02-01T00:00:00+00:00",
        entry_reason="trend_breakout",
        layers=[GridLayer("2026-02-01T00:00:00+00:00", 100.0, 100.0, 0.08)],
    )

    assert not should_add_grid_layer(position, 99.1, config)
    assert should_add_grid_layer(position, 99.0, config)


def test_bounded_grid_requires_panic_candle_when_layer_filter_enabled() -> None:
    config = BoundedGridConfig(
        max_rescue_layers=2,
        layer_step_pct=0.01,
        layer_multiplier=1.25,
        take_profit_pct=0.006,
        hard_stop_pct=0.05,
        max_symbol_notional_pct=0.20,
        max_total_notional_pct=0.40,
        volume_layer_lookback_bars=3,
        volume_layer_multiplier=2.0,
        min_layer_lower_wick_pct=0.05,
    )
    position = GridPosition(
        symbol="BTCUSDT",
        entry_time="2026-02-01T00:00:00+00:00",
        entry_reason="trend_breakout",
        layers=[GridLayer("2026-02-01T00:00:00+00:00", 100.0, 100.0, 10.0)],
    )
    quiet_drop = [
        Candle("2026-02-01T00:00:00+00:00", 100, 101, 99, 100, 100),
        Candle("2026-02-01T01:00:00+00:00", 100, 101, 99, 100, 110),
        Candle("2026-02-01T02:00:00+00:00", 100, 101, 99, 100, 120),
        Candle("2026-02-01T03:00:00+00:00", 99, 99, 98.8, 99, 130),
    ]
    panic_drop = [
        Candle("2026-02-01T00:00:00+00:00", 100, 101, 99, 100, 100),
        Candle("2026-02-01T01:00:00+00:00", 100, 101, 99, 100, 110),
        Candle("2026-02-01T02:00:00+00:00", 100, 101, 99, 100, 120),
        Candle("2026-02-01T03:00:00+00:00", 99, 99.2, 90, 98, 300),
    ]

    assert not should_add_grid_layer(position, 99.0, config, recent_candles=quiet_drop)
    assert should_add_grid_layer(position, 98.0, config, recent_candles=panic_drop)
    assert is_panic_layer_candle(panic_drop, config)


def test_bounded_grid_uses_atr_step_and_key_layer_multiplier() -> None:
    config = BoundedGridConfig(
        max_rescue_layers=2,
        layer_step_pct=0.01,
        layer_multiplier=1.0,
        take_profit_pct=0.006,
        hard_stop_pct=0.05,
        max_symbol_notional_pct=0.20,
        max_total_notional_pct=0.40,
        atr_layer_lookback_bars=3,
        atr_layer_step_multiplier=0.5,
        key_layer_multiplier=2.0,
    )
    position = GridPosition(
        symbol="BTCUSDT",
        entry_time="2026-02-01T00:00:00+00:00",
        entry_reason="trend_breakout",
        layers=[GridLayer("2026-02-01T00:00:00+00:00", 100.0, 100.0, 10.0)],
    )
    recent = [
        Candle("2026-02-01T00:00:00+00:00", 100, 102, 98, 100, 100),
        Candle("2026-02-01T01:00:00+00:00", 100, 103, 97, 100, 100),
        Candle("2026-02-01T02:00:00+00:00", 100, 104, 96, 100, 100),
    ]

    assert effective_layer_step_pct(config, recent) == 0.03
    assert not should_add_grid_layer(position, 98.0, config, recent_candles=recent)
    assert should_add_grid_layer(position, 97.0, config, recent_candles=recent)
    assert next_grid_notional(position, config, key_layer=True) == 200.0


def test_bounded_grid_records_equity_curve() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        data_dir = Path(raw_tmp)
        write_test_candles(data_dir / "BTCUSDT.csv")

        result = BoundedGridEngine(
            config=make_config(),
            grid_config=BoundedGridConfig(
                max_rescue_layers=1,
                layer_step_pct=0.01,
                layer_multiplier=1.1,
                take_profit_pct=0.006,
                hard_stop_pct=0.05,
                max_symbol_notional_pct=0.20,
                max_total_notional_pct=0.40,
            ),
            data_dir=data_dir,
            symbols=["BTCUSDT"],
            entry_start="2026-02-01T05:00:00+00:00",
            entry_end="2026-02-01T08:00:00+00:00",
            run_end="2026-02-01T10:00:00+00:00",
        ).run()

        assert result.equity_curve
        assert all(point.marked_equity > 0 for point in result.equity_curve)
        assert any(point.open_notional > 0 for point in result.equity_curve)
        open_points = [point for point in result.equity_curve if point.open_notional > 0]
        assert open_points
        assert max(point.max_layers for point in open_points) >= 1
        assert all(point.min_liquidation_buffer_pct is not None for point in open_points)


def test_bounded_grid_skips_entry_when_min_order_distortion_is_too_high() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        data_dir = Path(raw_tmp)
        write_test_candles(data_dir / "BTCUSDT.csv")

        result = BoundedGridEngine(
            config=replace(make_config(), portfolio=replace(make_config().portfolio, initial_equity=100.0)),
            grid_config=BoundedGridConfig(
                max_rescue_layers=1,
                layer_step_pct=0.01,
                layer_multiplier=1.1,
                take_profit_pct=0.006,
                hard_stop_pct=0.05,
                max_symbol_notional_pct=2.0,
                max_total_notional_pct=2.0,
            ),
            data_dir=data_dir,
            symbols=["BTCUSDT"],
            entry_start="2026-02-01T05:00:00+00:00",
            entry_end="2026-02-01T08:00:00+00:00",
            run_end="2026-02-01T10:00:00+00:00",
            symbol_filters={
                "BTCUSDT": SymbolFilter(min_qty=1.0, step_size=1.0, min_notional=0.0)
            },
            max_min_order_distortion=1.5,
        ).run()

        assert not result.trades
        assert result.skipped_entries["min_order_distortion"] > 0


def test_bounded_grid_skips_entries_when_market_regime_blocked() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        data_dir = Path(raw_tmp)
        write_test_candles(data_dir / "BTCUSDT.csv")
        config = replace(
            make_config(),
            strategy=replace(
                make_config().strategy,
                regime_filter_ma_bars=3,
                regime_filter_min_ma_ratio=0.50,
            ),
        )

        result = BoundedGridEngine(
            config=config,
            grid_config=BoundedGridConfig(
                max_rescue_layers=1,
                layer_step_pct=0.01,
                layer_multiplier=1.1,
                take_profit_pct=0.006,
                hard_stop_pct=0.05,
                max_symbol_notional_pct=0.20,
                max_total_notional_pct=0.40,
            ),
            data_dir=data_dir,
            symbols=["BTCUSDT"],
            entry_start="2026-02-01T05:00:00+00:00",
            entry_end="2026-02-01T08:00:00+00:00",
            run_end="2026-02-01T10:00:00+00:00",
        ).run()

        assert not result.trades
        assert result.skipped_entries["market_regime_filter"] > 0


def test_bounded_grid_skips_symbol_during_cooldown() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        data_dir = Path(raw_tmp)
        write_test_candles(data_dir / "BTCUSDT.csv")

        result = BoundedGridEngine(
            config=make_config(),
            grid_config=BoundedGridConfig(
                max_rescue_layers=1,
                layer_step_pct=0.01,
                layer_multiplier=1.1,
                take_profit_pct=0.006,
                hard_stop_pct=0.05,
                max_symbol_notional_pct=0.20,
                max_total_notional_pct=0.40,
                liquidation_exit_buffer_pct=1.0,
                symbol_cooldown_bars=100,
                symbol_cooldown_exit_reasons=["liquidation_buffer_exit"],
            ),
            data_dir=data_dir,
            symbols=["BTCUSDT"],
            entry_start="2026-02-01T05:00:00+00:00",
            entry_end="2026-02-01T10:00:00+00:00",
            run_end="2026-02-01T10:00:00+00:00",
        ).run()

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "liquidation_buffer_exit"
        assert result.skipped_entries["symbol_cooldown"] > 0


def test_bounded_grid_scales_symbol_during_probation() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        data_dir = Path(raw_tmp)
        write_test_candles(data_dir / "BTCUSDT.csv")

        result = BoundedGridEngine(
            config=make_config(),
            grid_config=BoundedGridConfig(
                max_rescue_layers=1,
                layer_step_pct=0.01,
                layer_multiplier=1.1,
                take_profit_pct=0.006,
                hard_stop_pct=0.05,
                max_symbol_notional_pct=0.20,
                max_total_notional_pct=0.40,
                liquidation_exit_buffer_pct=1.0,
                symbol_cooldown_bars=100,
                symbol_cooldown_position_scale=0.5,
                symbol_cooldown_exit_reasons=["liquidation_buffer_exit"],
            ),
            data_dir=data_dir,
            symbols=["BTCUSDT"],
            entry_start="2026-02-01T05:00:00+00:00",
            entry_end="2026-02-01T10:00:00+00:00",
            run_end="2026-02-01T10:00:00+00:00",
        ).run()

        assert len(result.trades) > 1
        assert result.trades[1].notional < result.trades[0].notional
        assert result.skipped_entries["symbol_probation_sizing"] > 0


def test_estimated_long_liquidation_price_uses_maintenance_margin() -> None:
    price = estimated_long_liquidation_price(
        average_entry=100.0,
        leverage=10.0,
        maintenance_margin_pct=0.005,
    )

    assert round(price, 4) == 90.5


def test_projected_average_entry_includes_next_layer() -> None:
    position = GridPosition(
        symbol="BTCUSDT",
        entry_time="2026-02-01T00:00:00+00:00",
        entry_reason="trend_breakout",
        layers=[GridLayer("2026-02-01T00:00:00+00:00", 100.0, 100.0, 10.0)],
    )

    assert round(projected_average_entry(position, 90.0, 100.0), 4) == 95.0


def test_grid_exit_reason_uses_liquidation_buffer_guard() -> None:
    base_config = make_config()
    config = replace(base_config, risk=replace(base_config.risk, leverage=10.0))
    guarded_grid = BoundedGridConfig(
        max_rescue_layers=2,
        layer_step_pct=0.01,
        layer_multiplier=1.25,
        take_profit_pct=0.006,
        hard_stop_pct=0.20,
        max_symbol_notional_pct=0.50,
        max_total_notional_pct=0.80,
        liquidation_exit_buffer_pct=0.04,
    )
    position = GridPosition(
        symbol="BTCUSDT",
        entry_time="2026-02-01T00:00:00+00:00",
        entry_reason="trend_breakout",
        layers=[GridLayer("2026-02-01T00:00:00+00:00", 100.0, 100.0, 10.0)],
    )
    candle = Candle(
        timestamp="2026-02-01T01:00:00+00:00",
        open=94.0,
        high=95.0,
        low=93.0,
        close=94.0,
        volume=1000.0,
    )

    assert grid_exit_reason(position, candle, False, config, guarded_grid) == "liquidation_buffer_exit"


def test_should_trim_grid_layers_after_buffer_warning() -> None:
    config = BoundedGridConfig(
        max_rescue_layers=4,
        layer_step_pct=0.01,
        layer_multiplier=1.25,
        take_profit_pct=0.006,
        hard_stop_pct=0.20,
        max_symbol_notional_pct=0.50,
        max_total_notional_pct=0.80,
        layer_trim_buffer_pct=0.02,
        layer_trim_min_layers=3,
    )
    position = GridPosition(
        symbol="BTCUSDT",
        entry_time="2026-02-01T00:00:00+00:00",
        entry_reason="trend_breakout",
        layers=[
            GridLayer("2026-02-01T00:00:00+00:00", 100.0, 100.0, 10.0),
            GridLayer("2026-02-01T01:00:00+00:00", 96.0, 100.0, 10.0),
            GridLayer("2026-02-01T02:00:00+00:00", 92.0, 100.0, 10.0),
        ],
    )

    assert should_trim_grid_layers(position, 88.5, config, leverage=10.0)
    assert not should_trim_grid_layers(position, 100.0, config, leverage=10.0)


def test_deep_take_profit_overrides_default_take_profit() -> None:
    config = BoundedGridConfig(
        max_rescue_layers=4,
        layer_step_pct=0.01,
        layer_multiplier=1.25,
        take_profit_pct=0.010,
        hard_stop_pct=0.20,
        max_symbol_notional_pct=0.50,
        max_total_notional_pct=0.80,
        deep_take_profit_layer=3,
        deep_take_profit_pct=0.003,
    )
    shallow = GridPosition(
        symbol="BTCUSDT",
        entry_time="2026-02-01T00:00:00+00:00",
        entry_reason="trend_breakout",
        layers=[GridLayer("2026-02-01T00:00:00+00:00", 100.0, 100.0, 10.0)],
    )
    deep = GridPosition(
        symbol="BTCUSDT",
        entry_time="2026-02-01T00:00:00+00:00",
        entry_reason="trend_breakout",
        layers=[
            GridLayer("2026-02-01T00:00:00+00:00", 100.0, 100.0, 10.0),
            GridLayer("2026-02-01T01:00:00+00:00", 96.0, 100.0, 10.0),
            GridLayer("2026-02-01T02:00:00+00:00", 92.0, 100.0, 10.0),
        ],
    )

    assert effective_take_profit_pct(shallow, config) == 0.010
    assert effective_take_profit_pct(deep, config) == 0.003


def test_dynamic_take_profit_uses_atr_inside_bounds() -> None:
    config = BoundedGridConfig(
        max_rescue_layers=4,
        layer_step_pct=0.01,
        layer_multiplier=1.25,
        take_profit_pct=0.006,
        hard_stop_pct=0.20,
        max_symbol_notional_pct=0.50,
        max_total_notional_pct=0.80,
        dynamic_take_profit_atr_multiplier=0.5,
        dynamic_take_profit_min_pct=0.005,
        dynamic_take_profit_max_pct=0.03,
    )
    position = GridPosition(
        symbol="BTCUSDT",
        entry_time="2026-02-01T00:00:00+00:00",
        entry_reason="trend_breakout",
        layers=[GridLayer("2026-02-01T00:00:00+00:00", 100.0, 100.0, 10.0)],
    )

    assert effective_take_profit_pct(position, config, recent_atr_pct=0.04) == 0.02
    assert effective_take_profit_pct(position, config, recent_atr_pct=0.20) == 0.03


def test_should_cooldown_symbol_after_tail_exit_or_deep_loss() -> None:
    config = BoundedGridConfig(
        max_rescue_layers=4,
        layer_step_pct=0.01,
        layer_multiplier=1.25,
        take_profit_pct=0.010,
        hard_stop_pct=0.20,
        max_symbol_notional_pct=0.50,
        max_total_notional_pct=0.80,
        symbol_cooldown_bars=24,
        symbol_cooldown_min_loss_layers=4,
        symbol_cooldown_exit_reasons=["liquidation_buffer_exit"],
    )
    tail_trade = TradeLike(
        "ONDOUSDT",
        1.0,
        "trend_breakout|layers=1",
        "liquidation_buffer_exit",
    )
    deep_loss = TradeLike(
        "ONDOUSDT",
        -1.0,
        "trend_breakout|layers=4",
        "grid_take_profit",
    )
    shallow_loss = TradeLike(
        "ONDOUSDT",
        -1.0,
        "trend_breakout|layers=2",
        "grid_take_profit",
    )

    assert should_cooldown_symbol(tail_trade, config)
    assert should_cooldown_symbol(deep_loss, config)
    assert not should_cooldown_symbol(shallow_loss, config)


def test_symbol_probation_rejects_tail_loss_symbol() -> None:
    trades = [
        TradeLike("ZECUSDT", 10.0, "trend_breakout|layers=1", "grid_take_profit"),
        TradeLike("ZECUSDT", 8.0, "trend_breakout|layers=2", "grid_take_profit"),
        TradeLike("HYPEUSDT", 6.0, "trend_breakout|layers=1", "grid_take_profit"),
        TradeLike("HYPEUSDT", -40.0, "trend_breakout|layers=4", "grid_hard_stop"),
    ]
    stats = symbol_stats(trades)

    selected = select_eligible_symbols(
        ["BTCUSDT", "ZECUSDT", "HYPEUSDT"],
        stats,
        benchmark_symbol="BTCUSDT",
        min_profit_factor=1.1,
        min_pnl=0.0,
        allow_deep_layer_loss=False,
    )

    assert selected == ["BTCUSDT", "ZECUSDT"]


def test_funding_cost_for_long_trade() -> None:
    trade = type(
        "FundingTrade",
        (),
        {
            "entry_time": "2026-02-01T00:00:00+00:00",
            "exit_time": "2026-02-01T16:00:00+00:00",
            "notional": 1000.0,
        },
    )()
    events = [
        FundingEvent("2026-02-01T00:00:00+00:00", 0.001, 100.0),
        FundingEvent("2026-02-01T08:00:00+00:00", 0.0001, 100.0),
        FundingEvent("2026-02-01T16:00:00+00:00", -0.00005, 100.0),
        FundingEvent("2026-02-02T00:00:00+00:00", 0.001, 100.0),
    ]

    assert round(funding_cost_for_trade(trade, events), 4) == 0.05


def test_signal_check_stops_when_equity_below_stage_stop() -> None:
    result = evaluate_signal_check(
        make_config(),
        {"BTCUSDT": make_signal_candles()},
        equity=77.0,
        stop_equity=78.0,
    )

    assert result.verdict == "STOP"
    assert result.primary_reason == "stage_stop"


def test_signal_check_goes_when_entry_signal_is_available() -> None:
    result = evaluate_signal_check(
        make_config(),
        {"BTCUSDT": make_signal_candles()},
        equity=100.0,
        stop_equity=78.0,
    )

    assert result.verdict == "GO"
    assert result.buy_symbols == ["BTCUSDT"]


def test_signal_check_stops_when_benchmark_circuit_breaker_triggers() -> None:
    candles = make_signal_candles()
    candles[-1] = Candle(
        timestamp=candles[-1].timestamp,
        open=94.0,
        high=95.0,
        low=92.0,
        close=92.0,
        volume=1000.0,
    )

    result = evaluate_signal_check(make_config(), {"BTCUSDT": candles})

    assert result.verdict == "STOP"
    assert result.primary_reason == "benchmark_drop"


def make_config() -> RaynConfig:
    return RaynConfig(
        portfolio=PortfolioConfig(
            initial_equity=10000.0,
            quote_asset="USDT",
            taker_fee_bps=5.0,
            slippage_bps=3.0,
        ),
        strategy=StrategyConfig(
            timeframe="1h",
            trend_ma_bars=2,
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
            volatility_lookback_bars=2,
            max_entry_range_pct=0.05,
            blocked_entry_hours_utc=[15],
        ),
        risk=RiskConfig(
            leverage=2.0,
            risk_per_trade_pct=0.0005,
            max_position_notional_pct=0.10,
            max_total_notional_pct=0.30,
            max_used_margin_pct=0.16,
            max_open_positions=2,
            max_daily_loss_pct=0.008,
            max_account_drawdown_pct=0.05,
            allow_scale_in=False,
            max_consecutive_losses=2,
            loss_cooldown_bars=2,
        ),
        circuit_breaker=CircuitBreakerConfig(
            benchmark_symbol="BTCUSDT",
            benchmark_drop_lookback_bars=4,
            benchmark_drop_pct=0.03,
            pause_bars=24,
            close_positions_on_trigger=True,
        ),
    )


def write_test_candles(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for hour in range(12):
            close = 100.0 + hour
            writer.writerow(
                [
                    f"2026-02-01T{hour:02d}:00:00+00:00",
                    close - 0.2,
                    close * 1.03,
                    close * 0.99,
                    close,
                    1000,
                ]
            )


def make_signal_candles() -> list[Candle]:
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 106.0]
    return [
        Candle(
            timestamp=f"2026-02-01T{hour:02d}:00:00+00:00",
            open=close - 0.2,
            high=close * 1.002,
            low=close * 0.998,
            close=close,
            volume=1000.0,
        )
        for hour, close in enumerate(closes)
    ]


if __name__ == "__main__":
    test_indicators()
    test_entry_signal_blocks_high_volatility()
    test_entry_signal_blocks_configured_utc_hour()
    test_entry_signal_blocks_overbought_breakout()
    test_entry_signal_blocks_weak_symbol_rebound()
    test_market_regime_filter_blocks_weak_benchmark_momentum()
    test_market_regime_filter_allows_stable_benchmark()
    test_risk_manager_enters_loss_cooldown_after_streak()
    test_backtest_only_opens_entries_inside_entry_window()
    test_backtest_run_end_closes_without_future_prices()
    test_bounded_grid_respects_rescue_layer_cap()
    test_bounded_grid_adds_next_layer_after_step()
    test_bounded_grid_records_equity_curve()
    test_bounded_grid_skips_entry_when_min_order_distortion_is_too_high()
    test_bounded_grid_skips_entries_when_market_regime_blocked()
    test_bounded_grid_skips_symbol_during_cooldown()
    test_bounded_grid_scales_symbol_during_probation()
    test_estimated_long_liquidation_price_uses_maintenance_margin()
    test_projected_average_entry_includes_next_layer()
    test_grid_exit_reason_uses_liquidation_buffer_guard()
    test_should_trim_grid_layers_after_buffer_warning()
    test_deep_take_profit_overrides_default_take_profit()
    test_should_cooldown_symbol_after_tail_exit_or_deep_loss()
    test_symbol_probation_rejects_tail_loss_symbol()
    test_funding_cost_for_long_trade()
    test_signal_check_stops_when_equity_below_stage_stop()
    test_signal_check_goes_when_entry_signal_is_available()
    test_signal_check_stops_when_benchmark_circuit_breaker_triggers()
    print("ok")
