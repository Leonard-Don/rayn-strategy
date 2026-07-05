from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.combined_momentum_backtest import DynamicShortRiskConfig, short_risk_multiplier
from tools.combined_momentum_backtest import CombinedMomentumEngine
from src.rayn_strategy.config import load_config
from src.rayn_strategy.momentum import MomentumConfig


def test_dynamic_short_risk_scales_only_in_deep_bear_with_drawdown_room():
    config = DynamicShortRiskConfig(
        enabled=True,
        max_multiplier=1.5,
        min_below_ma_pct=0.05,
        max_drawdown_pct=0.10,
        ma_bars=200,
    )
    deep_bear = [100.0] * 200 + [85.0]
    shallow_bear = [100.0] * 200 + [98.0]

    assert short_risk_multiplier(deep_bear, peak=100.0, marked=98.0, config=config) == 1.5
    assert short_risk_multiplier(deep_bear, peak=100.0, marked=85.0, config=config) == 1.0
    assert short_risk_multiplier(shallow_bear, peak=100.0, marked=100.0, config=config) == 1.0
    assert short_risk_multiplier(deep_bear, peak=100.0, marked=98.0, config=DynamicShortRiskConfig()) == 1.0


def test_combined_engine_blocks_short_entries_during_core_rebound():
    cfg = load_config(ROOT / "config.rayn-momentum-paper.toml")
    mom = MomentumConfig(
        trend_ma_bars=6,
        breakout_bars=3,
        momentum_bars=3,
        min_momentum_pct=0.0,
        rsi_bars=3,
        max_entry_rsi=85.0,
        atr_bars=3,
        stop_atr_mult=2.0,
        trail_atr_mult=3.0,
        risk_per_trade_pct=0.0075,
        regime_ma_bars=6,
        short_entry_mode="bounce_failure",
        short_bounce_lookback_bars=5,
        short_bounce_min_rebound_pct=0.03,
        short_bounce_failure_ma_bars=3,
        short_bounce_max_extension_below_ma_pct=0.03,
        short_rebound_filter_bars=4,
        short_rebound_filter_min_pct=0.02,
        short_rebound_filter_min_count=3,
        allow_long=False,
        allow_short=True,
    )
    engine = CombinedMomentumEngine(
        long_config=cfg,
        long_mom=mom,
        short_config=cfg,
        short_mom=mom,
        data_dir=ROOT,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    )
    closes = [100.0, 99.0, 98.0, 98.0, 98.0, 90.0, 91.0, 94.0, 93.0, 92.8]
    highs = {symbol: [close + 1.0 for close in closes] for symbol in engine.symbols}
    lows = {symbol: [close - 1.0 for close in closes] for symbol in engine.symbols}
    closes_by_symbol = {symbol: closes for symbol in engine.symbols}
    marks = {symbol: closes[-1] for symbol in engine.symbols}
    positions = {}

    cash = engine._try_entries(
        ts="t9",
        side=-1,
        helper=engine.short_helper,
        config=cfg,
        mom=mom,
        label="combined_short",
        closes=closes_by_symbol,
        highs=highs,
        lows=lows,
        candles={},
        marks=marks,
        positions=positions,
        cash=75.0,
    )

    assert cash == 75.0
    assert positions == {}


def test_combined_engine_applies_quality_gate_to_short_entries():
    cfg = load_config(ROOT / "config.rayn-momentum-paper.toml")
    mom = MomentumConfig(
        trend_ma_bars=6,
        breakout_bars=3,
        momentum_bars=3,
        min_momentum_pct=0.0,
        rsi_bars=3,
        max_entry_rsi=85.0,
        atr_bars=3,
        stop_atr_mult=2.0,
        trail_atr_mult=3.0,
        risk_per_trade_pct=0.0075,
        regime_ma_bars=6,
        short_entry_mode="bounce_failure",
        short_bounce_lookback_bars=5,
        short_bounce_min_rebound_pct=0.03,
        short_bounce_failure_ma_bars=3,
        short_bounce_max_extension_below_ma_pct=0.03,
        entry_quality_min_score=1.01,
        allow_long=False,
        allow_short=True,
    )
    engine = CombinedMomentumEngine(
        long_config=cfg,
        long_mom=mom,
        short_config=cfg,
        short_mom=mom,
        data_dir=ROOT,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    )
    closes = [100.0, 99.0, 98.0, 98.0, 98.0, 90.0, 91.0, 94.0, 93.0, 92.8]
    highs = {symbol: [close + 1.0 for close in closes] for symbol in engine.symbols}
    lows = {symbol: [close - 1.0 for close in closes] for symbol in engine.symbols}
    closes_by_symbol = {symbol: closes for symbol in engine.symbols}
    marks = {symbol: closes[-1] for symbol in engine.symbols}
    positions = {}

    cash = engine._try_entries(
        ts="t9",
        side=-1,
        helper=engine.short_helper,
        config=cfg,
        mom=mom,
        label="combined_short",
        closes=closes_by_symbol,
        highs=highs,
        lows=lows,
        candles={},
        marks=marks,
        positions=positions,
        cash=75.0,
    )

    assert cash == 75.0
    assert positions == {}
