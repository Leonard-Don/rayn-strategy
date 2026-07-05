"""Parity: the paper stepper (tools/momentum_paper.py) must be bit-identical to one
iteration of the MomentumEngine backtest loop. Step the paper book bar-by-bar over
the same slice the engine runs, and require the same closed-trade sequence and the
same final equity. If they agree, the paper flow is a faithful mirror of the
validated backtest."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.config import load_config, load_raw_config
from src.rayn_strategy.backtest import Candle
from src.rayn_strategy.market_filter import (
    ShortCandidateFilter,
    breadth_above_ma,
    long_breadth_allows_entry,
    short_candidate_allows_entry,
)
from src.rayn_strategy.momentum import MomentumConfig, MomentumEngine, entry_quality_risk_multiplier
from tools.momentum_paper import (
    Intent,
    MomentumPaper,
    _catch_up_start_index,
    _hist_upto,
    append_intent_ledger,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
N = 80


def _load_momentum_config(name: str) -> MomentumConfig:
    return MomentumConfig(**load_raw_config(ROOT / name)["momentum"])


def _setup():
    cfg = load_config(ROOT / "config.rayn-momentum.toml")
    # disable the account halt so engine (which has none) and paper match exactly
    cfg = replace(
        cfg,
        risk=replace(cfg.risk, max_account_drawdown_pct=0.99, max_open_positions=len(SYMBOLS)),
    )
    mom = MomentumConfig(
        trend_ma_bars=5,
        breakout_bars=3,
        momentum_bars=3,
        min_momentum_pct=0.0,
        rsi_bars=3,
        max_entry_rsi=100.0,
        atr_bars=3,
        stop_atr_mult=2.0,
        trail_atr_mult=3.0,
        risk_per_trade_pct=0.0075,
        max_hold_bars=6,
        regime_ma_bars=5,
        allow_long=True,
        allow_short=False,
    )
    rows = _synthetic_momentum_rows()
    return cfg, mom, rows


def _synthetic_momentum_rows() -> dict[str, list[Candle]]:
    rows = {}
    for offset, symbol in enumerate(SYMBOLS):
        candles = []
        for i in range(N):
            close = 100.0 + offset + i * 0.35
            open_ = close - 0.10
            candles.append(
                Candle(
                    f"t{i:04d}",
                    open_,
                    close + 0.80,
                    close - 0.80,
                    close,
                    1_000.0 + i,
                )
            )
        rows[symbol] = candles
    return rows


def test_paper_matches_engine():
    cfg, mom, rows = _setup()
    n = len(next(iter(rows.values())))

    # engine over the slice
    res = MomentumEngine(config=cfg, mom=mom, data_dir=ROOT / "data_multicycle",
                         symbols=SYMBOLS, preloaded_data=rows).run()
    engine_closed = [(t.symbol, t.exit_reason) for t in res.trades if t.exit_reason != "end_of_test"]
    assert engine_closed

    # paper, stepping every bar from a fresh book
    paper = MomentumPaper(cfg, mom)
    warm = max(mom.trend_ma_bars, mom.breakout_bars, mom.momentum_bars) + 1
    paper_closed = []
    for i in range(warm - 1, n):
        for it in paper.step(SYMBOLS, _hist_upto(rows, SYMBOLS, i)):
            if it.action == "CLOSE":
                paper_closed.append((it.symbol, it.reason))

    # same sequence of closed trades (entries/exits/reasons line up)
    assert paper_closed == engine_closed, (
        f"trade sequences differ: paper {len(paper_closed)} vs engine {len(engine_closed)}")

    # same final equity after closing paper's leftover opens at the final close (as the engine does)
    finals = {s: rows[s][n - 1].close for s in SYMBOLS}
    for s, p in list(paper.positions.items()):
        paper._close(p, finals[s], "end_of_test")
    assert abs(paper.cash - res.final_equity) < 1e-6, (
        f"final equity differs: paper {paper.cash} vs engine {res.final_equity}")


def test_account_halt_flattens_and_stops():
    """The paper profile's 20% account halt should flatten + stop (engine has no halt)."""
    cfg = load_config(ROOT / "config.rayn-momentum-paper.toml")
    mom = _load_momentum_config("config.rayn-momentum-paper.toml")
    paper = MomentumPaper(cfg, mom)
    # Open position large enough that a held -17% mark = >20% account DD, with the
    # stop (70) and liquidation (~67) BOTH untouched, so only the account halt can fire.
    from src.rayn_strategy.momentum import MomPosition
    paper.positions["BTCUSDT"] = MomPosition("BTCUSDT", "t0", 100.0, 600.0, 1, 70.0, 100.0)
    hist = {
        "BTCUSDT": {"closes": [100.0] * 250, "highs": [100.0] * 250, "lows": [100.0] * 250,
                    "bar": {"ts": "t1", "open": 83.0, "high": 85.0, "low": 83.0, "close": 83.0}},
    }
    intents = paper.step(["BTCUSDT"], hist)
    assert paper.halted is True
    assert any(it.reason == "account_halt" for it in intents)
    assert len(paper.positions) == 0


def test_paper_tick_is_idempotent_for_duplicate_timestamp():
    cfg = load_config(ROOT / "config.rayn-momentum-paper.toml")
    mom = _load_momentum_config("config.rayn-momentum-paper.toml")
    paper = MomentumPaper(cfg, mom)
    from src.rayn_strategy.momentum import MomPosition
    paper.positions["BTCUSDT"] = MomPosition("BTCUSDT", "t0", 100.0, 50.0, 1, 70.0, 100.0)
    hist = {
        "BTCUSDT": {
            "closes": [100.0] * 250,
            "highs": [101.0] * 250,
            "lows": [99.0] * 250,
            "bar": {"ts": "t1", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        },
    }

    paper.step(["BTCUSDT"], hist)
    first_bars_held = paper.positions["BTCUSDT"].bars_held
    paper.step(["BTCUSDT"], hist)

    assert paper.positions["BTCUSDT"].bars_held == first_bars_held
    assert paper.last_timestamp == "t1"


def test_paper_tick_ignores_older_timestamp_after_state_load():
    cfg = load_config(ROOT / "config.rayn-momentum-paper.toml")
    mom = _load_momentum_config("config.rayn-momentum-paper.toml")
    paper = MomentumPaper(cfg, mom)
    from src.rayn_strategy.momentum import MomPosition
    paper.positions["BTCUSDT"] = MomPosition("BTCUSDT", "t0", 100.0, 50.0, 1, 70.0, 100.0)
    paper.last_timestamp = "t2"
    hist = {
        "BTCUSDT": {
            "closes": [100.0] * 250,
            "highs": [101.0] * 250,
            "lows": [99.0] * 250,
            "bar": {"ts": "t1", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        },
    }

    intents = paper.step(["BTCUSDT"], hist)

    assert intents == []
    assert paper.positions["BTCUSDT"].bars_held == 0
    assert paper.last_timestamp == "t2"


def test_catch_up_start_index_uses_next_unprocessed_bar():
    ts = ["t1", "t2", "t3", "t4"]

    assert _catch_up_start_index(ts, "t2", warm=1) == 2
    assert _catch_up_start_index(ts, "t4", warm=1) == len(ts)
    assert _catch_up_start_index(ts, None, warm=1) == 3


def test_paper_opens_short_when_short_regime_breaks_down():
    cfg = load_config(ROOT / "config.rayn-momentum-paper.toml")
    cfg = replace(cfg, risk=replace(cfg.risk, max_open_positions=1))
    mom = MomentumConfig(
        trend_ma_bars=5,
        breakout_bars=3,
        momentum_bars=3,
        min_momentum_pct=0.0,
        rsi_bars=3,
        max_entry_rsi=100.0,
        atr_bars=3,
        stop_atr_mult=2.0,
        trail_atr_mult=3.0,
        risk_per_trade_pct=0.0075,
        regime_ma_bars=5,
        allow_short=True,
    )
    paper = MomentumPaper(cfg, mom)
    closes = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0]
    hist = {
        "BTCUSDT": {
            "closes": closes,
            "highs": [c + 1.0 for c in closes],
            "lows": [c - 1.0 for c in closes],
            "bar": {"ts": "t9", "open": 91.5, "high": 92.0, "low": 90.5, "close": 91.0},
        },
    }

    intents = paper.step(["BTCUSDT"], hist)

    assert len(intents) == 1
    assert intents[0].action == "OPEN"
    assert intents[0].side == "SELL"
    assert intents[0].reason == "momentum_short"
    assert intents[0].stop is not None
    assert intents[0].stop > intents[0].price
    assert paper.positions["BTCUSDT"].side == -1


def test_bounce_failure_short_entry_requires_rebound_then_failure():
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
        allow_long=False,
        allow_short=True,
    )
    engine = MomentumEngine(cfg, mom, ROOT, ["BTCUSDT"])
    failed_bounce = [100.0, 99.0, 98.0, 97.0, 96.0, 92.0, 90.0, 92.0, 94.0, 93.0, 91.5]
    waterfall = [104.0, 103.0, 102.0, 101.0, 100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0]

    assert engine._entry_short(failed_bounce) is True
    assert engine._entry_short(waterfall) is False


def test_paper_blocks_shorts_when_core_market_rebounds():
    cfg = load_config(ROOT / "config.rayn-momentum-paper.toml")
    cfg = replace(cfg, risk=replace(cfg.risk, max_open_positions=1))
    base_mom = MomentumConfig(
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
        allow_long=False,
        allow_short=True,
    )
    closes = [100.0, 99.0, 98.0, 98.0, 98.0, 90.0, 91.0, 94.0, 93.0, 92.8]
    hist = {
        symbol: {
            "closes": closes,
            "highs": [c + 1.0 for c in closes],
            "lows": [c - 1.0 for c in closes],
            "bar": {"ts": "t9", "open": 93.0, "high": 93.5, "low": 92.0, "close": 92.8},
        }
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    }

    unfiltered = MomentumPaper(cfg, base_mom).step(["BTCUSDT", "ETHUSDT", "SOLUSDT"], hist)
    filtered_mom = replace(
        base_mom,
        short_rebound_filter_bars=4,
        short_rebound_filter_min_pct=0.02,
        short_rebound_filter_min_count=3,
    )
    filtered = MomentumPaper(cfg, filtered_mom).step(["BTCUSDT", "ETHUSDT", "SOLUSDT"], hist)

    assert any(intent.action == "OPEN" and intent.side == "SELL" for intent in unfiltered)
    assert filtered == []


def test_engine_short_rebound_filter_blocks_core_rebound():
    cfg = load_config(ROOT / "config.rayn-momentum-paper.toml")
    mom = MomentumConfig(
        short_rebound_filter_bars=4,
        short_rebound_filter_min_pct=0.02,
        short_rebound_filter_min_count=3,
    )
    engine = MomentumEngine(cfg, mom, ROOT, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    closes = [100.0, 99.0, 98.0, 98.0, 98.0, 90.0, 91.0, 94.0, 93.0, 92.8]

    assert engine._short_rebound_filter_blocks(
        {"BTCUSDT": closes, "ETHUSDT": closes, "SOLUSDT": closes}
    ) is True
    assert engine._short_rebound_filter_blocks(
        {"BTCUSDT": closes, "ETHUSDT": closes, "SOLUSDT": [100.0] * len(closes)}
    ) is False


def test_entry_quality_scores_stronger_long_breakout_higher():
    cfg = load_config(ROOT / "config.rayn-momentum-paper.toml")
    mom = MomentumConfig(
        trend_ma_bars=5,
        breakout_bars=3,
        momentum_bars=3,
        min_momentum_pct=0.0,
        rsi_bars=3,
        max_entry_rsi=100.0,
        entry_quality_momentum_cap_pct=0.08,
        entry_quality_trend_cap_pct=0.08,
        entry_quality_breakout_cap_pct=0.04,
    )
    engine = MomentumEngine(cfg, mom, ROOT, ["BTCUSDT"])
    weak = [100.0, 100.0, 100.0, 100.0, 100.0, 100.2, 100.3, 100.4, 100.5]
    strong = [100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 102.0, 104.0, 108.0]

    assert engine._entry_long(weak) is True
    assert engine._entry_long(strong) is True
    assert engine._entry_quality(1, strong).score > engine._entry_quality(1, weak).score


def test_entry_quality_risk_multiplier_blocks_low_and_scales_high():
    mom = MomentumConfig(
        entry_quality_min_score=0.40,
        entry_quality_full_score=0.80,
        entry_quality_min_risk_mult=0.50,
        entry_quality_max_risk_mult=1.25,
    )

    assert entry_quality_risk_multiplier(0.39, mom) == 0.0
    assert entry_quality_risk_multiplier(0.40, mom) == pytest.approx(0.50)
    assert entry_quality_risk_multiplier(0.60, mom) == pytest.approx(0.875)
    assert entry_quality_risk_multiplier(0.80, mom) == pytest.approx(1.25)
    assert entry_quality_risk_multiplier(0.95, mom) == pytest.approx(1.25)


def test_trade_ledger_persists_costs_and_realized_pnl(tmp_path):
    path = tmp_path / "ledger.csv"
    intents = [
        Intent(
            "OPEN",
            "BTCUSDT",
            "SELL",
            100.0,
            100.0,
            105.0,
            "momentum_short",
            fee_cost=0.05,
            slippage_cost=0.07,
            realized_pnl=None,
            cash_delta=-0.12,
            quality_score=0.72,
            risk_multiplier=1.10,
        ),
        Intent(
            "CLOSE",
            "BTCUSDT",
            "BUY",
            100.0,
            90.0,
            None,
            "trail_stop",
            fee_cost=0.05,
            slippage_cost=0.07,
            realized_pnl=9.76,
            cash_delta=9.88,
        ),
    ]

    append_intent_ledger(path, "2026-06-08T01:00:00+00:00", intents, equity_after=109.64)

    rows = path.read_text().splitlines()
    assert rows[0] == (
        "timestamp,action,symbol,side,notional,price,stop,reason,fee_cost,"
        "slippage_cost,total_cost,realized_pnl,cash_delta,quality_score,risk_multiplier,equity_after"
    )
    assert "OPEN,BTCUSDT,SELL,100.00000000,100.00000000,105.00000000,momentum_short" in rows[1]
    assert rows[1].endswith(
        "0.05000000,0.07000000,0.12000000,,-0.12000000,0.72000000,1.10000000,109.64000000"
    )
    assert "CLOSE,BTCUSDT,BUY,100.00000000,90.00000000,,trail_stop" in rows[2]
    assert rows[2].endswith("0.05000000,0.07000000,0.12000000,9.76000000,9.88000000,,,109.64000000")


def test_paper_respects_allow_long_false():
    cfg = load_config(ROOT / "config.rayn-momentum-paper.toml")
    mom = MomentumConfig(
        trend_ma_bars=5,
        breakout_bars=3,
        momentum_bars=3,
        min_momentum_pct=0.0,
        rsi_bars=3,
        max_entry_rsi=100.0,
        atr_bars=3,
        stop_atr_mult=2.0,
        trail_atr_mult=3.0,
        risk_per_trade_pct=0.0075,
        regime_ma_bars=5,
        allow_long=False,
        allow_short=True,
    )
    paper = MomentumPaper(cfg, mom)
    closes = [91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0]
    hist = {
        "BTCUSDT": {
            "closes": closes,
            "highs": [c + 1.0 for c in closes],
            "lows": [c - 1.0 for c in closes],
            "bar": {"ts": "t9", "open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0},
        },
    }

    intents = paper.step(["BTCUSDT"], hist)

    assert intents == []
    assert paper.positions == {}


def test_short_candidate_filter_blocks_illiquid_and_extreme_waterfalls():
    candles = [
        Candle(f"t{i}", close * 0.99, close * 1.01, close * 0.99, close, volume)
        for i, (close, volume) in enumerate([(100.0, 1000.0)] * 210)
    ]
    ok_filter = ShortCandidateFilter(
        min_24h_quote_volume=1_000_000.0,
        max_24h_range_pct=0.20,
        max_below_ma_pct=0.20,
        ma_bars=200,
    )
    thin_filter = ShortCandidateFilter(min_24h_quote_volume=10_000_000.0)
    waterfall = [
        Candle(f"w{i}", close, close * 1.02, close * 0.98, close, 1000.0)
        for i, close in enumerate([100.0] * 186 + list(range(100, 75, -1)))
    ]

    assert short_candidate_allows_entry(candles, ok_filter) is True
    assert short_candidate_allows_entry(candles, thin_filter) is False
    assert short_candidate_allows_entry(waterfall, ok_filter) is False


def test_breadth_filter_counts_symbols_above_ma():
    breadth = breadth_above_ma(
        {
            "BTCUSDT": [100.0, 101.0, 102.0, 103.0],
            "ETHUSDT": [103.0, 102.0, 101.0, 100.0],
            "SOLUSDT": [100.0, 100.0],
        },
        ma_bars=3,
    )

    assert breadth.total == 2
    assert breadth.above == 1
    assert breadth.ratio == 0.5
    assert breadth.above_symbols == ("BTCUSDT",)
    assert breadth.below_symbols == ("ETHUSDT",)
    assert long_breadth_allows_entry(breadth, min_ratio=0.6) is False
    assert long_breadth_allows_entry(breadth, min_ratio=0.5) is True


def test_paper_blocks_long_when_market_breadth_is_weak():
    cfg = load_config(ROOT / "config.rayn-momentum-paper.toml")
    mom = MomentumConfig(
        trend_ma_bars=5,
        breakout_bars=3,
        momentum_bars=3,
        min_momentum_pct=0.0,
        rsi_bars=3,
        max_entry_rsi=100.0,
        atr_bars=3,
        stop_atr_mult=2.0,
        trail_atr_mult=3.0,
        risk_per_trade_pct=0.0075,
        regime_ma_bars=5,
        long_breadth_ma_bars=3,
        min_long_breadth_pct=0.75,
        allow_long=True,
        allow_short=False,
    )
    paper = MomentumPaper(cfg, mom)
    btc = [91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0]
    eth = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0]
    hist = {
        "BTCUSDT": {
            "closes": btc,
            "highs": [c + 1.0 for c in btc],
            "lows": [c - 1.0 for c in btc],
            "bar": {"ts": "t9", "open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0},
        },
        "ETHUSDT": {
            "closes": eth,
            "highs": [c + 1.0 for c in eth],
            "lows": [c - 1.0 for c in eth],
            "bar": {"ts": "t9", "open": 91.5, "high": 92.0, "low": 90.5, "close": 91.0},
        },
    }

    intents = paper.step(["BTCUSDT", "ETHUSDT"], hist)

    assert intents == []
    assert paper.positions == {}


def test_short_paper_config_is_independent_short_only_profile():
    cfg = load_config(ROOT / "config.rayn-short-paper.toml")
    mom = _load_momentum_config("config.rayn-short-paper.toml")

    assert cfg.portfolio.initial_equity == 75.0
    assert cfg.risk.max_open_positions == 2
    assert cfg.risk.max_account_drawdown_pct <= 0.15
    assert mom.allow_long is False
    assert mom.allow_short is True
    assert mom.max_entry_rsi <= 70.0
    assert mom.short_rebound_filter_bars == 24
    assert mom.short_rebound_filter_min_count == 3


def test_aggressive_short_paper_config_is_observer_profile():
    cfg = load_config(ROOT / "config.rayn-short-paper-aggressive.toml")
    mom = _load_momentum_config("config.rayn-short-paper-aggressive.toml")

    assert cfg.portfolio.initial_equity == 75.0
    assert cfg.risk.max_open_positions == 2
    assert cfg.risk.max_account_drawdown_pct <= 0.20
    assert mom.allow_long is False
    assert mom.allow_short is True
    assert mom.risk_per_trade_pct == 0.0075


def test_bounce_failure_short_config_is_observer_profile():
    cfg = load_config(ROOT / "config.rayn-short-bounce-paper.toml")
    mom = _load_momentum_config("config.rayn-short-bounce-paper.toml")

    assert cfg.portfolio.initial_equity == 75.0
    assert cfg.risk.max_open_positions == 2
    assert cfg.risk.max_account_drawdown_pct <= 0.15
    assert mom.allow_long is False
    assert mom.allow_short is True
    assert mom.short_entry_mode == "bounce_failure"
    assert mom.risk_per_trade_pct <= 0.005
    assert mom.short_rebound_filter_bars == 24
    assert mom.short_rebound_filter_min_count == 3


if __name__ == "__main__":
    test_paper_matches_engine()
    test_account_halt_flattens_and_stops()
    print("ok")
