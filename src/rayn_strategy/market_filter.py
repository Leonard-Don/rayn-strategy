"""Market-wide filters used by momentum paper/backtests."""

from __future__ import annotations

from dataclasses import dataclass

from .indicators import sma


@dataclass(frozen=True)
class MarketBreadth:
    total: int
    above: int
    ratio: float
    above_symbols: tuple[str, ...]
    below_symbols: tuple[str, ...]


@dataclass(frozen=True)
class ShortCandidateFilter:
    min_24h_quote_volume: float = 0.0
    max_24h_range_pct: float = 0.0
    max_below_ma_pct: float = 0.0
    ma_bars: int = 200
    lookback_bars: int = 24


def breadth_above_ma(closes_by_symbol: dict[str, list[float]], ma_bars: int) -> MarketBreadth:
    above: list[str] = []
    below: list[str] = []
    if ma_bars <= 0:
        return MarketBreadth(0, 0, 0.0, (), ())

    for symbol in sorted(closes_by_symbol):
        closes = closes_by_symbol[symbol]
        ma = sma(closes, ma_bars)
        if ma is None:
            continue
        if closes[-1] > ma:
            above.append(symbol)
        else:
            below.append(symbol)

    total = len(above) + len(below)
    ratio = len(above) / total if total else 0.0
    return MarketBreadth(total, len(above), ratio, tuple(above), tuple(below))


def long_breadth_allows_entry(breadth: MarketBreadth, min_ratio: float) -> bool:
    if min_ratio <= 0.0 or breadth.total == 0:
        return True
    return breadth.ratio >= min_ratio


def short_candidate_allows_entry(candles: list[object], config: ShortCandidateFilter) -> bool:
    if not candles:
        return False
    closes = [float(c.close) for c in candles]
    lookback = min(config.lookback_bars, len(candles))
    recent = candles[-lookback:]
    last_close = closes[-1]

    if config.min_24h_quote_volume > 0.0:
        quote_volume = sum(float(c.volume) * float(c.close) for c in recent)
        if quote_volume < config.min_24h_quote_volume:
            return False

    if config.max_24h_range_pct > 0.0:
        low = min(float(c.low) for c in recent)
        high = max(float(c.high) for c in recent)
        if low <= 0 or (high / low - 1.0) > config.max_24h_range_pct:
            return False

    if config.max_below_ma_pct > 0.0 and config.ma_bars > 0:
        ma = sma(closes, config.ma_bars)
        if ma is None or ma <= 0:
            return False
        if (ma - last_close) / ma > config.max_below_ma_pct:
            return False

    return True
