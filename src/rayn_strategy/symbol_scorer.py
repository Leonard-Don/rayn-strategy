"""Dynamic symbol scoring for multi-factor ranking and selection.

Scores each symbol on three dimensions — trend, volatility, and volume — then
combines them into a weighted composite.  The resulting ranking can be used to
prioritise which symbols the strategy actively trades.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backtest import Candle
from .indicators import sma, atr_pct, pct_change


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolScore:
    """Immutable snapshot of a symbol's scoring breakdown.

    Attributes:
        symbol: Trading pair identifier (e.g. ``"BTCUSDT"``).
        trend_score: MA direction + momentum component, range ``[0, 1]``.
        volatility_score: Moderate-volatility preference component, range
            ``[0, 1]``.
        volume_score: Recent-volume strength component, range ``[0, 1]``.
        composite: Weighted average of the three sub-scores.
        eligible: ``True`` when *composite* meets or exceeds the minimum
            threshold configured by the caller.
    """

    symbol: str
    trend_score: float
    volatility_score: float
    volume_score: float
    composite: float
    eligible: bool


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_symbols(
    candles_by_symbol: dict[str, list[Candle]],
    *,
    trend_ma_bars: int = 50,
    momentum_bars: int = 24,
    atr_bars: int = 24,
    volume_bars: int = 20,
    min_composite_score: float = 0.3,
    trend_weight: float = 0.4,
    volatility_weight: float = 0.3,
    volume_weight: float = 0.3,
) -> list[SymbolScore]:
    """Score every symbol on trend, volatility and volume factors.

    Parameters:
        candles_by_symbol: Mapping of symbol name → chronologically ordered
            candle history.  Each :class:`Candle` must expose *close*, *high*,
            *low* and *volume* fields.
        trend_ma_bars: Window for the simple moving average used to gauge
            trend direction.  Default ``50``.
        momentum_bars: Look-back for measuring close-over-close momentum.
            Default ``24``.
        atr_bars: Window for ATR-based volatility calculation.  Default
            ``24``.
        volume_bars: Window for computing recent vs. baseline volume.
            Default ``20``.
        min_composite_score: Composite threshold below which a symbol is
            flagged as ineligible.  Default ``0.3``.
        trend_weight: Weight applied to the trend sub-score.  Default
            ``0.4``.
        volatility_weight: Weight applied to the volatility sub-score.
            Default ``0.3``.
        volume_weight: Weight applied to the volume sub-score.  Default
            ``0.3``.

    Returns:
        A list of :class:`SymbolScore` instances sorted by *composite*
        descending.  Symbols with insufficient history receive zero scores.
    """
    results: list[SymbolScore] = []

    for symbol, candles in candles_by_symbol.items():
        trend = _compute_trend_score(candles, trend_ma_bars, momentum_bars)
        volatility = _compute_volatility_score(candles, atr_bars)
        volume = _compute_volume_score(candles, volume_bars)

        total_weight = trend_weight + volatility_weight + volume_weight
        if total_weight <= 0:
            composite = 0.0
        else:
            composite = (
                trend_weight * trend
                + volatility_weight * volatility
                + volume_weight * volume
            ) / total_weight

        results.append(
            SymbolScore(
                symbol=symbol,
                trend_score=trend,
                volatility_score=volatility,
                volume_score=volume,
                composite=composite,
                eligible=composite >= min_composite_score,
            )
        )

    results.sort(key=lambda s: s.composite, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def select_top_symbols(
    candles_by_symbol: dict[str, list[Candle]],
    **kwargs,
) -> list[str]:
    """Return symbol names that pass the composite eligibility threshold.

    Accepts the same keyword arguments as :func:`score_symbols`.  Only symbols
    whose ``eligible`` flag is ``True`` are returned, still sorted by composite
    score descending.

    Returns:
        A list of symbol name strings.
    """
    scores = score_symbols(candles_by_symbol, **kwargs)
    return [s.symbol for s in scores if s.eligible]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_trend_score(
    candles: list[Candle],
    ma_bars: int,
    momentum_bars: int,
) -> float:
    """Score trend from 0 to 1.

    * ``1.0`` — close above SMA **and** positive momentum.
    * ``0.5`` — only one of the two conditions holds.
    * ``0.0`` — neither condition holds (or insufficient data).
    """
    closes = [c.close for c in candles]
    if not closes:
        return 0.0

    ma_value = sma(closes, ma_bars)
    above_ma = ma_value is not None and closes[-1] > ma_value

    if len(closes) > momentum_bars and momentum_bars > 0:
        mom = pct_change(closes[-(momentum_bars + 1)], closes[-1])
        positive_momentum = mom > 0.0
    else:
        positive_momentum = False

    if above_ma and positive_momentum:
        return 1.0
    if above_ma or positive_momentum:
        return 0.5
    return 0.0


def _compute_volatility_score(
    candles: list[Candle],
    atr_bars: int,
) -> float:
    """Score volatility from 0 to 1 — peaks at moderate values.

    Uses a short-window ATR divided by a longer baseline (2× window).  The
    ratio is centred at ``1.0``; deviations in either direction reduce the
    score: ``score = 1.0 - clamp(|ratio - 1.0|, 0, 1)``.
    """
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]

    short_atr = atr_pct(highs, lows, closes, atr_bars)
    if short_atr is None:
        return 0.0

    long_bars = atr_bars * 2
    long_atr = atr_pct(highs, lows, closes, long_bars)
    if long_atr is None or long_atr <= 0.0:
        return 0.0

    atr_ratio = short_atr / long_atr
    deviation = abs(atr_ratio - 1.0)
    return 1.0 - min(max(deviation, 0.0), 1.0)


def _compute_volume_score(
    candles: list[Candle],
    volume_bars: int,
) -> float:
    """Score recent volume relative to a longer baseline, range [0, 1].

    ``score = min(1.0, recent_avg / baseline_avg)`` where the baseline spans
    twice the ``volume_bars`` window.
    """
    volumes = [c.volume for c in candles]
    if volume_bars <= 0 or len(volumes) < volume_bars:
        return 0.0

    recent_avg = sum(volumes[-volume_bars:]) / volume_bars

    baseline_bars = volume_bars * 2
    if len(volumes) < baseline_bars:
        baseline_avg = sum(volumes) / len(volumes)
    else:
        baseline_avg = sum(volumes[-baseline_bars:]) / baseline_bars

    if baseline_avg <= 0.0:
        return 0.0

    return min(1.0, recent_avg / baseline_avg)
