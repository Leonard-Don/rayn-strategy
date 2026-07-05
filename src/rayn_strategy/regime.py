"""Multi-dimensional market regime detection.

Combines trend, volatility, and volume analysis to produce a composite
market regime assessment.  The main entry point is :func:`detect_market_regime`,
which returns a frozen :class:`MarketRegime` dataclass.

All heavy lifting is delegated to the lightweight indicator helpers in
``rayn_strategy.indicators`` so this module stays thin and fully testable
without external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

from .indicators import atr_pct, sma


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketRegime:
    """Snapshot of the current market regime.

    Attributes:
        trend: One of ``"bullish"``, ``"neutral"``, or ``"bearish"``.
        volatility: One of ``"low"``, ``"normal"``, ``"high"``, or
            ``"extreme"``.  Set to ``"normal"`` when highs/lows are not
            available and volatility detection is skipped.
        volume_trend: One of ``"dry"``, ``"normal"``, or ``"elevated"``.
            Set to ``"normal"`` when volumes are not available.
        score: Composite score in the range ``[-1.0, +1.0]``.
        allow_entry: Whether new entries should be allowed given the
            current regime.
        position_scale: Suggested position-size multiplier.
            ``1.0`` = normal, ``0.5`` = reduced, ``0.0`` = blocked.
    """

    trend: str
    volatility: str
    volume_trend: str
    score: float
    allow_entry: bool
    position_scale: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_trend(
    closes: list[float],
    short_ma_bars: int,
    long_ma_bars: int,
) -> tuple[str, float]:
    """Return ``(label, component)`` for the trend dimension.

    * ``component`` ranges from roughly ``-1.0`` (strongly bearish) to
      ``+1.0`` (strongly bullish).
    * When there is insufficient data the trend is ``"neutral"`` with a
      component of ``0.0``.
    """
    short_val = sma(closes, short_ma_bars)
    long_val = sma(closes, long_ma_bars)

    if short_val is None or long_val is None or long_val == 0.0:
        return "neutral", 0.0

    # Normalised distance between the two MAs, clamped to [-1, 1].
    raw = (short_val - long_val) / long_val
    component = max(-1.0, min(1.0, raw * 10.0))  # scale up for sensitivity

    if component > 0.15:
        label = "bullish"
    elif component < -0.15:
        label = "bearish"
    else:
        label = "neutral"

    return label, component


def _detect_volatility(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    atr_short_bars: int,
    atr_long_bars: int,
) -> tuple[str, float]:
    """Return ``(label, component)`` for the volatility dimension.

    Compares short-term ATR% to long-term ATR%.  A ratio > 1 means
    volatility is expanding; < 1 means it is contracting.

    * ``component`` ranges from ``-1.0`` (extreme vol) to ``+1.0`` (very
      calm).
    """
    short_atr = atr_pct(highs, lows, closes, atr_short_bars)
    long_atr = atr_pct(highs, lows, closes, atr_long_bars)

    if short_atr is None or long_atr is None or long_atr == 0.0:
        return "normal", 0.0

    ratio = short_atr / long_atr

    if ratio > 2.0:
        label = "extreme"
        component = -1.0
    elif ratio > 1.3:
        label = "high"
        component = -0.5
    elif ratio < 0.7:
        label = "low"
        component = 0.5
    else:
        label = "normal"
        component = 0.0

    return label, component


def _detect_volume_trend(
    volumes: list[float],
    volume_short_bars: int,
    volume_long_bars: int,
) -> tuple[str, float]:
    """Return ``(label, component)`` for the volume dimension.

    Compares recent average volume to baseline average volume.

    * ``component`` ranges from ``-1.0`` (very dry) to ``+1.0`` (elevated).
    """
    short_vol = sma(volumes, volume_short_bars)
    long_vol = sma(volumes, volume_long_bars)

    if short_vol is None or long_vol is None or long_vol == 0.0:
        return "normal", 0.0

    ratio = short_vol / long_vol

    if ratio > 1.5:
        label = "elevated"
        component = 0.5
    elif ratio < 0.5:
        label = "dry"
        component = -0.5
    else:
        label = "normal"
        component = 0.0

    return label, component


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_market_regime(
    closes: list[float],
    volumes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    *,
    short_ma_bars: int = 50,
    long_ma_bars: int = 200,
    atr_short_bars: int = 24,
    atr_long_bars: int = 120,
    volume_short_bars: int = 20,
    volume_long_bars: int = 100,
    bearish_threshold: float = -0.3,
    cautious_threshold: float = 0.0,
    cautious_position_scale: float = 0.5,
) -> MarketRegime:
    """Detect the current market regime from price and volume data.

    The function combines three independent dimensions — **trend**,
    **volatility**, and **volume** — into a single composite score and
    derives entry-permission / position-scaling rules from it.

    Parameters:
        closes: List of close prices, oldest first.
        volumes: Optional list of volumes, oldest first.  When ``None``,
            the volume dimension defaults to ``"normal"`` with a 0.0
            contribution.
        highs: Optional list of high prices.  When ``None``, volatility
            detection is skipped (defaults to ``"normal"``).
        lows: Optional list of low prices.  When ``None``, volatility
            detection is skipped (defaults to ``"normal"``).
        short_ma_bars: Look-back for the short (fast) moving average used
            in trend detection.
        long_ma_bars: Look-back for the long (slow) moving average used
            in trend detection.
        atr_short_bars: Look-back for the short ATR window.
        atr_long_bars: Look-back for the long ATR window.
        volume_short_bars: Look-back for recent volume average.
        volume_long_bars: Look-back for baseline volume average.
        bearish_threshold: Score at or below which entries are **blocked**
            (``allow_entry=False, position_scale=0.0``).
        cautious_threshold: Score between *bearish_threshold* and this
            value triggers a reduced position scale.
        cautious_position_scale: Position-scale multiplier applied when
            the score falls in the cautious zone.

    Returns:
        A frozen :class:`MarketRegime` instance.
    """
    # --- Trend ---
    trend_label, trend_component = _detect_trend(
        closes, short_ma_bars, long_ma_bars
    )

    # --- Volatility ---
    if highs is not None and lows is not None:
        vol_label, vol_component = _detect_volatility(
            highs, lows, closes, atr_short_bars, atr_long_bars
        )
    else:
        vol_label, vol_component = "normal", 0.0

    # --- Volume ---
    if volumes is not None:
        volume_label, volume_component = _detect_volume_trend(
            volumes, volume_short_bars, volume_long_bars
        )
    else:
        volume_label, volume_component = "normal", 0.0

    # --- Composite score ---
    score = (
        trend_component * 0.5
        + vol_component * 0.25
        + volume_component * 0.25
    )
    score = max(-1.0, min(1.0, score))

    # --- Entry / position-scale rules ---
    if score < bearish_threshold:
        allow_entry = False
        position_scale = 0.0
    elif score < cautious_threshold:
        allow_entry = True
        position_scale = cautious_position_scale
    else:
        allow_entry = True
        position_scale = 1.0

    return MarketRegime(
        trend=trend_label,
        volatility=vol_label,
        volume_trend=volume_label,
        score=round(score, 4),
        allow_entry=allow_entry,
        position_scale=position_scale,
    )
