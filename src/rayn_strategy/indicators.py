"""Small indicator helpers implemented without external dependencies."""

from __future__ import annotations


def sma(values: list[float], length: int) -> float | None:
    if length <= 0:
        raise ValueError("length must be positive")
    if len(values) < length:
        return None
    return sum(values[-length:]) / length


def rolling_high(values: list[float], length: int) -> float | None:
    if length <= 0:
        raise ValueError("length must be positive")
    if len(values) < length:
        return None
    return max(values[-length:])


def average_range_pct(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    length: int,
) -> float | None:
    if length <= 0:
        raise ValueError("length must be positive")
    if len(highs) < length or len(lows) < length or len(closes) < length:
        return None

    total = 0.0
    for high, low, close in zip(highs[-length:], lows[-length:], closes[-length:]):
        if close <= 0:
            continue
        total += (high - low) / close
    return total / length


def atr_pct(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    length: int,
) -> float | None:
    if length <= 0:
        raise ValueError("length must be positive")
    if len(highs) < length or len(lows) < length or len(closes) < length:
        return None
    total = 0.0
    for high, low, close in zip(highs[-length:], lows[-length:], closes[-length:]):
        if close <= 0:
            continue
        total += (high - low) / close
    return total / length


def lower_wick_pct(*, open_price: float, low: float, close: float) -> float:
    if close <= 0:
        return 0.0
    lower_body = min(open_price, close)
    return max(0.0, lower_body - low) / close


def volume_spike(volumes: list[float], lookback: int, multiplier: float) -> bool:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    if len(volumes) <= lookback:
        return False
    baseline = volumes[-(lookback + 1) : -1]
    average = sum(baseline) / lookback
    return average > 0 and volumes[-1] >= average * multiplier


def rsi(values: list[float], length: int) -> float | None:
    if length <= 0:
        raise ValueError("length must be positive")
    if len(values) <= length:
        return None

    gains = 0.0
    losses = 0.0
    window = values[-(length + 1) :]
    for prev, curr in zip(window, window[1:]):
        change = curr - prev
        if change >= 0:
            gains += change
        else:
            losses += abs(change)

    avg_gain = gains / length
    avg_loss = losses / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def pct_change(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return (new / old) - 1.0
