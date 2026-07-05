"""Pure exchange-filter helpers for offline feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN

from .config import RaynConfig


@dataclass(frozen=True)
class SymbolFilter:
    min_qty: float = 0.0
    step_size: float = 0.0
    min_notional: float = 0.0


@dataclass(frozen=True)
class SymbolFeasibility:
    symbol: str
    target_notional: float
    target_quantity: float
    min_executable_quantity: float
    min_executable_notional: float
    distortion: float
    feasible: bool
    reason: str


def normalize_quantity(quantity: float, step_size: float) -> float:
    if step_size <= 0:
        return quantity
    quantity_dec = Decimal(str(quantity))
    step_dec = Decimal(str(step_size))
    steps = (quantity_dec / step_dec).to_integral_value(rounding=ROUND_DOWN)
    return float(steps * step_dec)


def normalize_quantity_up(quantity: float, step_size: float) -> float:
    if step_size <= 0:
        return quantity
    quantity_dec = Decimal(str(quantity))
    step_dec = Decimal(str(step_size))
    steps = (quantity_dec / step_dec).to_integral_value(rounding=ROUND_CEILING)
    return float(steps * step_dec)


def min_executable_quantity(price: float, symbol_filter: SymbolFilter) -> float:
    if price <= 0:
        return 0.0
    required_quantity = max(
        symbol_filter.min_qty,
        symbol_filter.min_notional / price if symbol_filter.min_notional > 0 else 0.0,
    )
    return normalize_quantity_up(required_quantity, symbol_filter.step_size)


def symbol_feasibility(
    *,
    symbol: str,
    target_notional: float,
    price: float,
    symbol_filter: SymbolFilter,
    max_min_order_distortion: float = 1.5,
) -> SymbolFeasibility:
    target_quantity = target_notional / price if price > 0 else 0.0
    min_quantity = min_executable_quantity(price, symbol_filter)
    min_notional = min_quantity * price
    distortion = min_notional / target_notional if target_notional > 0 else float("inf")
    effective_max_distortion = max(1.0, max_min_order_distortion)
    reason = "ok"
    feasible = True
    if price <= 0:
        feasible = False
        reason = "missing_price"
    elif target_notional <= 0:
        feasible = False
        reason = "target_notional_not_positive"
    elif min_notional > target_notional:
        reason = "min_order_adjusted"
        if distortion > effective_max_distortion:
            feasible = False
            reason = f"min_order_distortion_{distortion:.2f}x"
    return SymbolFeasibility(
        symbol=symbol,
        target_notional=target_notional,
        target_quantity=target_quantity,
        min_executable_quantity=min_quantity,
        min_executable_notional=min_notional,
        distortion=distortion,
        feasible=feasible,
        reason=reason,
    )


def build_symbol_feasibility_report(
    *,
    symbols: list[str],
    config: RaynConfig,
    equity: float,
    prices: dict[str, float],
    symbol_filters: dict[str, SymbolFilter],
    max_min_order_distortion: float = 1.5,
) -> list[SymbolFeasibility]:
    total_notional_cap = equity * config.risk.max_total_notional_pct
    target_notional = min(entry_notional_for_equity(config, equity), total_notional_cap)
    report: list[SymbolFeasibility] = []
    for symbol in symbols:
        price = prices.get(symbol, 0.0)
        symbol_filter = symbol_filters.get(symbol, SymbolFilter())
        report.append(
            symbol_feasibility(
                symbol=symbol,
                target_notional=target_notional,
                price=price,
                symbol_filter=symbol_filter,
                max_min_order_distortion=max_min_order_distortion,
            )
        )
    return report


def entry_notional_for_equity(config: RaynConfig, equity: float) -> float:
    if equity <= 0 or config.strategy.stop_loss_pct <= 0:
        return 0.0
    raw = equity * config.risk.risk_per_trade_pct / config.strategy.stop_loss_pct
    cap = equity * config.risk.max_position_notional_pct
    return max(0.0, min(raw, cap))


def symbol_filter_refusal(
    symbol: str,
    quantity: float,
    notional: float,
    symbol_filter: SymbolFilter,
) -> str:
    if symbol_filter.min_qty > 0 and quantity < symbol_filter.min_qty:
        return f"{symbol}:quantity_below_min_qty"
    if symbol_filter.min_notional > 0 and notional < symbol_filter.min_notional:
        return f"{symbol}:notional_below_min_notional"
    return ""


def symbol_filters_from_exchange_info(
    exchange_info: dict[str, object],
    symbols: list[str],
) -> dict[str, SymbolFilter]:
    wanted = set(symbols)
    parsed: dict[str, SymbolFilter] = {}
    for raw_symbol in exchange_info.get("symbols", []):
        if not isinstance(raw_symbol, dict):
            continue
        symbol = str(raw_symbol.get("symbol", ""))
        if symbol not in wanted:
            continue
        filters = {
            str(item.get("filterType")): item
            for item in raw_symbol.get("filters", [])
            if isinstance(item, dict)
        }
        market_lot_filter = filters.get("MARKET_LOT_SIZE") or {}
        lot_filter = filters.get("LOT_SIZE") or {}
        if _positive_float(market_lot_filter.get("minQty")) and _positive_float(
            market_lot_filter.get("stepSize")
        ):
            lot_filter = market_lot_filter
        min_notional_filter = filters.get("MIN_NOTIONAL") or {}
        parsed[symbol] = SymbolFilter(
            min_qty=float(lot_filter.get("minQty", 0.0)),
            step_size=float(lot_filter.get("stepSize", 0.0)),
            min_notional=float(
                min_notional_filter.get("notional", min_notional_filter.get("minNotional", 0.0))
            ),
        )
    return parsed


def _positive_float(value: object) -> bool:
    try:
        return float(value or 0.0) > 0
    except (TypeError, ValueError):
        return False
