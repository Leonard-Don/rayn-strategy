"""Symbol probation helpers for grid research."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeLike:
    symbol: str
    pnl: float
    entry_reason: str
    exit_reason: str


@dataclass
class SymbolStats:
    trades: int = 0
    pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    deep_layer_pnl: float = 0.0
    tail_loss: float = 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float("inf") if self.gross_profit > 0 else 0.0
        return self.gross_profit / abs(self.gross_loss)


def symbol_stats(trades: list[TradeLike]) -> dict[str, SymbolStats]:
    stats: dict[str, SymbolStats] = {}
    for trade in trades:
        item = stats.setdefault(trade.symbol, SymbolStats())
        item.trades += 1
        item.pnl += trade.pnl
        if trade.pnl > 0:
            item.gross_profit += trade.pnl
        else:
            item.gross_loss += trade.pnl
        if trade_layers(trade.entry_reason) >= 3:
            item.deep_layer_pnl += trade.pnl
        if trade.exit_reason in {"grid_hard_stop", "circuit_breaker", "time_stop"}:
            item.tail_loss += min(0.0, trade.pnl)
    return stats


def select_eligible_symbols(
    symbols: list[str],
    stats: dict[str, SymbolStats],
    *,
    benchmark_symbol: str,
    min_profit_factor: float,
    min_pnl: float,
    allow_deep_layer_loss: bool,
) -> list[str]:
    selected: list[str] = []
    for symbol in symbols:
        if symbol == benchmark_symbol:
            selected.append(symbol)
            continue
        item = stats.get(symbol)
        if item is None or item.trades == 0:
            continue
        if item.pnl < min_pnl:
            continue
        if item.profit_factor < min_profit_factor:
            continue
        if not allow_deep_layer_loss and item.deep_layer_pnl < 0:
            continue
        selected.append(symbol)
    return selected


def trade_layers(entry_reason: str) -> int:
    marker = "layers="
    if marker not in entry_reason:
        return 1
    raw = entry_reason.split(marker, 1)[1].split("|", 1)[0]
    try:
        return int(raw)
    except ValueError:
        return 1
