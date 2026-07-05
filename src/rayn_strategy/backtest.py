"""Minimal portfolio backtester for the Rayn-like research strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
from datetime import datetime, timezone

from .config import RaynConfig
from .risk import Position, RiskManager
from .signals import entry_signal, exit_signal, market_regime_filter_reason


@dataclass(frozen=True)
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Trade:
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    notional: float
    pnl: float
    entry_reason: str
    exit_reason: str


@dataclass(frozen=True)
class EquityPoint:
    timestamp: str
    cash_equity: float
    marked_equity: float
    open_notional: float
    open_margin: float
    floating_pnl: float = 0.0
    max_layers: int = 0
    longest_hold_bars: int = 0
    worst_symbol_unrealized_pct: float = 0.0
    min_liquidation_buffer_pct: float | None = None


@dataclass
class BacktestResult:
    initial_equity: float
    final_equity: float
    peak_equity: float
    max_drawdown_pct: float
    trades: list[Trade] = field(default_factory=list)
    skipped_entries: dict[str, int] = field(default_factory=dict)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    halted: bool = False

    def to_text(self) -> str:
        total_pnl = self.final_equity - self.initial_equity
        roi = total_pnl / self.initial_equity if self.initial_equity else 0.0
        wins = [trade for trade in self.trades if trade.pnl > 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0

        lines = [
            "Rayn-like Backtest Result",
            f"initial_equity: {self.initial_equity:.2f}",
            f"final_equity:   {self.final_equity:.2f}",
            f"total_pnl:      {total_pnl:.2f}",
            f"roi:            {roi:.2%}",
            f"trades:         {len(self.trades)}",
            f"win_rate:       {win_rate:.2%}",
            f"max_drawdown:   {self.max_drawdown_pct:.2%}",
            f"halted:         {self.halted}",
        ]
        if self.trades:
            lines.extend(self._grouped_lines("by_symbol", "symbol"))
            lines.extend(self._grouped_lines("by_entry", "entry_reason"))
            lines.extend(self._grouped_lines("by_exit", "exit_reason"))
        if self.skipped_entries:
            lines.append("skipped_entries:")
            for reason, count in sorted(self.skipped_entries.items()):
                lines.append(f"  {reason}: {count}")
        if self.equity_curve:
            lines.extend(self._hidden_risk_lines())
        return "\n".join(lines)

    def _hidden_risk_lines(self) -> list[str]:
        max_notional_pct = max(
            (
                point.open_notional / point.marked_equity
                for point in self.equity_curve
                if point.marked_equity > 0
            ),
            default=0.0,
        )
        max_margin_pct = max(
            (
                point.open_margin / point.marked_equity
                for point in self.equity_curve
                if point.marked_equity > 0
            ),
            default=0.0,
        )
        max_floating_loss = min((point.floating_pnl for point in self.equity_curve), default=0.0)
        max_floating_loss_pct = -max_floating_loss / self.initial_equity if self.initial_equity else 0.0
        min_liquidation_buffer = min(
            (
                point.min_liquidation_buffer_pct
                for point in self.equity_curve
                if point.min_liquidation_buffer_pct is not None
            ),
            default=None,
        )
        worst_symbol_unrealized_pct = min(
            (point.worst_symbol_unrealized_pct for point in self.equity_curve),
            default=0.0,
        )
        lines = [
            "hidden_risk:",
            f"  max_floating_loss: {max_floating_loss:.2f} ({max_floating_loss_pct:.2%} of initial)",
            f"  max_notional_to_equity: {max_notional_pct:.2f}x",
            f"  max_margin_to_equity: {max_margin_pct:.2f}x",
            f"  max_layers: {max(point.max_layers for point in self.equity_curve)}",
            f"  longest_hold_bars: {max(point.longest_hold_bars for point in self.equity_curve)}",
            f"  worst_symbol_unrealized: {worst_symbol_unrealized_pct:.2%}",
        ]
        if min_liquidation_buffer is None:
            lines.append("  min_liquidation_buffer: n/a")
        else:
            lines.append(f"  min_liquidation_buffer: {min_liquidation_buffer:.2%}")
        return lines

    def _grouped_lines(self, title: str, attribute: str) -> list[str]:
        grouped: dict[str, dict[str, float]] = {}
        for trade in self.trades:
            key = str(getattr(trade, attribute))
            stats = grouped.setdefault(key, {"trades": 0.0, "pnl": 0.0, "wins": 0.0})
            stats["trades"] += 1
            stats["pnl"] += trade.pnl
            if trade.pnl > 0:
                stats["wins"] += 1

        lines = [f"{title}:"]
        for key, stats in sorted(grouped.items()):
            trades = int(stats["trades"])
            win_rate = stats["wins"] / trades if trades else 0.0
            lines.append(
                f"  {key}: trades={trades} pnl={stats['pnl']:.2f} win_rate={win_rate:.2%}"
            )
        return lines


class BacktestEngine:
    def __init__(
        self,
        config: RaynConfig,
        data_dir: Path,
        symbols: list[str],
        entry_start: str | None = None,
        entry_end: str | None = None,
        run_end: str | None = None,
    ) -> None:
        self.config = config
        self.data_dir = data_dir
        self.symbols = symbols
        self.risk = RiskManager(config)
        self.entry_start = normalize_timestamp(entry_start) if entry_start else None
        self.entry_end = normalize_timestamp(entry_end) if entry_end else None
        self.run_end = normalize_timestamp(run_end) if run_end else None

    def run(self) -> BacktestResult:
        data = {symbol: load_candles(self.data_dir / f"{symbol}.csv") for symbol in self.symbols}
        benchmark = self.config.circuit_breaker.benchmark_symbol
        if benchmark not in data:
            raise ValueError(f"benchmark symbol {benchmark} must be included in symbols")

        timestamps = sorted(set.intersection(*(set(row.timestamp for row in rows) for rows in data.values())))
        if not timestamps:
            raise ValueError("no overlapping timestamps across symbol data")

        by_time = {
            symbol: {row.timestamp: row for row in rows}
            for symbol, rows in data.items()
        }
        closes: dict[str, list[float]] = {symbol: [] for symbol in self.symbols}
        highs: dict[str, list[float]] = {symbol: [] for symbol in self.symbols}
        lows: dict[str, list[float]] = {symbol: [] for symbol in self.symbols}
        volumes: dict[str, list[float]] = {symbol: [] for symbol in self.symbols}
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        skipped: dict[str, int] = {}
        cash_equity = self.config.portfolio.initial_equity
        risk_state = self.risk.initial_state()
        max_drawdown = 0.0
        last_timestamp = timestamps[0]

        for timestamp in timestamps:
            if self.run_end and timestamp >= self.run_end:
                break
            last_timestamp = timestamp
            for symbol in self.symbols:
                candle = by_time[symbol][timestamp]
                closes[symbol].append(candle.close)
                highs[symbol].append(candle.high)
                lows[symbol].append(candle.low)
                volumes[symbol].append(candle.volume)

            mark_prices = {symbol: by_time[symbol][timestamp].close for symbol in self.symbols}
            marked_equity = self._marked_equity(cash_equity, positions, mark_prices)
            self.risk.mark_day(risk_state, timestamp, marked_equity)
            circuit_active = self.risk.update_circuit_breaker(risk_state, closes[benchmark])

            for symbol, position in list(positions.items()):
                candle = by_time[symbol][timestamp]
                price = candle.close
                position.bars_held += 1
                signal = exit_signal(
                    entry_price=position.entry_price,
                    current_price=price,
                    high_price=candle.high,
                    low_price=candle.low,
                    bars_held=position.bars_held,
                    circuit_breaker_active=circuit_active
                    and self.config.circuit_breaker.close_positions_on_trigger,
                    config=self.config.strategy,
                )
                if signal.action == "sell":
                    exit_price = self._exit_price(position.entry_price, price, signal.reason)
                    trade, cash_equity = self._close_position(
                        position,
                        timestamp,
                        exit_price,
                        signal.reason,
                        cash_equity,
                    )
                    trades.append(trade)
                    self.risk.record_trade_result(risk_state, trade.pnl)
                    del positions[symbol]

            marked_equity = self._marked_equity(cash_equity, positions, mark_prices)
            self.risk.update_account_limits(risk_state, marked_equity)
            max_drawdown = max(max_drawdown, current_drawdown(marked_equity, risk_state.peak_equity))
            if risk_state.halted:
                break
            if circuit_active:
                continue
            if not self._entry_window_open(timestamp):
                continue

            regime_block = market_regime_filter_reason(closes[benchmark], self.config.strategy)
            if regime_block:
                skipped[regime_block] = skipped.get(regime_block, 0) + len(
                    [symbol for symbol in self.symbols if symbol not in positions]
                )
                continue

            loss_cooldown_active = self.risk.loss_cooldown_active(risk_state)
            for symbol in self.symbols:
                signal = entry_signal(
                    closes[symbol],
                    self.config.strategy,
                    highs=highs[symbol],
                    lows=lows[symbol],
                    volumes=volumes[symbol],
                    timestamp=timestamp,
                )
                if signal.action != "buy":
                    if signal.reason != "no_signal":
                        skipped[signal.reason] = skipped.get(signal.reason, 0) + 1
                    continue
                if loss_cooldown_active:
                    skipped["loss_cooldown"] = skipped.get("loss_cooldown", 0) + 1
                    continue

                marked_equity = self._marked_equity(cash_equity, positions, mark_prices)
                allowed, reason = self.risk.can_open(symbol, marked_equity, positions)
                if not allowed:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue

                price = mark_prices[symbol]
                position = self.risk.build_position(
                    symbol,
                    timestamp,
                    price,
                    marked_equity,
                    signal.reason,
                )
                if position.notional <= 0:
                    skipped["zero_notional"] = skipped.get("zero_notional", 0) + 1
                    continue

                entry_cost = trading_cost(position.notional, self.config)
                cash_equity -= entry_cost
                positions[symbol] = position
            if loss_cooldown_active:
                self.risk.advance_loss_cooldown(risk_state)

        final_prices = {symbol: by_time[symbol][last_timestamp].close for symbol in self.symbols}
        for symbol, position in list(positions.items()):
            trade, cash_equity = self._close_position(
                position,
                last_timestamp,
                final_prices[symbol],
                "end_of_test",
                cash_equity,
            )
            trades.append(trade)
            self.risk.record_trade_result(risk_state, trade.pnl)
            del positions[symbol]
        risk_state.peak_equity = max(risk_state.peak_equity, cash_equity)
        max_drawdown = max(max_drawdown, current_drawdown(cash_equity, risk_state.peak_equity))

        return BacktestResult(
            initial_equity=self.config.portfolio.initial_equity,
            final_equity=cash_equity,
            peak_equity=risk_state.peak_equity,
            max_drawdown_pct=max_drawdown,
            trades=trades,
            skipped_entries=skipped,
            halted=risk_state.halted,
        )

    def _marked_equity(
        self,
        cash_equity: float,
        positions: dict[str, Position],
        prices: dict[str, float],
    ) -> float:
        unrealized = 0.0
        for symbol, position in positions.items():
            unrealized += position.notional * ((prices[symbol] / position.entry_price) - 1.0)
        return cash_equity + unrealized

    def _entry_window_open(self, timestamp: str) -> bool:
        if self.entry_start and timestamp < self.entry_start:
            return False
        if self.entry_end and timestamp >= self.entry_end:
            return False
        return True

    def _exit_price(self, entry_price: float, close_price: float, reason: str) -> float:
        if reason == "stop_loss":
            return entry_price * (1.0 - self.config.strategy.stop_loss_pct)
        if reason == "take_profit":
            return entry_price * (1.0 + self.config.strategy.take_profit_pct)
        return close_price

    def _close_position(
        self,
        position: Position,
        timestamp: str,
        price: float,
        reason: str,
        equity: float,
    ) -> tuple[Trade, float]:
        gross_pnl = position.notional * ((price / position.entry_price) - 1.0)
        one_side_cost = trading_cost(position.notional, self.config)
        pnl = gross_pnl - (2 * one_side_cost)
        new_equity = equity + gross_pnl - one_side_cost
        return (
            Trade(
                symbol=position.symbol,
                entry_time=position.entry_time,
                exit_time=timestamp,
                entry_price=position.entry_price,
                exit_price=price,
                notional=position.notional,
                pnl=pnl,
                entry_reason=position.entry_reason,
                exit_reason=reason,
            ),
            new_equity,
        )


def trading_cost(notional: float, config: RaynConfig) -> float:
    bps = config.portfolio.taker_fee_bps + config.portfolio.slippage_bps
    return notional * bps / 10000.0


def current_drawdown(equity: float, peak_equity: float) -> float:
    if peak_equity <= 0:
        return 0.0
    return max(0.0, 1.0 - (equity / peak_equity))


def load_candles(path: Path) -> list[Candle]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[Candle] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                Candle(
                    timestamp=normalize_timestamp(raw["timestamp"]),
                    open=float(raw["open"]),
                    high=float(raw["high"]),
                    low=float(raw["low"]),
                    close=float(raw["close"]),
                    volume=float(raw["volume"]),
                )
            )
    rows.sort(key=lambda row: row.timestamp)
    return rows


def normalize_timestamp(value: str) -> str:
    item = value.strip()
    if item.isdigit():
        number = int(item)
        if number > 10_000_000_000:
            number = number // 1000
        return datetime.fromtimestamp(number, tz=timezone.utc).isoformat()
    if item.endswith("Z"):
        item = item[:-1] + "+00:00"
    parsed = datetime.fromisoformat(item)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()
