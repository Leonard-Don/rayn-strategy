"""Bounded grid research engine for Rayn-style strategy experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .backtest import (
    BacktestResult,
    Candle,
    EquityPoint,
    Trade,
    current_drawdown,
    load_candles,
    normalize_timestamp,
    trading_cost,
)
from .config import RaynConfig
from .exchange_filters import (
    SymbolFilter,
    normalize_quantity,
    symbol_feasibility,
)
from .probation import TradeLike, trade_layers
from .risk import RiskManager
from .signals import entry_signal, market_regime_filter_reason
from .indicators import atr_pct, lower_wick_pct, volume_spike


MAINTENANCE_MARGIN_PCT = 0.005


@dataclass(frozen=True)
class BoundedGridConfig:
    max_rescue_layers: int
    layer_step_pct: float
    layer_multiplier: float
    take_profit_pct: float
    hard_stop_pct: float
    max_symbol_notional_pct: float
    max_total_notional_pct: float
    min_liquidation_buffer_pct: float = 0.0
    liquidation_exit_buffer_pct: float = 0.0
    layer_trim_buffer_pct: float = 0.0
    layer_trim_min_layers: int = 0
    layer_trim_count: int = 1
    deep_take_profit_layer: int = 0
    deep_take_profit_pct: float = 0.0
    symbol_cooldown_bars: int = 0
    symbol_cooldown_position_scale: float = 0.0
    symbol_cooldown_min_loss_layers: int = 0
    symbol_cooldown_exit_reasons: list[str] = field(default_factory=list)
    atr_layer_lookback_bars: int = 0
    atr_layer_step_multiplier: float = 0.0
    volume_layer_lookback_bars: int = 0
    volume_layer_multiplier: float = 0.0
    min_layer_lower_wick_pct: float = 0.0
    key_layer_multiplier: float = 1.0
    dynamic_take_profit_atr_lookback_bars: int = 0
    dynamic_take_profit_atr_multiplier: float = 0.0
    dynamic_take_profit_min_pct: float = 0.0
    dynamic_take_profit_max_pct: float = 0.0
    layer_step_expansion_rate: float = 0.0  # Per-layer spacing increase (0=disabled, 0.15=+15%/layer)


@dataclass(frozen=True)
class GridLayer:
    timestamp: str
    price: float
    notional: float
    margin: float


@dataclass
class GridPosition:
    symbol: str
    entry_time: str
    entry_reason: str
    layers: list[GridLayer] = field(default_factory=list)
    bars_held: int = 0

    @property
    def notional(self) -> float:
        return sum(layer.notional for layer in self.layers)

    @property
    def margin(self) -> float:
        return sum(layer.margin for layer in self.layers)

    @property
    def average_entry(self) -> float:
        if self.notional <= 0:
            return 0.0
        weighted = sum(layer.price * layer.notional for layer in self.layers)
        return weighted / self.notional

    @property
    def rescue_layers(self) -> int:
        return max(0, len(self.layers) - 1)

    @property
    def last_layer_price(self) -> float:
        return self.layers[-1].price


class BoundedGridEngine:
    def __init__(
        self,
        config: RaynConfig,
        grid_config: BoundedGridConfig,
        data_dir: Path,
        symbols: list[str],
        entry_start: str | None = None,
        entry_end: str | None = None,
        run_end: str | None = None,
        preloaded_data: dict[str, list[Candle]] | None = None,
        symbol_filters: dict[str, SymbolFilter] | None = None,
        max_min_order_distortion: float = 1.5,
        intrabar_liquidation: bool = False,
        quote_haircut_pct: float = 0.0,
        quote_haircut_time: str | None = None,
    ) -> None:
        self.config = config
        self.grid_config = grid_config
        self.data_dir = data_dir
        self.symbols = symbols
        self.risk = RiskManager(config)
        self.entry_start = normalize_timestamp(entry_start) if entry_start else None
        self.entry_end = normalize_timestamp(entry_end) if entry_end else None
        self.run_end = normalize_timestamp(run_end) if run_end else None
        self.preloaded_data = preloaded_data
        self.symbol_filters = symbol_filters or {}
        self.max_min_order_distortion = max_min_order_distortion
        # Honest tail modeling is opt-in; default False preserves the legacy
        # close-based research model and existing tests.
        self.intrabar_liquidation = intrabar_liquidation
        # Quote-asset (e.g. USDT) depeg as a one-time collateral haircut: the USDT
        # cash balance loses value, shrinking the account buffer. Open
        # USDT-margined position PnL is unchanged (it is already in USDT).
        self.quote_haircut_pct = quote_haircut_pct
        self.quote_haircut_time = normalize_timestamp(quote_haircut_time) if quote_haircut_time else None

    def run(self) -> BacktestResult:
        if self.preloaded_data is None:
            data = {symbol: load_candles(self.data_dir / f"{symbol}.csv") for symbol in self.symbols}
        else:
            data = {symbol: self.preloaded_data[symbol] for symbol in self.symbols}
        benchmark = self.config.circuit_breaker.benchmark_symbol
        if benchmark not in data:
            raise ValueError(f"benchmark symbol {benchmark} must be included in symbols")

        timestamps = sorted(set.intersection(*(set(row.timestamp for row in rows) for rows in data.values())))
        if not timestamps:
            raise ValueError("no overlapping timestamps across symbol data")

        by_time = {symbol: {row.timestamp: row for row in rows} for symbol, rows in data.items()}
        closes: dict[str, list[float]] = {symbol: [] for symbol in self.symbols}
        highs: dict[str, list[float]] = {symbol: [] for symbol in self.symbols}
        lows: dict[str, list[float]] = {symbol: [] for symbol in self.symbols}
        volumes: dict[str, list[float]] = {symbol: [] for symbol in self.symbols}
        candles_seen: dict[str, list[Candle]] = {symbol: [] for symbol in self.symbols}
        positions: dict[str, GridPosition] = {}
        trades: list[Trade] = []
        equity_curve: list[EquityPoint] = []
        skipped: dict[str, int] = {}
        symbol_cooldowns: dict[str, int] = {}
        cash_equity = self.config.portfolio.initial_equity
        risk_state = self.risk.initial_state()
        max_drawdown = 0.0
        last_timestamp = timestamps[0]
        quote_haircut_applied = False

        for bar_index, timestamp in enumerate(timestamps):
            if self.run_end and timestamp >= self.run_end:
                break
            last_timestamp = timestamp
            if (
                self.quote_haircut_time is not None
                and not quote_haircut_applied
                and timestamp >= self.quote_haircut_time
            ):
                cash_equity *= (1.0 - self.quote_haircut_pct)
                quote_haircut_applied = True
            for symbol in self.symbols:
                candle = by_time[symbol][timestamp]
                closes[symbol].append(candle.close)
                highs[symbol].append(candle.high)
                lows[symbol].append(candle.low)
                volumes[symbol].append(candle.volume)
                candles_seen[symbol].append(candle)

            mark_prices = {symbol: by_time[symbol][timestamp].close for symbol in self.symbols}
            marked_equity = self._marked_equity(cash_equity, positions, mark_prices)
            self.risk.mark_day(risk_state, timestamp, marked_equity)
            circuit_active = self.risk.update_circuit_breaker(risk_state, closes[benchmark])

            for symbol, position in list(positions.items()):
                position.bars_held += 1
                candle = by_time[symbol][timestamp]
                recent_atr = recent_atr_pct_for_take_profit(candles_seen[symbol], self.grid_config)
                reason = grid_exit_reason(
                    position,
                    candle,
                    circuit_active,
                    self.config,
                    self.grid_config,
                    recent_atr_pct=recent_atr,
                    intrabar_liquidation=self.intrabar_liquidation,
                )
                if reason:
                    exit_price = grid_exit_price(
                        position,
                        candle.close,
                        reason,
                        self.grid_config,
                        recent_atr_pct=recent_atr,
                        leverage=self.config.risk.leverage,
                    )
                    trade, cash_equity = self._close_position(position, timestamp, exit_price, reason, cash_equity)
                    trades.append(trade)
                    self.risk.record_trade_result(risk_state, trade.pnl)
                    self._maybe_cooldown_symbol(trade, bar_index, symbol_cooldowns)
                    del positions[symbol]

            for symbol, position in list(positions.items()):
                candle = by_time[symbol][timestamp]
                if not should_trim_grid_layers(
                    position,
                    candle.close,
                    self.grid_config,
                    self.config.risk.leverage,
                ):
                    continue
                trim_trade, cash_equity = self._trim_position_layers(
                    position,
                    timestamp,
                    candle.close,
                    "layer_trim_exit",
                    cash_equity,
                )
                trades.append(trim_trade)
                self.risk.record_trade_result(risk_state, trim_trade.pnl)
                self._maybe_cooldown_symbol(trim_trade, bar_index, symbol_cooldowns)
                if not position.layers:
                    del positions[symbol]

            marked_equity = self._marked_equity(cash_equity, positions, mark_prices)
            self.risk.update_account_limits(risk_state, marked_equity)
            max_drawdown = max(max_drawdown, current_drawdown(marked_equity, risk_state.peak_equity))
            if risk_state.halted:
                equity_curve.append(
                    self._equity_point(timestamp, cash_equity, marked_equity, positions, mark_prices)
                )
                break

            if not circuit_active:
                cash_equity = self._maybe_add_layers(
                    timestamp,
                    by_time,
                    positions,
                    mark_prices,
                    cash_equity,
                    marked_equity,
                    skipped,
                    candles_seen,
                )

            if circuit_active or not self._entry_window_open(timestamp):
                marked_equity = self._marked_equity(cash_equity, positions, mark_prices)
                risk_state.peak_equity = max(risk_state.peak_equity, marked_equity)
                max_drawdown = max(max_drawdown, current_drawdown(marked_equity, risk_state.peak_equity))
                equity_curve.append(
                    self._equity_point(timestamp, cash_equity, marked_equity, positions, mark_prices)
                )
                continue

            regime_block = market_regime_filter_reason(closes[benchmark], self.config.strategy)
            if regime_block:
                skipped[regime_block] = skipped.get(regime_block, 0) + len(
                    [symbol for symbol in self.symbols if symbol not in positions]
                )
                marked_equity = self._marked_equity(cash_equity, positions, mark_prices)
                risk_state.peak_equity = max(risk_state.peak_equity, marked_equity)
                max_drawdown = max(max_drawdown, current_drawdown(marked_equity, risk_state.peak_equity))
                equity_curve.append(
                    self._equity_point(timestamp, cash_equity, marked_equity, positions, mark_prices)
                )
                continue

            loss_cooldown_active = self.risk.loss_cooldown_active(risk_state)
            for symbol in self.symbols:
                if symbol in positions:
                    continue
                if symbol_cooldowns.get(symbol, -1) > bar_index:
                    if self.grid_config.symbol_cooldown_position_scale <= 0:
                        skipped["symbol_cooldown"] = skipped.get("symbol_cooldown", 0) + 1
                        continue
                    skipped["symbol_probation_sizing"] = skipped.get("symbol_probation_sizing", 0) + 1
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
                base_notional = self.risk.position_notional(marked_equity)
                if symbol_cooldowns.get(symbol, -1) > bar_index:
                    base_notional *= self.grid_config.symbol_cooldown_position_scale
                price = mark_prices[symbol]
                base_notional = self._executable_notional(
                    symbol,
                    base_notional,
                    price,
                    skipped,
                )
                if base_notional is None:
                    continue
                if not self._can_open_grid(symbol, marked_equity, positions, base_notional):
                    skipped["grid_exposure_cap"] = skipped.get("grid_exposure_cap", 0) + 1
                    continue

                if base_notional <= 0:
                    skipped["zero_notional"] = skipped.get("zero_notional", 0) + 1
                    continue
                layer = GridLayer(timestamp, price, base_notional, base_notional / self.config.risk.leverage)
                cash_equity -= trading_cost(base_notional, self.config)
                positions[symbol] = GridPosition(
                    symbol=symbol,
                    entry_time=timestamp,
                    entry_reason=signal.reason,
                    layers=[layer],
                )
            if loss_cooldown_active:
                self.risk.advance_loss_cooldown(risk_state)

            marked_equity = self._marked_equity(cash_equity, positions, mark_prices)
            risk_state.peak_equity = max(risk_state.peak_equity, marked_equity)
            max_drawdown = max(max_drawdown, current_drawdown(marked_equity, risk_state.peak_equity))
            equity_curve.append(
                self._equity_point(timestamp, cash_equity, marked_equity, positions, mark_prices)
            )

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
        if not equity_curve or equity_curve[-1].timestamp != last_timestamp:
            equity_curve.append(self._equity_point(last_timestamp, cash_equity, cash_equity, positions, final_prices))

        return BacktestResult(
            initial_equity=self.config.portfolio.initial_equity,
            final_equity=cash_equity,
            peak_equity=risk_state.peak_equity,
            max_drawdown_pct=max_drawdown,
            trades=trades,
            skipped_entries=skipped,
            equity_curve=equity_curve,
            halted=risk_state.halted,
        )

    def _maybe_cooldown_symbol(
        self,
        trade: Trade,
        bar_index: int,
        symbol_cooldowns: dict[str, int],
    ) -> None:
        if self.grid_config.symbol_cooldown_bars <= 0:
            return
        trade_like = TradeLike(
            symbol=trade.symbol,
            pnl=trade.pnl,
            entry_reason=trade.entry_reason,
            exit_reason=trade.exit_reason,
        )
        if not should_cooldown_symbol(trade_like, self.grid_config):
            return
        until = bar_index + self.grid_config.symbol_cooldown_bars
        symbol_cooldowns[trade.symbol] = max(symbol_cooldowns.get(trade.symbol, -1), until)

    def _maybe_add_layers(
        self,
        timestamp: str,
        by_time: dict[str, dict[str, Candle]],
        positions: dict[str, GridPosition],
        mark_prices: dict[str, float],
        cash_equity: float,
        marked_equity: float,
        skipped: dict[str, int],
        candles_seen: dict[str, list[Candle]],
    ) -> float:
        for symbol, position in positions.items():
            price = mark_prices[symbol]
            recent_candles = candles_seen[symbol]
            if not should_add_grid_layer(position, price, self.grid_config, recent_candles=recent_candles):
                continue
            key_layer = is_panic_layer_candle(recent_candles, self.grid_config)
            next_notional = next_grid_notional(position, self.grid_config, key_layer=key_layer)
            next_notional = self._executable_notional(symbol, next_notional, price, skipped)
            if next_notional is None:
                continue
            if not self._can_add_layer(symbol, next_notional, marked_equity, positions):
                skipped["grid_layer_cap"] = skipped.get("grid_layer_cap", 0) + 1
                continue
            projected_average = projected_average_entry(position, by_time[symbol][timestamp].close, next_notional)
            projected_buffer = long_liquidation_buffer_pct(
                average_entry=projected_average,
                current_price=price,
                leverage=self.config.risk.leverage,
            )
            if projected_buffer < self.grid_config.min_liquidation_buffer_pct:
                skipped["liquidation_buffer_cap"] = skipped.get("liquidation_buffer_cap", 0) + 1
                continue
            layer = GridLayer(
                timestamp=timestamp,
                price=by_time[symbol][timestamp].close,
                notional=next_notional,
                margin=next_notional / self.config.risk.leverage,
            )
            position.layers.append(layer)
            cash_equity -= trading_cost(next_notional, self.config)
        return cash_equity

    def _can_open_grid(
        self,
        symbol: str,
        equity: float,
        positions: dict[str, GridPosition],
        base_notional: float | None = None,
    ) -> bool:
        if equity <= 0 or symbol in positions:
            return False
        if len(positions) >= self.config.risk.max_open_positions:
            return False
        notional = self.risk.position_notional(equity) if base_notional is None else base_notional
        return self._can_add_layer(symbol, notional, equity, positions)

    def _can_add_layer(
        self,
        symbol: str,
        layer_notional: float,
        equity: float,
        positions: dict[str, GridPosition],
    ) -> bool:
        if equity <= 0:
            return False
        symbol_notional = positions[symbol].notional if symbol in positions else 0.0
        total_notional = sum(position.notional for position in positions.values())
        if (symbol_notional + layer_notional) / equity > self.grid_config.max_symbol_notional_pct:
            return False
        if (total_notional + layer_notional) / equity > self.grid_config.max_total_notional_pct:
            return False
        return True

    def _executable_notional(
        self,
        symbol: str,
        target_notional: float,
        price: float,
        skipped: dict[str, int],
    ) -> float | None:
        symbol_filter = self.symbol_filters.get(symbol)
        if symbol_filter is None:
            return target_notional
        feasibility = symbol_feasibility(
            symbol=symbol,
            target_notional=target_notional,
            price=price,
            symbol_filter=symbol_filter,
            max_min_order_distortion=self.max_min_order_distortion,
        )
        if not feasibility.feasible:
            skipped["min_order_distortion"] = skipped.get("min_order_distortion", 0) + 1
            return None
        if feasibility.reason == "min_order_adjusted":
            return feasibility.min_executable_notional
        quantity = normalize_quantity(target_notional / price, symbol_filter.step_size)
        return quantity * price

    def _marked_equity(
        self,
        cash_equity: float,
        positions: dict[str, GridPosition],
        prices: dict[str, float],
    ) -> float:
        unrealized = 0.0
        for symbol, position in positions.items():
            for layer in position.layers:
                unrealized += layer.notional * ((prices[symbol] / layer.price) - 1.0)
        return cash_equity + unrealized

    def _equity_point(
        self,
        timestamp: str,
        cash_equity: float,
        marked_equity: float,
        positions: dict[str, GridPosition],
        prices: dict[str, float],
    ) -> EquityPoint:
        floating_pnl = 0.0
        worst_symbol_unrealized_pct = 0.0
        liquidation_buffers: list[float] = []
        for symbol, position in positions.items():
            current_price = prices[symbol]
            symbol_pnl = sum(layer.notional * ((current_price / layer.price) - 1.0) for layer in position.layers)
            floating_pnl += symbol_pnl
            if position.notional > 0:
                worst_symbol_unrealized_pct = min(
                    worst_symbol_unrealized_pct,
                    symbol_pnl / position.notional,
                )
            liquidation_buffers.append(
                long_liquidation_buffer_pct(
                    average_entry=position.average_entry,
                    current_price=current_price,
                    leverage=self.config.risk.leverage,
                )
            )
        return EquityPoint(
            timestamp=timestamp,
            cash_equity=cash_equity,
            marked_equity=marked_equity,
            open_notional=sum(position.notional for position in positions.values()),
            open_margin=sum(position.margin for position in positions.values()),
            floating_pnl=floating_pnl,
            max_layers=max((len(position.layers) for position in positions.values()), default=0),
            longest_hold_bars=max((position.bars_held for position in positions.values()), default=0),
            worst_symbol_unrealized_pct=worst_symbol_unrealized_pct,
            min_liquidation_buffer_pct=min(liquidation_buffers) if liquidation_buffers else None,
        )

    def _entry_window_open(self, timestamp: str) -> bool:
        if self.entry_start and timestamp < self.entry_start:
            return False
        if self.entry_end and timestamp >= self.entry_end:
            return False
        return True

    def _close_position(
        self,
        position: GridPosition,
        timestamp: str,
        price: float,
        reason: str,
        equity: float,
    ) -> tuple[Trade, float]:
        gross_pnl = sum(layer.notional * ((price / layer.price) - 1.0) for layer in position.layers)
        entry_cost = sum(trading_cost(layer.notional, self.config) for layer in position.layers)
        exit_cost = trading_cost(position.notional, self.config)
        pnl = gross_pnl - entry_cost - exit_cost
        new_equity = equity + gross_pnl - exit_cost
        return (
            Trade(
                symbol=position.symbol,
                entry_time=position.entry_time,
                exit_time=timestamp,
                entry_price=position.average_entry,
                exit_price=price,
                notional=position.notional,
                pnl=pnl,
                entry_reason=f"{position.entry_reason}|layers={len(position.layers)}",
                exit_reason=reason,
            ),
            new_equity,
        )

    def _trim_position_layers(
        self,
        position: GridPosition,
        timestamp: str,
        price: float,
        reason: str,
        equity: float,
    ) -> tuple[Trade, float]:
        original_layer_count = len(position.layers)
        trim_count = min(
            max(1, self.grid_config.layer_trim_count),
            max(0, original_layer_count - 1),
        )
        closed_layers = [position.layers.pop() for _ in range(trim_count)]
        trimmed_notional = sum(layer.notional for layer in closed_layers)
        weighted_entry = (
            sum(layer.price * layer.notional for layer in closed_layers) / trimmed_notional
            if trimmed_notional > 0
            else 0.0
        )
        gross_pnl = sum(layer.notional * ((price / layer.price) - 1.0) for layer in closed_layers)
        entry_cost = sum(trading_cost(layer.notional, self.config) for layer in closed_layers)
        exit_cost = trading_cost(trimmed_notional, self.config)
        pnl = gross_pnl - entry_cost - exit_cost
        new_equity = equity + gross_pnl - exit_cost
        return (
            Trade(
                symbol=position.symbol,
                entry_time=closed_layers[-1].timestamp,
                exit_time=timestamp,
                entry_price=weighted_entry,
                exit_price=price,
                notional=trimmed_notional,
                pnl=pnl,
                entry_reason=f"{position.entry_reason}|trimmed={trim_count}|layers={original_layer_count}",
                exit_reason=reason,
            ),
            new_equity,
        )


def should_add_grid_layer(
    position: GridPosition,
    current_price: float,
    config: BoundedGridConfig,
    recent_candles: list[Candle] | None = None,
) -> bool:
    if position.rescue_layers >= config.max_rescue_layers:
        return False
    step_pct = effective_layer_step_pct(config, recent_candles, current_rescue_layers=position.rescue_layers)
    if current_price > position.last_layer_price * (1.0 - step_pct):
        return False
    if layer_panic_filter_enabled(config):
        return is_panic_layer_candle(recent_candles or [], config)
    return True


def next_grid_notional(
    position: GridPosition,
    config: BoundedGridConfig,
    *,
    key_layer: bool = False,
) -> float:
    base_notional = position.layers[0].notional
    rescue_index = position.rescue_layers + 1
    notional = base_notional * (config.layer_multiplier ** rescue_index)
    if key_layer and config.key_layer_multiplier > 1:
        notional *= config.key_layer_multiplier
    return notional


def effective_layer_step_pct(
    config: BoundedGridConfig,
    recent_candles: list[Candle] | None = None,
    *,
    current_rescue_layers: int = 0,
) -> float:
    base = config.layer_step_pct
    if (
        config.atr_layer_lookback_bars > 0
        and config.atr_layer_step_multiplier > 0
        and recent_candles
    ):
        highs = [candle.high for candle in recent_candles]
        lows = [candle.low for candle in recent_candles]
        closes = [candle.close for candle in recent_candles]
        recent_atr = atr_pct(highs, lows, closes, config.atr_layer_lookback_bars)
        if recent_atr is not None:
            base = max(base, recent_atr * config.atr_layer_step_multiplier)
    # Widen spacing for deeper rescue layers
    if config.layer_step_expansion_rate > 0 and current_rescue_layers > 0:
        expansion = 1.0 + current_rescue_layers * config.layer_step_expansion_rate
        base *= expansion
    return base


def layer_panic_filter_enabled(config: BoundedGridConfig) -> bool:
    volume_filter = config.volume_layer_lookback_bars > 0 and config.volume_layer_multiplier > 0
    wick_filter = config.min_layer_lower_wick_pct > 0
    return volume_filter or wick_filter


def is_panic_layer_candle(recent_candles: list[Candle], config: BoundedGridConfig) -> bool:
    if not layer_panic_filter_enabled(config):
        return False
    if not recent_candles:
        return False
    latest = recent_candles[-1]
    if config.min_layer_lower_wick_pct > 0:
        wick_pct = lower_wick_pct(
            open_price=latest.open,
            low=latest.low,
            close=latest.close,
        )
        if wick_pct < config.min_layer_lower_wick_pct:
            return False
    if config.volume_layer_lookback_bars > 0 and config.volume_layer_multiplier > 0:
        volumes = [candle.volume for candle in recent_candles]
        if not volume_spike(volumes, config.volume_layer_lookback_bars, config.volume_layer_multiplier):
            return False
    return True


def should_trim_grid_layers(
    position: GridPosition,
    current_price: float,
    config: BoundedGridConfig,
    leverage: float,
) -> bool:
    if config.layer_trim_buffer_pct <= 0:
        return False
    min_layers = max(2, config.layer_trim_min_layers)
    if len(position.layers) < min_layers:
        return False
    buffer_pct = long_liquidation_buffer_pct(
        average_entry=position.average_entry,
        current_price=current_price,
        leverage=leverage,
    )
    return buffer_pct <= config.layer_trim_buffer_pct


def projected_average_entry(position: GridPosition, next_price: float, next_notional: float) -> float:
    projected_notional = position.notional + next_notional
    if projected_notional <= 0:
        return 0.0
    weighted = sum(layer.price * layer.notional for layer in position.layers)
    weighted += next_price * next_notional
    return weighted / projected_notional


def estimated_long_liquidation_price(
    average_entry: float,
    leverage: float,
    maintenance_margin_pct: float = MAINTENANCE_MARGIN_PCT,
) -> float:
    if average_entry <= 0 or leverage <= 0:
        return 0.0
    return average_entry * (1.0 - (1.0 / leverage) + maintenance_margin_pct)


def long_liquidation_buffer_pct(
    *,
    average_entry: float,
    current_price: float,
    leverage: float,
    maintenance_margin_pct: float = MAINTENANCE_MARGIN_PCT,
) -> float:
    if current_price <= 0:
        return 0.0
    liquidation_price = estimated_long_liquidation_price(
        average_entry,
        leverage,
        maintenance_margin_pct,
    )
    return (current_price - liquidation_price) / current_price


def grid_exit_reason(
    position: GridPosition,
    candle: Candle,
    circuit_active: bool,
    config: RaynConfig,
    grid_config: BoundedGridConfig,
    recent_atr_pct: float | None = None,
    intrabar_liquidation: bool = False,
) -> str:
    average = position.average_entry
    leverage = config.risk.leverage
    take_profit_pct = effective_take_profit_pct(position, grid_config, recent_atr_pct=recent_atr_pct)
    # Hard intrabar liquidation. The exchange force-closes a long when the worst
    # intrabar price (the bar low) reaches the maintenance-margin liquidation
    # price. Checked first and on the low, not the close, because it happens
    # mid-bar before any close-based or soft control: a deep wick that "recovers"
    # by the close has still already been liquidated. At high leverage this
    # pre-empts the hard stop entirely (liq is nearer than the configured stop),
    # which is the realistic tail the close-based model hid.
    if intrabar_liquidation and candle.low <= estimated_long_liquidation_price(average, leverage):
        return "liquidation"
    if circuit_active and config.circuit_breaker.close_positions_on_trigger:
        return "circuit_breaker"
    if grid_config.liquidation_exit_buffer_pct > 0:
        # Intrabar model evaluates the soft de-risk on the bar low, so a wick that
        # recovers by the close still trips it; the legacy model used the close.
        ref_price = candle.low if intrabar_liquidation else candle.close
        buffer_pct = long_liquidation_buffer_pct(
            average_entry=average,
            current_price=ref_price,
            leverage=leverage,
        )
        if buffer_pct <= grid_config.liquidation_exit_buffer_pct:
            return "liquidation_buffer_exit"
    if candle.low <= average * (1.0 - grid_config.hard_stop_pct):
        return "grid_hard_stop"
    if candle.high >= average * (1.0 + take_profit_pct):
        return "grid_take_profit"
    if position.bars_held >= config.strategy.max_hold_bars:
        return "time_stop"
    return ""


def grid_exit_price(
    position: GridPosition,
    close_price: float,
    reason: str,
    grid_config: BoundedGridConfig,
    recent_atr_pct: float | None = None,
    leverage: float | None = None,
) -> float:
    average = position.average_entry
    if reason == "liquidation" and leverage is not None:
        return estimated_long_liquidation_price(average, leverage)
    if reason == "grid_hard_stop":
        return average * (1.0 - grid_config.hard_stop_pct)
    if reason == "grid_take_profit":
        return average * (1.0 + effective_take_profit_pct(position, grid_config, recent_atr_pct=recent_atr_pct))
    return close_price


def effective_take_profit_pct(
    position: GridPosition,
    config: BoundedGridConfig,
    recent_atr_pct: float | None = None,
) -> float:
    if (
        config.deep_take_profit_layer > 0
        and config.deep_take_profit_pct > 0
        and len(position.layers) >= config.deep_take_profit_layer
    ):
        base = config.deep_take_profit_pct
    else:
        base = config.take_profit_pct
    if config.dynamic_take_profit_atr_multiplier <= 0 or recent_atr_pct is None:
        return base
    target = max(base, recent_atr_pct * config.dynamic_take_profit_atr_multiplier)
    if config.dynamic_take_profit_min_pct > 0:
        target = max(target, config.dynamic_take_profit_min_pct)
    if config.dynamic_take_profit_max_pct > 0:
        target = min(target, config.dynamic_take_profit_max_pct)
    return target


def recent_atr_pct_for_take_profit(
    recent_candles: list[Candle],
    config: BoundedGridConfig,
) -> float | None:
    if config.dynamic_take_profit_atr_lookback_bars <= 0:
        return None
    highs = [candle.high for candle in recent_candles]
    lows = [candle.low for candle in recent_candles]
    closes = [candle.close for candle in recent_candles]
    return atr_pct(highs, lows, closes, config.dynamic_take_profit_atr_lookback_bars)


def should_cooldown_symbol(trade: TradeLike, config: BoundedGridConfig) -> bool:
    if config.symbol_cooldown_bars <= 0:
        return False
    if trade.exit_reason in set(config.symbol_cooldown_exit_reasons):
        return True
    if config.symbol_cooldown_min_loss_layers <= 0:
        return False
    return trade.pnl < 0 and trade_layers(trade.entry_reason) >= config.symbol_cooldown_min_loss_layers
