"""Risk controls and position sizing helpers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import MetaTrader5 as mt5


logger = logging.getLogger(__name__)

STOP_NONE = "NONE"
STOP_LOSS = "STOP_LOSS"
STOP_TARGET = "STOP_TARGET"
STOP_MAX_TRADES = "MAX_TRADES"


@dataclass
class DailyState:
    start_equity: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    last_equity: float = 0.0
    trades: int = 0
    hedge_used: bool = False

    def reset(self, equity: float) -> None:
        self.start_equity = equity
        self.peak_equity = equity
        self.trough_equity = equity
        self.last_equity = equity
        self.trades = 0
        self.hedge_used = False

    def update_equity(self, equity: float) -> None:
        self.peak_equity = max(self.peak_equity, equity)
        self.trough_equity = min(self.trough_equity, equity)
        self.last_equity = equity

    @property
    def change_pct(self) -> float:
        if self.start_equity <= 0:
            return 0.0
        return ((self.last_equity or self.start_equity) - self.start_equity) / self.start_equity * 100.0

    @property
    def dd_pct(self) -> float:
        if self.start_equity <= 0:
            return 0.0
        return (self.trough_equity - self.start_equity) / self.start_equity * 100.0


def _snap_to_step(volume: float, step: float) -> float:
    step = max(step, 1e-8)
    return round(volume / step) * step


def lots_for_risk(symbol: str, equity: float, risk_pct: float, stop_distance_price: float) -> float:
    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError(f"Symbol info unavailable for {symbol}")

    tick_value = getattr(si, "trade_tick_value", 0.0) or None
    tick_size = getattr(si, "trade_tick_size", 0.0) or None
    if tick_value and tick_size:
        dollars_per_price = tick_value / tick_size
    else:
        dollars_per_price = si.trade_contract_size * si.point

    stop_distance_price = abs(stop_distance_price)
    if stop_distance_price <= 0:
        raise ValueError("Stop distance must be positive for sizing")

    dollars_per_lot_at_stop = stop_distance_price * dollars_per_price
    acct_risk = equity * risk_pct
    raw_lots = acct_risk / max(1e-9, dollars_per_lot_at_stop)

    min_vol = si.volume_min
    max_vol = si.volume_max
    raw_lots = max(min_vol, min(max_vol, raw_lots))
    snapped = _snap_to_step(raw_lots, si.volume_step)
    precision = getattr(si, "volume_precision", 2)
    return round(max(min_vol, min(max_vol, snapped)), precision)


def dollars_per_price_unit(symbol: str) -> float:
    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError(f"Symbol info unavailable for {symbol}")

    tick_value = getattr(si, "trade_tick_value", 0.0) or None
    tick_size = getattr(si, "trade_tick_size", 0.0) or None
    if tick_value and tick_size:
        return tick_value / tick_size
    return si.trade_contract_size * si.point


def trade_risk_dollars(symbol: str, volume: float, stop_distance_price: float) -> float:
    return abs(stop_distance_price) * dollars_per_price_unit(symbol) * volume


Decision = Literal["TRAIL", "BREAKEVEN_SL", "CUT_OR_HEDGE", "HOLD"]


def manage_open_trade(
    pnl_dollars: float,
    r_value_dollars: float,
    be_trigger_r: float,
    trail_after_r: float,
) -> Decision:
    if r_value_dollars <= 0:
        return "HOLD"

    r_multiple = pnl_dollars / r_value_dollars
    if r_multiple >= trail_after_r:
        return "TRAIL"
    if r_multiple >= be_trigger_r:
        return "BREAKEVEN_SL"
    if r_multiple <= -1.0:
        return "CUT_OR_HEDGE"
    return "HOLD"


def daily_stop(change_pct: float, dd_pct: float, target_pct: float, max_dd_pct: float) -> str:
    if dd_pct <= -max_dd_pct:
        return STOP_LOSS
    if change_pct >= target_pct:
        return STOP_TARGET
    return STOP_NONE


def max_trades_reached(count: int, limit: int) -> bool:
    return count >= limit
