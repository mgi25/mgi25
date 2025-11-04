"""Broker execution utilities for MetaTrader 5."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import MetaTrader5 as mt5


logger = logging.getLogger(__name__)


def _snap_to_step(volume: float, step: float) -> float:
    step = max(step, 1e-8)
    snapped = round(volume / step) * step
    return max(0.0, snapped)


def _round_volume(symbol_info, volume: float) -> float:
    """Round the requested volume to the nearest legal lot size."""
    volume = max(symbol_info.volume_min, min(symbol_info.volume_max, volume))
    volume = _snap_to_step(volume, symbol_info.volume_step)
    precision = getattr(symbol_info, "volume_precision", 2)
    return round(volume, precision)


def make_legal_sl_tp(
    direction: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    symbol: str,
) -> Tuple[float, float]:
    """Adjust stop-loss / take-profit to respect symbol stop and freeze levels."""
    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError(f"Symbol info unavailable for {symbol}")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Symbol tick unavailable for {symbol}")

    digits = si.digits
    point = si.point
    stops = getattr(si, "trade_stops_level", 0) * point
    freeze = getattr(si, "trade_freeze_level", 0) * point

    bid = float(tick.bid)
    ask = float(tick.ask)

    buffer = point
    if direction.upper() == "LONG":
        min_sl = bid - stops - buffer
        if sl_price > min_sl:
            logger.info(
                "[BROKER] SL adjusted for stops level (long): target=%.5f adjusted=%.5f",
                sl_price,
                min_sl,
            )
            sl_price = min_sl
        min_tp = ask + stops + buffer
        if tp_price < min_tp:
            logger.info(
                "[BROKER] TP adjusted for stops level (long): target=%.5f adjusted=%.5f",
                tp_price,
                min_tp,
            )
            tp_price = min_tp
    else:
        min_sl = ask + stops + buffer
        if sl_price < min_sl:
            logger.info(
                "[BROKER] SL adjusted for stops level (short): target=%.5f adjusted=%.5f",
                sl_price,
                min_sl,
            )
            sl_price = min_sl
        min_tp = bid - stops - buffer
        if tp_price > min_tp:
            logger.info(
                "[BROKER] TP adjusted for stops level (short): target=%.5f adjusted=%.5f",
                tp_price,
                min_tp,
            )
            tp_price = min_tp

    def _normalize(price: float) -> float:
        factor = 10 ** digits
        return int(price * factor + 0.5) / factor

    sl_price = _normalize(sl_price)
    tp_price = _normalize(tp_price)

    # Respect freeze level: ensure distance is outside freeze, otherwise defer a tick away.
    if direction.upper() == "LONG":
        freeze_floor = bid - freeze - buffer
        if sl_price > freeze_floor:
            logger.info(
                "[BROKER] SL within freeze zone (long). Deferring to %.5f",
                freeze_floor,
            )
            sl_price = _normalize(freeze_floor)
    else:
        freeze_ceiling = ask + freeze + buffer
        if sl_price < freeze_ceiling:
            logger.info(
                "[BROKER] SL within freeze zone (short). Deferring to %.5f",
                freeze_ceiling,
            )
            sl_price = _normalize(freeze_ceiling)

    return sl_price, tp_price


def send_entry(
    symbol: str,
    direction: str,
    volume: float,
    comment: str,
    sl_price: float,
    tp_price: float,
    dryrun: bool = False,
) -> Optional[mt5.OrderSendResult]:
    """Submit a market order with attached protective stops."""
    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError(f"Symbol info unavailable for {symbol}")

    volume = _round_volume(si, volume)

    direction = direction.upper()
    order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Symbol tick unavailable for {symbol}")

    price = float(tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid)
    sl_price, tp_price = make_legal_sl_tp(direction, price, sl_price, tp_price, symbol)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20,
        "comment": comment,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    logger.info(
        "[BROKER] Sending order: dir=%s volume=%.2f price=%.5f sl=%.5f tp=%.5f dryrun=%s",
        direction,
        volume,
        price,
        sl_price,
        tp_price,
        dryrun,
    )

    if dryrun:
        return None

    result = mt5.order_send(request)
    if result is None:
        logger.error("[BROKER] order_send returned None")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error("[BROKER] Order failed: retcode=%s comment=%s", result.retcode, result.comment)
    else:
        logger.info("[BROKER] Order placed. ticket=%s", result.order)
    return result


def modify_stop_to_breakeven(ticket: int, symbol: str, dryrun: bool = False) -> bool:
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning("[BROKER] No position found for ticket %s", ticket)
        return False

    pos = positions[0]
    direction = "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
    current_sl = float(pos.sl)
    breakeven_price = float(pos.price_open)
    if direction == "LONG" and current_sl >= breakeven_price:
        logger.info("[BROKER] Breakeven SL already in place for ticket %s", ticket)
        return True
    if direction == "SHORT" and (current_sl <= breakeven_price and current_sl != 0.0):
        logger.info("[BROKER] Breakeven SL already in place for ticket %s", ticket)
        return True

    _, tp = make_legal_sl_tp(direction, pos.price_open, breakeven_price, pos.tp or 0.0, symbol)
    sl, _ = make_legal_sl_tp(direction, pos.price_open, breakeven_price, tp, symbol)

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
        "sl": sl,
        "tp": pos.tp,
        "comment": "breakeven",
    }

    logger.info("[BROKER] Moving SL to breakeven for ticket=%s sl=%.5f", ticket, sl)

    if dryrun:
        return True

    result = mt5.order_send(request)
    if result is None:
        logger.error("[BROKER] Breakeven modification failed (None)")
        return False

    if result.retcode not in {mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_NO_CHANGES}:
        logger.error(
            "[BROKER] Breakeven modification failed: retcode=%s comment=%s",
            result.retcode,
            result.comment,
        )
        return False

    return True


def trail_stop(ticket: int, trail_price: float, symbol: str, dryrun: bool = False) -> bool:
    """Trail the stop for a position while respecting freeze levels."""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning("[BROKER] No position found for trailing ticket %s", ticket)
        return False

    pos = positions[0]
    direction = "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
    current_sl = float(pos.sl)
    desired_sl = float(trail_price)

    if direction == "LONG" and current_sl != 0.0 and desired_sl <= current_sl:
        logger.debug("[BROKER] Trail not applied (long). desired=%.5f current=%.5f", desired_sl, current_sl)
        return False
    if direction == "SHORT" and current_sl != 0.0 and desired_sl >= current_sl:
        logger.debug("[BROKER] Trail not applied (short). desired=%.5f current=%.5f", desired_sl, current_sl)
        return False

    sl, tp = make_legal_sl_tp(direction, pos.price_open, desired_sl, pos.tp or 0.0, symbol)

    if direction == "LONG" and sl <= current_sl:
        logger.debug("[BROKER] Adjusted SL is not higher for long trail. sl=%.5f current=%.5f", sl, current_sl)
        return False
    if direction == "SHORT" and (current_sl == 0.0 or sl >= current_sl):
        logger.debug("[BROKER] Adjusted SL is not lower for short trail. sl=%.5f current=%.5f", sl, current_sl)
        return False

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
        "sl": sl,
        "tp": pos.tp,
        "comment": "trail",
    }

    logger.info("[BROKER] Trailing SL for ticket=%s to %.5f", ticket, sl)

    if dryrun:
        return True

    result = mt5.order_send(request)
    if result is None:
        logger.error("[BROKER] Trail modification failed (None)")
        return False

    if result.retcode not in {mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_NO_CHANGES}:
        logger.error("[BROKER] Trail modification failed: retcode=%s comment=%s", result.retcode, result.comment)
        return False

    return True


def close_position(ticket: int, symbol: str, dryrun: bool = False) -> bool:
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning("[BROKER] No position to close for ticket %s", ticket)
        return False

    pos = positions[0]
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Symbol tick unavailable for {symbol}")

    price = float(tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "position": ticket,
        "volume": pos.volume,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "comment": "close",
    }

    logger.info("[BROKER] Closing ticket=%s at price %.5f", ticket, price)

    if dryrun:
        return True

    result = mt5.order_send(request)
    if result is None:
        logger.error("[BROKER] Close order failed (None)")
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error("[BROKER] Close order failed: retcode=%s comment=%s", result.retcode, result.comment)
        return False

    return True


def get_open_positions(symbol: str):
    return mt5.positions_get(symbol=symbol)
