"""Main orchestration loop for the MetaTrader 5 execution bot."""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import statistics
import time
from collections import deque
from typing import Dict, List, Optional

import MetaTrader5 as mt5
import numpy as np

from broker import (
    close_position,
    get_open_positions,
    make_legal_sl_tp,
    modify_stop_to_breakeven,
    send_entry,
    trail_stop,
)
from config import CFG
from filters import REGIME_UNSURE, market_state
from risk import (
    DailyState,
    STOP_LOSS,
    STOP_MAX_TRADES,
    STOP_NONE,
    STOP_TARGET,
    daily_stop,
    lots_for_risk,
    manage_open_trade,
    max_trades_reached,
    trade_risk_dollars,
)


logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, cfg: Dict, dryrun: bool = False) -> None:
        self.cfg = cfg
        self.symbol = cfg["SYMBOL"]
        self.dryrun = dryrun
        self.daily_state = DailyState()
        self.last_day = None
        self.spreads = deque(maxlen=120)
        self.timeframe = mt5.TIMEFRAME_M5

    def initialize(self) -> None:
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MetaTrader5")
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"Unable to select symbol {self.symbol}")
        info = mt5.account_info()
        if info is None:
            raise RuntimeError("Account info unavailable")
        self._reset_daily(info.equity)

    def _reset_daily(self, equity: float) -> None:
        today = dt.date.today()
        self.last_day = today
        self.daily_state.reset(equity)
        logger.info("[INIT] Daily counters reset. equity=%.2f", equity)

    def _update_daily(self, equity: float) -> None:
        today = dt.date.today()
        if self.last_day != today:
            self._reset_daily(equity)
        self.daily_state.update_equity(equity)

    def _dynamic_spread_cap(self) -> int:
        base = self.cfg["SPREAD_POINTS_BASE_CAP"]
        if len(self.spreads) < 5:
            return base
        median_spread = statistics.median(self.spreads)
        cap = max(120, min(240, int(median_spread * 2.2)))
        return max(cap, base)

    def _fetch_rates(self, count: int = 200) -> Optional[np.ndarray]:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return None
        return rates

    def step(self) -> None:
        account = mt5.account_info()
        if account is None:
            logger.error("[STATE] Account info unavailable; skipping step")
            return

        self._update_daily(account.equity)

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error("[STATE] Tick unavailable; skipping step")
            return

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error("[STATE] Symbol info unavailable; skipping step")
            return

        point = symbol_info.point
        spread_points = (tick.ask - tick.bid) / point if point else 0.0
        self.spreads.append(spread_points)
        spread_cap = self._dynamic_spread_cap()

        rates = self._fetch_rates()
        if rates is None:
            logger.error("[STATE] Failed to fetch rates; skipping step")
            return

        market = market_state(rates, self.cfg)
        atr = market.get("atr")
        adx = market.get("adx")
        regime = market.get("regime", REGIME_UNSURE)

        logger.info(
            "[STATE] regime=%s spread=%.1f/%.0f ATR=%s ADX=%s",
            regime,
            spread_points,
            spread_cap,
            f"{atr:.2f}" if atr is not None else "--",
            f"{adx:.1f}" if adx is not None else "--",
        )

        daily_reason = daily_stop(
            self.daily_state.change_pct,
            self.daily_state.dd_pct,
            self.cfg["DAILY_TARGET_PCT"],
            self.cfg["DAILY_MAX_DD_PCT"],
        )
        trades_limit_hit = max_trades_reached(self.daily_state.trades, self.cfg["MAX_TRADES_PER_DAY"])

        daily_label = daily_reason
        if trades_limit_hit and daily_label == STOP_NONE:
            daily_label = STOP_MAX_TRADES

        logger.info(
            "[RISK] equity=%.2f trades_today=%s/%s daily_stop=%s",
            account.equity,
            self.daily_state.trades,
            self.cfg["MAX_TRADES_PER_DAY"],
            daily_label,
        )

        gate_reasons: List[str] = []
        if atr is None:
            gate_reasons.append("ATR unavailable")
        elif atr < self.cfg["ATR_MIN"]:
            gate_reasons.append("ATR<min")
        if adx is None:
            gate_reasons.append("ADX unavailable")
        if spread_points > spread_cap:
            gate_reasons.append("spread>cap")
        if daily_reason in {STOP_LOSS, STOP_TARGET}:
            gate_reasons.append(f"daily {daily_reason.lower()}")
        if trades_limit_hit:
            gate_reasons.append("daily max trades")

        logger.info(
            "[GATE] %s",
            ", ".join(gate_reasons) if gate_reasons else "clear",
        )

        positions = get_open_positions(self.symbol) or []

        if positions:
            self._manage_positions(positions, atr, tick)
            return

        # Reset hedge allowance when flat
        self.daily_state.hedge_used = False

        if gate_reasons:
            return

        signal = market.get("signal")
        size_factor = market.get("size_factor", 0.0)
        if signal not in {"LONG", "SHORT"}:
            logger.info("[GATE] no_trade: regime UNSURE")
            return

        if size_factor <= 0:
            logger.info("[GATE] no_trade: size factor 0")
            return

        if self.cfg["MAX_CONCURRENT_POS"] <= 0:
            logger.info("[GATE] no_trade: max concurrent 0")
            return

        stop_distance = self.cfg["SL_ATR_MULT"] * atr
        tp_distance = stop_distance * self.cfg["TP_R_MULT"]

        risk_pct = self.cfg["RISK_PCT_PER_TRADE"] * size_factor
        try:
            lots = lots_for_risk(self.symbol, account.equity, risk_pct, stop_distance)
        except ValueError as exc:
            logger.error("[ENTRY] sizing_error: %s", exc)
            return
        if lots <= 0:
            logger.info("[GATE] no_trade: lot size<=0")
            return

        entry_price = float(tick.ask if signal == "LONG" else tick.bid)
        if signal == "LONG":
            sl_price = entry_price - stop_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + stop_distance
            tp_price = entry_price - tp_distance

        sl_price, tp_price = make_legal_sl_tp(signal, entry_price, sl_price, tp_price, self.symbol)
        r_value = trade_risk_dollars(self.symbol, lots, stop_distance)

        logger.info(
            "[ENTRY] side=%s lot=%.2f price=%.3f SL=%.3f TP=%.3f R=$%.2f",
            signal,
            lots,
            entry_price,
            sl_price,
            tp_price,
            r_value,
        )

        if not self.dryrun:
            result = send_entry(
                self.symbol,
                signal,
                lots,
                "bot_entry",
                sl_price,
                tp_price,
                dryrun=False,
            )
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.daily_state.trades += 1
        else:
            logger.info("[ENTRY] dryrun active - order not sent")
            self.daily_state.trades += 1

    def _manage_positions(self, positions, atr: Optional[float], tick) -> None:
        if atr is None or atr <= 0:
            logger.info("[MX] ATR unavailable for management")
            return

        for pos in positions:
            direction = "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
            price = float(tick.bid if direction == "LONG" else tick.ask)
            stop_distance = (
                pos.price_open - pos.sl
                if direction == "LONG"
                else pos.sl - pos.price_open
            )
            if stop_distance <= 0:
                stop_distance = self.cfg["SL_ATR_MULT"] * atr

            pnl_dollars = float(pos.profit)
            r_value = trade_risk_dollars(self.symbol, pos.volume, stop_distance)
            decision = manage_open_trade(
                pnl_dollars,
                r_value,
                self.cfg["BE_TRIGGER_R"],
                self.cfg["TRAIL_AFTER_R"],
            )
            r_multiple = pnl_dollars / r_value if r_value else 0.0

            logger.info(
                "[MX] ticket=%s action=%s r_mult=%.2f pnl=$%.2f",
                pos.ticket,
                decision,
                r_multiple,
                pnl_dollars,
            )

            if decision == "BREAKEVEN_SL":
                modify_stop_to_breakeven(pos.ticket, self.symbol, dryrun=self.dryrun)
            elif decision == "TRAIL":
                trail_distance = self.cfg["TRAIL_ATR_MULT"] * atr
                if direction == "LONG":
                    candidate_sl = max(float(pos.sl), price - trail_distance)
                else:
                    candidate_sl = min(float(pos.sl) if pos.sl else price + trail_distance, price + trail_distance)
                trail_stop(pos.ticket, candidate_sl, self.symbol, dryrun=self.dryrun)
            elif decision == "CUT_OR_HEDGE":
                self._handle_cut_or_hedge(pos, atr, price)

    def _handle_cut_or_hedge(self, pos, atr: float, price: float) -> None:
        direction = "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
        if self.cfg["ALLOW_SINGLE_HEDGE"] and not self.daily_state.hedge_used:
            hedge_dir = "SHORT" if direction == "LONG" else "LONG"
            stop_distance = self.cfg["SL_ATR_MULT"] * atr
            tp_distance = stop_distance * self.cfg["TP_R_MULT"]
            entry_price = price
            if hedge_dir == "LONG":
                sl_price = entry_price - stop_distance
                tp_price = entry_price + tp_distance
            else:
                sl_price = entry_price + stop_distance
                tp_price = entry_price - tp_distance

            sl_price, tp_price = make_legal_sl_tp(hedge_dir, entry_price, sl_price, tp_price, self.symbol)
            logger.info(
                "[HEDGE] placing hedge dir=%s lot=%.2f price=%.3f SL=%.3f TP=%.3f",
                hedge_dir,
                pos.volume,
                entry_price,
                sl_price,
                tp_price,
            )
            if not self.dryrun:
                send_entry(
                    self.symbol,
                    hedge_dir,
                    pos.volume,
                    "hedge",
                    sl_price,
                    tp_price,
                    dryrun=False,
                )
            self.daily_state.hedge_used = True
        else:
            logger.info("[HEDGE] closing losing leg ticket=%s", pos.ticket)
            close_position(pos.ticket, self.symbol, dryrun=self.dryrun)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MT5 execution bot")
    parser.add_argument("--dryrun", action="store_true", help="Simulate without sending orders")
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> None:
    args = parse_args()
    setup_logging()

    bot = TradingBot(CFG, dryrun=args.dryrun)
    bot.initialize()

    logger.info("[INIT] Bot started. dryrun=%s", args.dryrun)

    try:
        while True:
            bot.step()
            if args.once:
                break
            time.sleep(CFG["ENTRY_COOLDOWN_SEC"])
    finally:
        mt5.shutdown()
        logger.info("[SHUTDOWN] MT5 connection closed")


if __name__ == "__main__":
    main()
