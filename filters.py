"""Indicator and regime filters for the execution bot."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


REGIME_TREND_LONG = "TREND_LONG"
REGIME_TREND_SHORT = "TREND_SHORT"
REGIME_UNSURE = "UNSURE"


def _ema(values: np.ndarray, period: int) -> Optional[float]:
    if period <= 0 or len(values) < period:
        return None
    multiplier = 2 / (period + 1.0)
    ema = values[0]
    for price in values[1:]:
        ema = (price - ema) * multiplier + ema
    return float(ema)


def calculate_atr(rates: np.ndarray, period: int) -> Optional[float]:
    if len(rates) < period + 1:
        return None

    highs = rates["high"].astype(float)
    lows = rates["low"].astype(float)
    closes = rates["close"].astype(float)

    tr = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1]),
    ])

    atr = tr[:period].mean()
    for value in tr[period:]:
        atr = (atr * (period - 1) + value) / period
    return float(atr)


def calculate_adx(rates: np.ndarray, period: int) -> Optional[float]:
    if len(rates) < period + 1:
        return None

    highs = rates["high"].astype(float)
    lows = rates["low"].astype(float)
    closes = rates["close"].astype(float)

    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_components = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1]),
    ])

    atr = tr_components[:period].sum()
    plus_dm_sum = plus_dm[:period].sum()
    minus_dm_sum = minus_dm[:period].sum()

    adx_values = []
    for i in range(period, len(tr_components)):
        atr = atr - atr / period + tr_components[i]
        plus_dm_sum = plus_dm_sum - plus_dm_sum / period + plus_dm[i]
        minus_dm_sum = minus_dm_sum - minus_dm_sum / period + minus_dm[i]

        plus_di = 100 * (plus_dm_sum / atr if atr else 0)
        minus_di = 100 * (minus_dm_sum / atr if atr else 0)
        dx = 100 * (abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-9))
        adx_values.append(dx)

    if not adx_values:
        return None

    adx = adx_values[0]
    for value in adx_values[1:]:
        adx = (adx * (period - 1) + value) / period
    return float(adx)


def donchian_channels(highs: np.ndarray, lows: np.ndarray, lookback: int) -> Tuple[float, float]:
    if lookback <= 0 or len(highs) < lookback or len(lows) < lookback:
        lookback = min(len(highs), len(lows))
    window_high = highs[-lookback:]
    window_low = lows[-lookback:]
    return float(window_high.max()), float(window_low.min())


def market_state(rates: np.ndarray, cfg: Dict) -> Dict[str, Optional[float]]:
    period = cfg["ATR_PERIOD"]
    atr = calculate_atr(rates, period)
    adx = calculate_adx(rates, period)
    if atr is None or adx is None:
        return {
            "regime": REGIME_UNSURE,
            "atr": atr,
            "adx": adx,
            "ema_fast": None,
            "ema_slow": None,
            "donchian_high": None,
            "donchian_low": None,
            "signal": None,
            "size_factor": 0.0,
        }

    if atr < cfg["ATR_MIN"]:
        return {
            "regime": REGIME_UNSURE,
            "atr": atr,
            "adx": adx,
            "ema_fast": None,
            "ema_slow": None,
            "donchian_high": None,
            "donchian_low": None,
            "signal": None,
            "size_factor": 0.0,
        }

    closes = rates["close"].astype(float)
    highs = rates["high"].astype(float)
    lows = rates["low"].astype(float)

    ema_fast = _ema(closes[-cfg["EMA_FAST"] * 2 :], cfg["EMA_FAST"])
    ema_slow = _ema(closes[-cfg["EMA_SLOW"] * 2 :], cfg["EMA_SLOW"])

    donchian_high = None
    donchian_low = None
    regime = REGIME_UNSURE
    signal = None
    size_factor = 0.0

    if ema_fast is not None and ema_slow is not None:
        donchian_high, donchian_low = donchian_channels(highs, lows, cfg["DONCHIAN_LKB"])
        price = closes[-1]

        if adx >= cfg["ADX_TREND_MIN"]:
            if ema_fast > ema_slow and price > donchian_high:
                regime = REGIME_TREND_LONG
                signal = "LONG"
                size_factor = 1.0
            elif ema_fast < ema_slow and price < donchian_low:
                regime = REGIME_TREND_SHORT
                signal = "SHORT"
                size_factor = 1.0
        elif cfg["ADX_MICRO_MIN"] <= adx < cfg["ADX_TREND_MIN"]:
            half_atr = 0.5 * atr
            if ema_fast > ema_slow and 0 <= donchian_high - price <= half_atr:
                regime = REGIME_TREND_LONG
                signal = "LONG"
                size_factor = 0.5
            elif ema_fast < ema_slow and 0 <= price - donchian_low <= half_atr:
                regime = REGIME_TREND_SHORT
                signal = "SHORT"
                size_factor = 0.5
    return {
        "regime": regime,
        "atr": atr,
        "adx": adx,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "donchian_high": donchian_high if 'donchian_high' in locals() else None,
        "donchian_low": donchian_low if 'donchian_low' in locals() else None,
        "signal": signal,
        "size_factor": size_factor,
    }
