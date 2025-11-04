CFG = {
    "SYMBOL": "XAUUSDm",
    # session/risk
    "RISK_PCT_PER_TRADE": 0.0075,
    "DAILY_MAX_DD_PCT": 6,
    "DAILY_TARGET_PCT": 4,
    "MAX_TRADES_PER_DAY": 30,
    # spread & regime gates
    "SPREAD_POINTS_BASE_CAP": 190,
    "ADX_TREND_MIN": 25,
    "ADX_MICRO_MIN": 14,
    "DONCHIAN_LKB": 14,
    "EMA_FAST": 13,
    "EMA_SLOW": 34,
    # volatility targets (ATR in price units)
    "ATR_PERIOD": 14,
    "ATR_MIN": 0.8,
    "SL_ATR_MULT": 1.1,
    "TP_R_MULT": 1.6,
    # management
    "BE_TRIGGER_R": 0.5,
    "TRAIL_AFTER_R": 1.0,
    "TRAIL_ATR_MULT": 0.8,
    # exposure
    "ALLOW_SINGLE_HEDGE": False,
    "MAX_CONCURRENT_POS": 1,
    # cadence
    "ENTRY_COOLDOWN_SEC": 10,
}
