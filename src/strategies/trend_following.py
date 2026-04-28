"""Strategy A: Trend Following (EMA Cross + ADX Filter + MACD Confirmation)."""

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """Trend Following strategy using EMA crossovers.

    Entry rules:
      LONG:  EMA_fast > EMA_slow AND ADX > threshold AND MACD_hist > 0
      SHORT: EMA_fast < EMA_slow AND ADX > threshold AND MACD_hist < 0

    Exit rules:
      - ATR-based stop loss and take profit
      - Opposite signal closes position

    Anti-bias:
      - All signals computed from previous bar data
      - Entry at open of NEXT bar after signal
    """

    def __init__(
        self,
        ema_fast: int = 12,
        ema_slow: int = 50,
        adx_threshold: float = 25.0,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.0,
        spread_pips: float = 1.0,
    ):
        super().__init__("TrendFollowing", spread_pips)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_threshold = adx_threshold
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)

        if "ema_fast" not in df.columns:
            return signals

        long_cond = (
            (df["ema_fast"] > df["ema_slow"])
            & (df["adx"] > self.adx_threshold)
            & (df["macd_hist"] > 0)
        )
        short_cond = (
            (df["ema_fast"] < df["ema_slow"])
            & (df["adx"] > self.adx_threshold)
            & (df["macd_hist"] < 0)
        )

        signals[long_cond] = 1
        signals[short_cond] = -1

        return signals

    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        atr = df["atr_14"].iloc[idx] if "atr_14" in df.columns else df["close"].iloc[idx] * 0.01
        price = df["close"].iloc[idx]
        return price - direction * self.atr_sl_mult * atr

    def get_take_profit(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        atr = df["atr_14"].iloc[idx] if "atr_14" in df.columns else df["close"].iloc[idx] * 0.01
        price = df["close"].iloc[idx]
        return price + direction * self.atr_tp_mult * atr

    def get_params(self) -> dict:
        return {
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "adx_threshold": self.adx_threshold,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
        }
