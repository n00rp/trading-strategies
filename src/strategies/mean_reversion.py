"""Strategy B: Mean Reversion (RSI Extremes + Bollinger Band Reversals)."""

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion strategy using RSI + Bollinger Bands.

    Entry rules:
      LONG:  RSI < oversold AND price < BB_lower (bounce expected)
      SHORT: RSI > overbought AND price > BB_upper (pullback expected)

    Filters:
      - Stochastic confirmation (K crosses D from extreme)
      - BB width filter (avoid very tight bands = trending market)

    Exit rules:
      - Price returns to BB midline (mean)
      - ATR-based stop loss
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_sl_mult: float = 1.0,
        atr_tp_mult: float = 2.0,
        min_bb_width: float = 0.02,
        spread_pips: float = 1.0,
    ):
        super().__init__("MeanReversion", spread_pips)
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.min_bb_width = min_bb_width

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)

        if "rsi" not in df.columns or "bb_lower" not in df.columns:
            return signals

        # Oversold bounce (long)
        long_cond = (
            (df["rsi"] < self.rsi_oversold)
            & (df["close"] < df["bb_lower"])
            & (df["bb_width"] > self.min_bb_width)
        )

        # Overbought reversal (short)
        short_cond = (
            (df["rsi"] > self.rsi_overbought)
            & (df["close"] > df["bb_upper"])
            & (df["bb_width"] > self.min_bb_width)
        )

        # Optional: stochastic confirmation
        if "stoch_k" in df.columns and "stoch_d" in df.columns:
            stoch_long = (df["stoch_k"] < 20) & (df["stoch_k"] > df["stoch_d"])
            stoch_short = (df["stoch_k"] > 80) & (df["stoch_k"] < df["stoch_d"])
            long_cond = long_cond & stoch_long
            short_cond = short_cond & stoch_short

        signals[long_cond] = 1
        signals[short_cond] = -1

        return signals

    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        atr = df["atr_14"].iloc[idx] if "atr_14" in df.columns else df["close"].iloc[idx] * 0.01
        price = df["close"].iloc[idx]
        return price - direction * self.atr_sl_mult * atr

    def get_take_profit(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        # Mean reversion: target is the BB midline
        if "bb_mid" in df.columns:
            return df["bb_mid"].iloc[idx]
        atr = df["atr_14"].iloc[idx] if "atr_14" in df.columns else df["close"].iloc[idx] * 0.01
        price = df["close"].iloc[idx]
        return price + direction * self.atr_tp_mult * atr

    def get_params(self) -> dict:
        return {
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
            "min_bb_width": self.min_bb_width,
        }
