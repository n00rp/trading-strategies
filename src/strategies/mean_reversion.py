"""Strategy B: Mean Reversion (RSI Dip-Buying for Indices)."""

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion strategy optimized for equity indices.

    Uses RSI as the primary signal with optional BB confirmation.
    Designed for daily timeframe where extreme signals are rare,
    so thresholds are relaxed compared to textbook values.

    Long entry: RSI < threshold (dip buy in uptrending market)
    Short entry: RSI > threshold (optional, disabled by default)
    Exit: ATR-based take profit and stop loss

    The edge: indices tend to mean-revert after short-term oversold
    conditions due to institutional dip-buying and long-term upward bias.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 70.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 2.0,
        min_bb_width: float = 0.01,
        use_stochastic: bool = False,
        use_zscore: bool = False,
        zscore_threshold: float = 2.0,
        max_adx: float = 100.0,
        long_only: bool = True,
        require_bb: bool = False,
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
        self.use_stochastic = use_stochastic
        self.use_zscore = use_zscore
        self.zscore_threshold = zscore_threshold
        self.max_adx = max_adx
        self.long_only = long_only
        self.require_bb = require_bb

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)

        # Primary long signal
        long_cond = pd.Series(False, index=df.index)

        if self.use_zscore and "zscore" in df.columns:
            long_cond = df["zscore"] < -self.zscore_threshold
        elif "rsi" in df.columns:
            long_cond = df["rsi"] < self.rsi_oversold

            if self.require_bb and "bb_lower" in df.columns:
                if "bb_pct" in df.columns:
                    long_cond = long_cond & (df["bb_pct"] < 0.2)
                else:
                    long_cond = long_cond & (df["close"] <= df["bb_lower"] * 1.02)

        # Short signal (optional)
        short_cond = pd.Series(False, index=df.index)
        if not self.long_only:
            if self.use_zscore and "zscore" in df.columns:
                short_cond = df["zscore"] > self.zscore_threshold
            elif "rsi" in df.columns:
                short_cond = df["rsi"] > self.rsi_overbought
                if self.require_bb and "bb_upper" in df.columns:
                    short_cond = short_cond & (df["close"] > df["bb_upper"])

        # BB width filter
        if "bb_width" in df.columns:
            width_ok = df["bb_width"] > self.min_bb_width
            long_cond = long_cond & width_ok
            if not self.long_only:
                short_cond = short_cond & width_ok

        # Stochastic confirmation
        if self.use_stochastic and "stoch_k" in df.columns and "stoch_d" in df.columns:
            stoch_long = (df["stoch_k"] < 30) & (df["stoch_k"] > df["stoch_d"])
            long_cond = long_cond & stoch_long
            if not self.long_only:
                stoch_short = (df["stoch_k"] > 70) & (df["stoch_k"] < df["stoch_d"])
                short_cond = short_cond & stoch_short

        # ADX filter (only trade in non-trending)
        if self.max_adx < 100 and "adx" in df.columns:
            no_trend = df["adx"] < self.max_adx
            long_cond = long_cond & no_trend
            if not self.long_only:
                short_cond = short_cond & no_trend

        signals[long_cond] = 1
        if not self.long_only:
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
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
            "min_bb_width": self.min_bb_width,
            "use_stochastic": self.use_stochastic,
            "use_zscore": self.use_zscore,
            "zscore_threshold": self.zscore_threshold,
            "max_adx": self.max_adx,
            "long_only": self.long_only,
            "require_bb": self.require_bb,
        }
