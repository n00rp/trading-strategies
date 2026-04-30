"""Strategy B: Mean Reversion (RSI Extremes + Bollinger Band Reversals)."""

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion strategy using RSI + Bollinger Bands.

    Entry modes:
      1. RSI + BB mode (default): RSI extreme + price outside BB
      2. Z-score mode: Price z-score below/above threshold

    Filters:
      - Optional stochastic confirmation
      - BB width filter
      - Max ADX filter (avoid strong trends)
      - Long-only mode

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
        use_stochastic: bool = False,
        use_zscore: bool = False,
        zscore_threshold: float = 2.0,
        max_adx: float = 100.0,
        long_only: bool = False,
        require_bb: bool = True,
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

        if "rsi" not in df.columns:
            return signals

        # BB width filter
        bb_ok = pd.Series(True, index=df.index)
        if "bb_width" in df.columns:
            bb_ok = df["bb_width"] > self.min_bb_width

        # ADX filter (avoid strong trending markets)
        adx_ok = pd.Series(True, index=df.index)
        if "adx" in df.columns and self.max_adx < 100.0:
            adx_ok = df["adx"] < self.max_adx

        if self.use_zscore and "zscore" in df.columns:
            # Z-score mode
            long_cond = (
                (df["zscore"] < -self.zscore_threshold)
                & bb_ok & adx_ok
            )
            short_cond = (
                (df["zscore"] > self.zscore_threshold)
                & bb_ok & adx_ok
            )
        else:
            # Classic RSI + BB mode
            long_cond = (df["rsi"] < self.rsi_oversold) & bb_ok & adx_ok
            if self.require_bb and "bb_lower" in df.columns:
                long_cond = long_cond & (df["close"] < df["bb_lower"])

            short_cond = (df["rsi"] > self.rsi_overbought) & bb_ok & adx_ok
            if self.require_bb and "bb_upper" in df.columns:
                short_cond = short_cond & (df["close"] > df["bb_upper"])

        # Optional stochastic confirmation
        if self.use_stochastic and "stoch_k" in df.columns and "stoch_d" in df.columns:
            stoch_long = (df["stoch_k"] < 20) & (df["stoch_k"] > df["stoch_d"])
            stoch_short = (df["stoch_k"] > 80) & (df["stoch_k"] < df["stoch_d"])
            long_cond = long_cond & stoch_long
            short_cond = short_cond & stoch_short

        signals[long_cond] = 1
        if not self.long_only:
            signals[short_cond] = -1

        return signals

    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        atr = df["atr_14"].iloc[idx] if "atr_14" in df.columns else df["close"].iloc[idx] * 0.01
        price = df["close"].iloc[idx]
        return price - direction * self.atr_sl_mult * atr

    def get_take_profit(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        if "bb_mid" in df.columns:
            return df["bb_mid"].iloc[idx]
        atr = df["atr_14"].iloc[idx] if "atr_14" in df.columns else df["close"].iloc[idx] * 0.01
        price = df["close"].iloc[idx]
        return price + direction * self.atr_tp_mult * atr

    def _diagnose_no_signal(
        self, df: pd.DataFrame, idx: int, direction: int
    ) -> str:
        """Diagnose why mean reversion didn't signal at this bar."""
        reasons = []
        if self.use_zscore and "zscore" in df.columns:
            zs = df["zscore"].iloc[idx]
            if direction == 1 and zs >= -self.zscore_threshold:
                reasons.append(f"zscore({zs:.2f})>=-{self.zscore_threshold}")
            elif direction == -1 and zs <= self.zscore_threshold:
                reasons.append(f"zscore({zs:.2f})<={self.zscore_threshold}")
        else:
            if "rsi" in df.columns:
                rsi = df["rsi"].iloc[idx]
                if direction == 1 and rsi >= self.rsi_oversold:
                    reasons.append(f"rsi({rsi:.1f})>={self.rsi_oversold}")
                elif direction == -1 and rsi <= self.rsi_overbought:
                    reasons.append(f"rsi({rsi:.1f})<={self.rsi_overbought}")
            if self.require_bb and "bb_lower" in df.columns and "bb_upper" in df.columns:
                close = df["close"].iloc[idx]
                if direction == 1 and close >= df["bb_lower"].iloc[idx]:
                    reasons.append("close>=bb_lower")
                elif direction == -1 and close <= df["bb_upper"].iloc[idx]:
                    reasons.append("close<=bb_upper")
        if self.long_only and direction == -1:
            reasons.append("long_only_mode")
        if "bb_width" in df.columns:
            bw = df["bb_width"].iloc[idx]
            if bw <= self.min_bb_width:
                reasons.append(f"bb_width({bw:.4f})<={self.min_bb_width}")
        if "adx" in df.columns and self.max_adx < 100.0:
            adx_val = df["adx"].iloc[idx]
            if adx_val >= self.max_adx:
                reasons.append(f"adx({adx_val:.1f})>={self.max_adx}")
        return "; ".join(reasons) if reasons else "unknown"

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
