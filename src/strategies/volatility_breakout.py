"""Strategy C: Volatility/Breakout (Squeeze + Donchian Breakout)."""

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility squeeze + breakout strategy.

    Two modes:
      1. SQUEEZE MODE: When BB inside KC detected, wait for squeeze release + breakout
      2. VOLATILITY COMPRESSION MODE: When historical volatility contracts (hvol_20 < hvol_50),
         followed by expansion + Donchian breakout

    Entry rules (either mode):
      LONG:  Volatility compression -> expansion + close > Donchian high (20)
      SHORT: Volatility compression -> expansion + close < Donchian low (20)

    Exit rules:
      - ATR-based stop loss and take profit
    """

    def __init__(
        self,
        squeeze_lookback: int = 5,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 2.5,
        vol_ratio_threshold: float = 0.8,
        require_volume: bool = False,
        use_squeeze: bool = True,
        spread_pips: float = 1.0,
    ):
        super().__init__("VolatilityBreakout", spread_pips)
        self.squeeze_lookback = squeeze_lookback
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.vol_ratio_threshold = vol_ratio_threshold
        self.require_volume = require_volume
        self.use_squeeze = use_squeeze

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)

        if "donchian_high_20" not in df.columns or "donchian_low_20" not in df.columns:
            return signals

        # Breakout direction from Donchian
        breakout_long = df["close"] > df["donchian_high_20"].shift(1)
        breakout_short = df["close"] < df["donchian_low_20"].shift(1)

        # Volatility compression detection (multiple methods)
        compression = pd.Series(False, index=df.index)

        # Method 1: BB inside KC (classic squeeze)
        if self.use_squeeze and "squeeze" in df.columns:
            was_squeezing = df["squeeze"].rolling(self.squeeze_lookback).sum() > 0
            squeeze_released = (df["squeeze"] == 0) & was_squeezing
            compression = compression | squeeze_released

        # Method 2: Historical volatility contraction -> expansion
        if "vol_ratio" in df.columns:
            # Volatility was compressed (low) and now expanding
            was_compressed = (df["vol_ratio"].shift(1) < self.vol_ratio_threshold)
            now_expanding = df["vol_ratio"] >= self.vol_ratio_threshold
            vol_expansion = was_compressed & now_expanding
            compression = compression | vol_expansion

        # Method 3: BB width contraction (below 20-period avg of BB width)
        if "bb_width" not in df.columns and "bb_upper_vol" in df.columns:
            bb_width = (df["bb_upper_vol"] - df["bb_lower_vol"]) / df["close"]
            bb_width_avg = bb_width.rolling(20).mean()
            was_tight = bb_width.shift(1) < bb_width_avg.shift(1) * 0.8
            now_wider = bb_width >= bb_width_avg
            compression = compression | (was_tight & now_wider)

        # If no compression detected at all, fall back to pure Donchian breakout
        # with ATR acceleration filter
        if compression.sum() == 0 and "atr_14" in df.columns:
            atr_sma = df["atr_14"].rolling(20).mean()
            atr_expanding = df["atr_14"] > atr_sma * 1.2  # ATR > 120% of average
            atr_was_low = df["atr_14"].shift(1) < atr_sma.shift(1)
            compression = atr_was_low & atr_expanding

        # Volume filter (optional)
        vol_filter = pd.Series(True, index=df.index)
        if self.require_volume and "volume_ratio" in df.columns:
            vol_filter = df["volume_ratio"] > 1.0

        long_cond = compression & breakout_long & vol_filter
        short_cond = compression & breakout_short & vol_filter

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

    def _diagnose_no_signal(
        self, df: pd.DataFrame, idx: int, direction: int
    ) -> str:
        """Diagnose why volatility breakout didn't signal at this bar."""
        reasons = []
        if "donchian_high_20" in df.columns and direction == 1:
            if df["close"].iloc[idx] <= df["donchian_high_20"].iloc[max(0, idx - 1)]:
                reasons.append("no_breakout_above_donchian")
        if "donchian_low_20" in df.columns and direction == -1:
            if df["close"].iloc[idx] >= df["donchian_low_20"].iloc[max(0, idx - 1)]:
                reasons.append("no_breakout_below_donchian")
        if "squeeze" in df.columns:
            squeeze_sum = df["squeeze"].iloc[max(0, idx - self.squeeze_lookback):idx + 1].sum()
            if squeeze_sum == 0:
                reasons.append("no_squeeze_detected")
        if "vol_ratio" in df.columns:
            vr = df["vol_ratio"].iloc[idx]
            if vr < self.vol_ratio_threshold:
                reasons.append(f"vol_ratio({vr:.2f})<{self.vol_ratio_threshold}")
        return "; ".join(reasons) if reasons else "no_compression"

    def get_params(self) -> dict:
        return {
            "squeeze_lookback": self.squeeze_lookback,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
            "vol_ratio_threshold": self.vol_ratio_threshold,
            "require_volume": self.require_volume,
            "use_squeeze": self.use_squeeze,
        }
