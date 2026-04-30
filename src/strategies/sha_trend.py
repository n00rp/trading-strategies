"""Strategy D: Smoothed Heiken Ashi Trend (SHA + HA dual-timeframe).

Uses Anchor Timeframe (ATF) for trend detection via SHA,
and Execution Timeframe (ETF) for entry via HA pullback + reversal.

Long entry:
    1. ATF SHA is bullish (green, no lower wick)
    2. ETF HA shows pullback (red candle)
    3. ETF HA reverses (green candle, no lower wick, price > SHA close)

Short entry:
    1. ATF SHA is bearish (red, no upper wick)
    2. ETF HA shows pullback (green candle)
    3. ETF HA reverses (red candle, no upper wick, price < SHA close)
"""

import logging

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, Trade, MissedTrade
from src.features.heiken_ashi import (
    compute_heiken_ashi,
    compute_smoothed_heiken_ashi,
    is_bullish,
    is_bearish,
    has_no_lower_wick,
    has_no_upper_wick,
)

logger = logging.getLogger(__name__)


class SHATrendStrategy(BaseStrategy):
    """Smoothed Heiken Ashi dual-timeframe trend strategy.

    Parameters:
        sha_period: EMA period for SHA smoothing (optimize 6-14, default 10)
        atf_interval: Anchor timeframe interval string (e.g. '1h')
        etf_interval: Execution timeframe interval string (e.g. '5m')
        epsilon: Tolerance for wick comparisons (in price units)
        epsilon_pct: Tolerance as percentage of price (used if epsilon=0)
        require_atf_wick: Require ATF SHA candle to have no wick
        atr_sl_factor: ATR multiplier for stop loss (default 0.5)
        rrr: Risk-reward ratio for take profit (default 2.0)
        trailing_exit: Use SHA color change for trailing exit
    """

    def __init__(
        self,
        sha_period: int = 10,
        atf_interval: str = "1h",
        etf_interval: str = "5m",
        epsilon: float = 0.0,
        epsilon_pct: float = 0.0001,
        require_atf_wick: bool = True,
        atr_sl_factor: float = 0.5,
        rrr: float = 2.0,
        trailing_exit: bool = True,
        spread_pips: float = 1.0,
    ):
        super().__init__("SHATrend", spread_pips)
        self.sha_period = sha_period
        self.atf_interval = atf_interval
        self.etf_interval = etf_interval
        self.epsilon = epsilon
        self.epsilon_pct = epsilon_pct
        self.require_atf_wick = require_atf_wick
        self.atr_sl_factor = atr_sl_factor
        self.rrr = rrr
        self.trailing_exit = trailing_exit

    def _get_epsilon(self, price: float) -> float:
        """Get epsilon tolerance in price units."""
        if self.epsilon > 0:
            return self.epsilon
        return price * self.epsilon_pct

    def prepare_data(
        self, df_etf: pd.DataFrame, df_atf: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare ETF data with HA/SHA indicators and ATF trend alignment.

        Args:
            df_etf: Execution timeframe OHLCV data
            df_atf: Anchor timeframe OHLCV data

        Returns:
            ETF DataFrame augmented with HA, SHA, ATF trend columns
        """
        import ta

        # Compute HA on ETF
        df_etf = compute_heiken_ashi(df_etf)

        # Compute SHA on ETF
        df_etf = compute_smoothed_heiken_ashi(df_etf, period=self.sha_period)

        # Compute SHA on ATF
        df_atf = compute_smoothed_heiken_ashi(df_atf, period=self.sha_period)

        # ATF trend: bullish/bearish + wick strength
        df_atf = df_atf.copy()
        df_atf["atf_bullish"] = df_atf["sha_close"] > df_atf["sha_open"]
        df_atf["atf_bearish"] = df_atf["sha_close"] < df_atf["sha_open"]

        # Wick check on ATF (vectorized with epsilon)
        # For bullish: no lower wick means L_SHA >= O_SHA - eps
        # For bearish: no upper wick means H_SHA <= O_SHA + eps
        atf_prices = df_atf["sha_open"].values
        atf_eps = np.where(
            self.epsilon > 0,
            self.epsilon,
            atf_prices * self.epsilon_pct,
        )
        df_atf["atf_strong_bull"] = (
            df_atf["atf_bullish"]
            & (df_atf["sha_low"].values >= df_atf["sha_open"].values - atf_eps)
        )
        df_atf["atf_strong_bear"] = (
            df_atf["atf_bearish"]
            & (df_atf["sha_high"].values <= df_atf["sha_open"].values + atf_eps)
        )

        # Map ATF trend to ETF bars using forward-fill
        # Each ATF bar's trend applies to all ETF bars until the next ATF bar
        atf_trend = df_atf[["atf_bullish", "atf_bearish",
                            "atf_strong_bull", "atf_strong_bear"]].copy()

        # Reindex ATF to ETF timestamps
        df_etf = df_etf.copy()
        atf_reindexed = atf_trend.reindex(df_etf.index, method="ffill")
        for col in atf_trend.columns:
            df_etf[col] = atf_reindexed[col].fillna(False)

        # ATR on ETF
        df_etf["atr_14"] = ta.volatility.average_true_range(
            df_etf["high"], df_etf["low"], df_etf["close"], window=14
        )

        # Swing low/high (last 5 bars)
        df_etf["swing_low_5"] = df_etf["low"].rolling(5).min()
        df_etf["swing_high_5"] = df_etf["high"].rolling(5).max()

        df_etf = df_etf.dropna()
        return df_etf

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals from prepared dual-timeframe data.

        Expects df to already have: ha_*, sha_*, atf_* columns
        (from prepare_data).
        """
        signals = pd.Series(0, index=df.index)

        required = ["ha_open", "ha_close", "ha_low", "ha_high",
                     "sha_open", "sha_close", "atf_bullish", "atf_bearish"]
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column {col} — returning no signals")
                return signals

        ha_o = df["ha_open"].values
        ha_c = df["ha_close"].values
        ha_l = df["ha_low"].values
        ha_h = df["ha_high"].values
        sha_c = df["sha_close"].values
        raw_c = df["close"].values

        atf_bull = df["atf_bullish"].values if "atf_bullish" in df.columns else np.zeros(len(df), dtype=bool)
        atf_bear = df["atf_bearish"].values if "atf_bearish" in df.columns else np.zeros(len(df), dtype=bool)
        atf_strong_bull = df["atf_strong_bull"].values if "atf_strong_bull" in df.columns else atf_bull
        atf_strong_bear = df["atf_strong_bear"].values if "atf_strong_bear" in df.columns else atf_bear

        for i in range(2, len(df)):
            eps = self._get_epsilon(ha_o[i])

            # --- LONG ---
            trend_ok = atf_strong_bull[i] if self.require_atf_wick else atf_bull[i]
            if trend_ok:
                # Setup: previous HA was red (bearish pullback)
                prev_was_red = ha_c[i - 1] < ha_o[i - 1]
                # Trigger: current HA is green + no lower wick + price > SHA close
                curr_is_green = ha_c[i] > ha_o[i]
                no_lower_wick = ha_l[i] >= ha_o[i] - eps
                price_above_sha = raw_c[i] > sha_c[i]

                if prev_was_red and curr_is_green and no_lower_wick and price_above_sha:
                    signals.iloc[i] = 1
                    continue

            # --- SHORT ---
            trend_ok = atf_strong_bear[i] if self.require_atf_wick else atf_bear[i]
            if trend_ok:
                # Setup: previous HA was green (bullish pullback)
                prev_was_green = ha_c[i - 1] > ha_o[i - 1]
                # Trigger: current HA is red + no upper wick + price < SHA close
                curr_is_red = ha_c[i] < ha_o[i]
                no_upper_wick = ha_h[i] <= ha_o[i] + eps
                price_below_sha = raw_c[i] < sha_c[i]

                if prev_was_green and curr_is_red and no_upper_wick and price_below_sha:
                    signals.iloc[i] = -1

        return signals

    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        """SL based on SHA + ATR, or swing low/high."""
        atr = df["atr_14"].iloc[idx] if "atr_14" in df.columns else df["close"].iloc[idx] * 0.01

        if direction == 1:
            # Long: SL = SHA_Low - ATR * factor
            sha_low = df["sha_low"].iloc[idx] if "sha_low" in df.columns else df["low"].iloc[idx]
            sl_sha = sha_low - atr * self.atr_sl_factor
            # Alternative: swing low
            sl_swing = df["swing_low_5"].iloc[idx] if "swing_low_5" in df.columns else sl_sha
            return min(sl_sha, sl_swing)
        else:
            # Short: SL = SHA_High + ATR * factor
            sha_high = df["sha_high"].iloc[idx] if "sha_high" in df.columns else df["high"].iloc[idx]
            sl_sha = sha_high + atr * self.atr_sl_factor
            sl_swing = df["swing_high_5"].iloc[idx] if "swing_high_5" in df.columns else sl_sha
            return max(sl_sha, sl_swing)

    def get_take_profit(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        """TP based on RRR (risk-reward ratio)."""
        entry_price = df["close"].iloc[idx]
        sl = self.get_stop_loss(df, idx, direction)
        risk = abs(entry_price - sl)
        return entry_price + direction * risk * self.rrr

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000,
        risk_per_trade: float = 0.02,
    ) -> tuple[list[Trade], np.ndarray]:
        """Run backtest with trailing exit on SHA color change.

        Overrides base backtest to add:
        - Partial close at RRR 1:1 (50% position)
        - Move SL to break-even after partial close
        - Full exit on SHA color change (trailing)
        """
        signals = self.generate_signals(df)
        df = df.copy()
        df["signal"] = signals

        trades: list[Trade] = []
        equity = initial_capital
        equity_curve = [equity]
        position = 0
        entry_price = 0.0
        entry_time = None
        stop_loss = 0.0
        take_profit = 0.0
        entry_idx = 0
        partial_closed = False
        original_risk = 0.0

        min_sl_pct = 0.003  # minimum SL distance: 0.3% to prevent excessive leverage

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            if position != 0:
                # Check SL/TP first (priority over trailing)
                hit_sl = (
                    (position == 1 and row["low"] <= stop_loss)
                    or (position == -1 and row["high"] >= stop_loss)
                )
                hit_tp = (
                    (position == 1 and row["high"] >= take_profit)
                    or (position == -1 and row["low"] <= take_profit)
                )

                # Check partial close at RRR 1:1
                if not partial_closed and original_risk > 0:
                    partial_level = entry_price + position * original_risk
                    hit_partial = (
                        (position == 1 and row["high"] >= partial_level)
                        or (position == -1 and row["low"] <= partial_level)
                    )
                    if hit_partial:
                        partial_closed = True
                        stop_loss = entry_price  # move SL to break-even

                # Trailing exit: SHA color change, only AFTER partial close
                sha_exit = False
                if self.trailing_exit and partial_closed:
                    if "sha_open" in df.columns and "sha_close" in df.columns:
                        if position == 1:
                            sha_exit = row["sha_close"] < row["sha_open"]
                        else:
                            sha_exit = row["sha_close"] > row["sha_open"]

                if hit_sl or hit_tp or sha_exit:
                    if hit_sl:
                        exit_price = stop_loss
                    elif hit_tp:
                        exit_price = take_profit
                    else:
                        exit_price = row["close"]

                    spread_cost = self.spread_pips * 0.01
                    exit_price_adj = exit_price - position * spread_cost

                    pnl_pct = position * (exit_price_adj - entry_price) / entry_price
                    sl_dist = max(
                        abs(entry_price - stop_loss) / entry_price,
                        min_sl_pct,
                    )
                    pnl = equity * risk_per_trade * pnl_pct / sl_dist if sl_dist > 0 else 0

                    equity += pnl
                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=row.name,
                        direction=position,
                        entry_price=entry_price,
                        exit_price=exit_price_adj,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        bars_held=i - entry_idx,
                    ))
                    position = 0
                    partial_closed = False

            # New signal
            sig = prev_row["signal"]
            if position == 0 and sig != 0:
                position = int(sig)
                entry_price = row["open"]
                spread_cost = self.spread_pips * 0.01
                entry_price += position * spread_cost

                entry_time = row.name
                entry_idx = i
                stop_loss = self.get_stop_loss(df, i - 1, position)
                take_profit = self.get_take_profit(df, i - 1, position)
                # Enforce minimum SL distance
                raw_risk = abs(entry_price - stop_loss)
                min_risk = entry_price * min_sl_pct
                if raw_risk < min_risk:
                    if position == 1:
                        stop_loss = entry_price - min_risk
                    else:
                        stop_loss = entry_price + min_risk
                    take_profit = entry_price + position * min_risk * self.rrr
                original_risk = abs(entry_price - stop_loss)
                partial_closed = False

            equity_curve.append(equity)

        return trades, np.array(equity_curve)

    def _diagnose_no_signal(
        self, df: pd.DataFrame, idx: int, direction: int
    ) -> str:
        """Diagnose why SHA trend strategy didn't signal at this bar."""
        reasons = []
        eps = self._get_epsilon(df["ha_open"].iloc[idx] if "ha_open" in df.columns else df["close"].iloc[idx])

        if direction == 1:
            if "atf_bullish" in df.columns and not df["atf_bullish"].iloc[idx]:
                reasons.append("atf_not_bullish")
            elif self.require_atf_wick and "atf_strong_bull" in df.columns and not df["atf_strong_bull"].iloc[idx]:
                reasons.append("atf_no_strong_bull_wick")
            if "ha_close" in df.columns and "ha_open" in df.columns and idx > 0:
                if df["ha_close"].iloc[idx - 1] >= df["ha_open"].iloc[idx - 1]:
                    reasons.append("prev_not_red_pullback")
                if df["ha_close"].iloc[idx] <= df["ha_open"].iloc[idx]:
                    reasons.append("curr_not_green")
                elif df["ha_low"].iloc[idx] < df["ha_open"].iloc[idx] - eps:
                    reasons.append("has_lower_wick")
            if "sha_close" in df.columns and df["close"].iloc[idx] <= df["sha_close"].iloc[idx]:
                reasons.append("price<=sha_close")
        else:
            if "atf_bearish" in df.columns and not df["atf_bearish"].iloc[idx]:
                reasons.append("atf_not_bearish")
            elif self.require_atf_wick and "atf_strong_bear" in df.columns and not df["atf_strong_bear"].iloc[idx]:
                reasons.append("atf_no_strong_bear_wick")
            if "ha_close" in df.columns and "ha_open" in df.columns and idx > 0:
                if df["ha_close"].iloc[idx - 1] <= df["ha_open"].iloc[idx - 1]:
                    reasons.append("prev_not_green_pullback")
                if df["ha_close"].iloc[idx] >= df["ha_open"].iloc[idx]:
                    reasons.append("curr_not_red")
                elif df["ha_high"].iloc[idx] > df["ha_open"].iloc[idx] + eps:
                    reasons.append("has_upper_wick")
            if "sha_close" in df.columns and df["close"].iloc[idx] >= df["sha_close"].iloc[idx]:
                reasons.append("price>=sha_close")

        return "; ".join(reasons) if reasons else "unknown"

    def get_params(self) -> dict:
        return {
            "sha_period": self.sha_period,
            "atf_interval": self.atf_interval,
            "etf_interval": self.etf_interval,
            "epsilon": self.epsilon,
            "epsilon_pct": self.epsilon_pct,
            "require_atf_wick": self.require_atf_wick,
            "atr_sl_factor": self.atr_sl_factor,
            "rrr": self.rrr,
            "trailing_exit": self.trailing_exit,
        }
