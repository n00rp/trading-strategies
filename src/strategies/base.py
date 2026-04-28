"""Base strategy class and backtesting engine."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Represents a single trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_pct: float
    bars_held: int


class BaseStrategy(ABC):
    """Abstract base for rule-based strategies."""

    def __init__(self, name: str, spread_pips: float = 1.0):
        self.name = name
        self.spread_pips = spread_pips

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals: 1=buy, -1=sell, 0=hold.

        CRITICAL: Must only use data available at time t to generate signal at t.
        No look-ahead allowed.
        """

    @abstractmethod
    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        """Get stop loss price for a trade."""

    @abstractmethod
    def get_take_profit(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        """Get take profit price for a trade."""

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000,
        risk_per_trade: float = 0.02,
    ) -> tuple[list[Trade], np.ndarray]:
        """Run backtest on prepared data.

        Returns:
            trades: List of Trade objects
            equity_curve: Numpy array of equity values
        """
        signals = self.generate_signals(df)
        df = df.copy()
        df["signal"] = signals

        trades: list[Trade] = []
        equity = initial_capital
        equity_curve = [equity]
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0.0
        entry_time = None
        stop_loss = 0.0
        take_profit = 0.0
        entry_idx = 0

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # Check stop loss / take profit if in position
            if position != 0:
                hit_sl = (
                    (position == 1 and row["low"] <= stop_loss)
                    or (position == -1 and row["high"] >= stop_loss)
                )
                hit_tp = (
                    (position == 1 and row["high"] >= take_profit)
                    or (position == -1 and row["low"] <= take_profit)
                )

                if hit_sl or hit_tp:
                    exit_price = stop_loss if hit_sl else take_profit
                    # Apply spread on exit
                    spread_cost = self.spread_pips * 0.01  # rough pip to price
                    exit_price_adj = exit_price - position * spread_cost

                    pnl_pct = position * (exit_price_adj - entry_price) / entry_price
                    pnl = equity * risk_per_trade * pnl_pct / abs(
                        (entry_price - stop_loss) / entry_price
                    ) if abs(entry_price - stop_loss) > 0 else 0

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

            # New signal — enter position if flat
            # Use PREVIOUS bar's signal to avoid look-ahead
            sig = prev_row["signal"]
            if position == 0 and sig != 0:
                position = int(sig)
                entry_price = row["open"]  # enter at open of current bar
                # Apply spread on entry
                spread_cost = self.spread_pips * 0.01
                entry_price += position * spread_cost

                entry_time = row.name
                entry_idx = i
                stop_loss = self.get_stop_loss(df, i - 1, position)
                take_profit = self.get_take_profit(df, i - 1, position)

            equity_curve.append(equity)

        return trades, np.array(equity_curve)

    def get_trade_returns(self, trades: list[Trade]) -> np.ndarray:
        """Extract return array from trades."""
        return np.array([t.pnl_pct for t in trades])
