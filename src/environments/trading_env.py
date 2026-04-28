"""Gymnasium-compatible trading environment for RL-based strategy training.

Used with Stable Baselines3 (PPO/A2C/SAC) for learning optimal
position sizing and entry/exit timing.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnv(gym.Env):
    """Trading environment for reinforcement learning.

    Observation: Feature vector from the strategy's feature engineering pipeline.
    Action: Discrete(3) -> 0=hold, 1=buy, 2=sell
    Reward: Risk-adjusted PnL per step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        initial_capital: float = 100_000,
        spread_pips: float = 1.0,
        risk_per_trade: float = 0.02,
        max_position: int = 1,
        reward_type: str = "sharpe",
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_capital = initial_capital
        self.spread_pips = spread_pips
        self.risk_per_trade = risk_per_trade
        self.max_position = max_position
        self.reward_type = reward_type

        self.n_features = len(feature_columns)
        self.n_steps = len(df)

        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # Observation space: features + position + unrealized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features + 2,),
            dtype=np.float32,
        )

        # State
        self.current_step = 0
        self.position = 0  # -1, 0, 1
        self.entry_price = 0.0
        self.capital = initial_capital
        self.equity_curve: list[float] = []
        self.trade_returns: list[float] = []

    def _get_obs(self) -> np.ndarray:
        features = self.df[self.feature_columns].iloc[self.current_step].values.astype(np.float32)
        # Append position and unrealized PnL
        if self.position != 0:
            current_price = self.df["close"].iloc[self.current_step]
            unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = 0.0

        return np.concatenate([features, [float(self.position), unrealized_pnl]]).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {
            "step": self.current_step,
            "capital": self.capital,
            "position": self.position,
            "total_trades": len(self.trade_returns),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.capital = self.initial_capital
        self.equity_curve = [self.capital]
        self.trade_returns = []
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        reward = 0.0
        current_price = self.df["close"].iloc[self.current_step]
        spread_cost = self.spread_pips * 0.01

        # Execute action
        if action == 1 and self.position <= 0:  # Buy
            if self.position == -1:  # Close short first
                pnl_pct = -1 * (current_price + spread_cost - self.entry_price) / self.entry_price
                self.trade_returns.append(pnl_pct)
                self.capital *= (1 + pnl_pct * self.risk_per_trade)
            self.position = 1
            self.entry_price = current_price + spread_cost

        elif action == 2 and self.position >= 0:  # Sell
            if self.position == 1:  # Close long first
                pnl_pct = (current_price - spread_cost - self.entry_price) / self.entry_price
                self.trade_returns.append(pnl_pct)
                self.capital *= (1 + pnl_pct * self.risk_per_trade)
            self.position = -1
            self.entry_price = current_price - spread_cost

        # Compute reward
        if self.reward_type == "pnl" and self.position != 0:
            next_price = self.df["close"].iloc[min(self.current_step + 1, self.n_steps - 1)]
            reward = self.position * (next_price - current_price) / current_price
        elif self.reward_type == "sharpe" and len(self.trade_returns) >= 2:
            returns = np.array(self.trade_returns[-20:])
            reward = np.mean(returns) / (np.std(returns) + 1e-8)

        self.equity_curve.append(self.capital)
        self.current_step += 1

        terminated = self.current_step >= self.n_steps - 1
        truncated = self.capital <= self.initial_capital * 0.5  # 50% drawdown = stop

        if truncated:
            reward -= 10.0  # large penalty for blowing up

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()


def make_trading_env(
    df: pd.DataFrame,
    feature_columns: list[str],
    **kwargs,
) -> TradingEnv:
    """Factory function for creating TradingEnv instances."""
    return TradingEnv(df=df, feature_columns=feature_columns, **kwargs)
