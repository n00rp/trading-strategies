#!/usr/bin/env python3
"""Train Model C: Volatility/Breakout RL Agent.

Standalone training script — run locally for full training.

Usage:
    python scripts/train_model_C.py [--timesteps 500000] [--symbol sp500]

Output:
    models/model_C_volatility_breakout.zip (SB3 PPO model)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

CONFIG = {
    "symbol": "sp500",
    "interval": "1d",
    "strategy_type": "volatility",

    "feature_params": {
        "bb_period": 20,
        "kc_period": 20,
        "bb_mult": 2.0,
        "kc_mult": 1.5,
    },

    "algorithm": "PPO",
    "total_timesteps": 500_000,
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_steps": 2048,
    "gamma": 0.99,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,

    "initial_capital": 100_000,
    "spread_pips": 1.0,
    "risk_per_trade": 0.02,
    "reward_type": "sharpe",

    "model_dir": "models",
    "model_name": "model_C_volatility_breakout",
}


def main():
    parser = argparse.ArgumentParser(description="Train Model C: Volatility Breakout")
    parser.add_argument("--timesteps", type=int, default=CONFIG["total_timesteps"])
    parser.add_argument("--symbol", type=str, default=CONFIG["symbol"])
    parser.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    args = parser.parse_args()

    CONFIG["total_timesteps"] = args.timesteps
    CONFIG["symbol"] = args.symbol
    CONFIG["learning_rate"] = args.lr

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

    from src.data.fetcher import load_data
    from src.features.engineering import prepare_strategy_data
    from src.environments.trading_env import TradingEnv
    from src.validation.pipeline import chronological_split

    logger.info(f"Loading {CONFIG['symbol']} data...")
    df_raw = load_data(CONFIG["symbol"], CONFIG["interval"])
    df = prepare_strategy_data(df_raw, CONFIG["strategy_type"], CONFIG["feature_params"])
    logger.info(f"  {len(df)} bars after feature engineering")

    train, val, test = chronological_split(df)

    exclude_cols = {"open", "high", "low", "close", "volume", "symbol", "returns", "log_returns"}
    feature_cols = [c for c in train.columns if c not in exclude_cols]
    logger.info(f"  Features ({len(feature_cols)}): {feature_cols[:10]}...")

    train_env = TradingEnv(
        df=train,
        feature_columns=feature_cols,
        initial_capital=CONFIG["initial_capital"],
        spread_pips=CONFIG["spread_pips"],
        risk_per_trade=CONFIG["risk_per_trade"],
        reward_type=CONFIG["reward_type"],
    )

    eval_env = TradingEnv(
        df=val,
        feature_columns=feature_cols,
        initial_capital=CONFIG["initial_capital"],
        spread_pips=CONFIG["spread_pips"],
    )

    model_dir = Path(CONFIG["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=20, verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "logs"),
        eval_freq=5000,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
    )

    logger.info("\nCreating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=CONFIG["learning_rate"],
        batch_size=CONFIG["batch_size"],
        n_steps=CONFIG["n_steps"],
        gamma=CONFIG["gamma"],
        ent_coef=CONFIG["ent_coef"],
        clip_range=CONFIG["clip_range"],
        n_epochs=CONFIG["n_epochs"],
        gae_lambda=CONFIG["gae_lambda"],
        max_grad_norm=CONFIG["max_grad_norm"],
        vf_coef=CONFIG["vf_coef"],
        verbose=1,
        seed=42,
    )

    logger.info(f"\nTraining for {CONFIG['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=eval_callback,
        progress_bar=True,
    )

    save_path = model_dir / CONFIG["model_name"]
    model.save(str(save_path))
    logger.info(f"\nModel saved to {save_path}.zip")

    # Quick test
    logger.info("\nRunning test evaluation...")
    test_env = TradingEnv(
        df=test,
        feature_columns=feature_cols,
        initial_capital=CONFIG["initial_capital"],
        spread_pips=CONFIG["spread_pips"],
    )

    obs, info = test_env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        done = terminated or truncated

    logger.info(f"  Test total reward: {total_reward:.4f}")
    logger.info(f"  Test trades: {info['total_trades']}")
    logger.info(f"  Test final capital: {info['capital']:.2f}")
    logger.info("\n✓ Training complete!")


if __name__ == "__main__":
    main()
