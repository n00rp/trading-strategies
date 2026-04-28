"""Optuna-based hyperparameter optimization for each strategy.

Optimizes strategy parameters to meet OOS criteria:
  Sharpe > 1.2, PF > 1.5, MaxDD < 15% (with spread/fees)
"""

import logging
from typing import Callable

import numpy as np
import optuna
import pandas as pd

from src.features.engineering import prepare_strategy_data
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.validation.pipeline import (
    chronological_split,
    compute_metrics,
    PerformanceMetrics,
)

logger = logging.getLogger(__name__)


def optimize_trend_following(
    df_raw: pd.DataFrame,
    n_trials: int = 200,
    spread_pips: float = 1.0,
) -> dict:
    """Optimize Strategy A: Trend Following."""

    def objective(trial: optuna.Trial) -> float:
        ema_fast = trial.suggest_int("ema_fast", 8, 21)
        ema_slow = trial.suggest_int("ema_slow", 34, 89)
        adx_threshold = trial.suggest_float("adx_threshold", 15.0, 35.0)
        atr_sl_mult = trial.suggest_float("atr_sl_mult", 1.0, 3.0)
        atr_tp_mult = trial.suggest_float("atr_tp_mult", 1.5, 5.0)

        if ema_fast >= ema_slow:
            return -10.0

        params = {"ema_fast": ema_fast, "ema_slow": ema_slow}
        df = prepare_strategy_data(df_raw, "trend", params)

        train, val, _ = chronological_split(df)

        strategy = TrendFollowingStrategy(
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            adx_threshold=adx_threshold,
            atr_sl_mult=atr_sl_mult,
            atr_tp_mult=atr_tp_mult,
            spread_pips=spread_pips,
        )

        # Backtest on train, evaluate on val
        trades_train, _ = strategy.backtest(train)
        trades_val, _ = strategy.backtest(val)

        rets_train = strategy.get_trade_returns(trades_train)
        rets_val = strategy.get_trade_returns(trades_val)

        if len(rets_val) < 10:
            return -10.0

        # Estimate trading days from index
        if hasattr(val.index, 'date'):
            n_val_days = len(set(val.index.date))
            n_train_days = len(set(train.index.date))
        else:
            n_val_days = len(val)
            n_train_days = len(train)

        metrics_val = compute_metrics(rets_val, n_trading_days=n_val_days)

        # Penalize overfitting
        metrics_train = compute_metrics(rets_train, n_trading_days=n_train_days)
        overfit_penalty = 0
        if metrics_train.sharpe_ratio > 0:
            degradation = (
                (metrics_train.sharpe_ratio - metrics_val.sharpe_ratio)
                / metrics_train.sharpe_ratio
            )
            if degradation > 0.5:
                overfit_penalty = degradation * 2

        # Multi-objective: maximize Sharpe, constrain DD and PF
        score = metrics_val.sharpe_ratio
        if metrics_val.max_drawdown_pct > 15:
            score -= (metrics_val.max_drawdown_pct - 15) * 0.5
        if metrics_val.profit_factor < 1.5:
            score -= (1.5 - metrics_val.profit_factor) * 2

        score -= overfit_penalty

        trial.set_user_attr("val_sharpe", metrics_val.sharpe_ratio)
        trial.set_user_attr("val_pf", metrics_val.profit_factor)
        trial.set_user_attr("val_maxdd", metrics_val.max_drawdown_pct)
        trial.set_user_attr("val_trades", metrics_val.total_trades)

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"Best Trend Following trial: {best.params}")
    logger.info(f"  Val Sharpe={best.user_attrs.get('val_sharpe', 'N/A')}")

    return {
        "best_params": best.params,
        "best_score": best.value,
        "best_attrs": best.user_attrs,
        "study": study,
    }


def optimize_mean_reversion(
    df_raw: pd.DataFrame,
    n_trials: int = 200,
    spread_pips: float = 1.0,
) -> dict:
    """Optimize Strategy B: Mean Reversion."""

    def objective(trial: optuna.Trial) -> float:
        rsi_period = trial.suggest_int("rsi_period", 7, 21)
        rsi_oversold = trial.suggest_float("rsi_oversold", 20.0, 35.0)
        rsi_overbought = trial.suggest_float("rsi_overbought", 65.0, 80.0)
        bb_period = trial.suggest_int("bb_period", 14, 30)
        bb_std = trial.suggest_float("bb_std", 1.5, 2.5)
        atr_sl_mult = trial.suggest_float("atr_sl_mult", 0.5, 2.0)
        atr_tp_mult = trial.suggest_float("atr_tp_mult", 1.0, 3.0)
        min_bb_width = trial.suggest_float("min_bb_width", 0.01, 0.05)

        params = {"rsi_period": rsi_period, "bb_period": bb_period, "bb_std": bb_std}
        df = prepare_strategy_data(df_raw, "mean_reversion", params)

        train, val, _ = chronological_split(df)

        strategy = MeanReversionStrategy(
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            bb_period=bb_period,
            bb_std=bb_std,
            atr_sl_mult=atr_sl_mult,
            atr_tp_mult=atr_tp_mult,
            min_bb_width=min_bb_width,
            spread_pips=spread_pips,
        )

        trades_train, _ = strategy.backtest(train)
        trades_val, _ = strategy.backtest(val)

        rets_train = strategy.get_trade_returns(trades_train)
        rets_val = strategy.get_trade_returns(trades_val)

        if len(rets_val) < 10:
            return -10.0

        if hasattr(val.index, 'date'):
            n_val_days = len(set(val.index.date))
            n_train_days = len(set(train.index.date))
        else:
            n_val_days = len(val)
            n_train_days = len(train)

        metrics_val = compute_metrics(rets_val, n_trading_days=n_val_days)
        metrics_train = compute_metrics(rets_train, n_trading_days=n_train_days)

        overfit_penalty = 0
        if metrics_train.sharpe_ratio > 0:
            degradation = (
                (metrics_train.sharpe_ratio - metrics_val.sharpe_ratio)
                / metrics_train.sharpe_ratio
            )
            if degradation > 0.5:
                overfit_penalty = degradation * 2

        score = metrics_val.sharpe_ratio
        if metrics_val.max_drawdown_pct > 15:
            score -= (metrics_val.max_drawdown_pct - 15) * 0.5
        if metrics_val.profit_factor < 1.5:
            score -= (1.5 - metrics_val.profit_factor) * 2

        score -= overfit_penalty

        trial.set_user_attr("val_sharpe", metrics_val.sharpe_ratio)
        trial.set_user_attr("val_pf", metrics_val.profit_factor)
        trial.set_user_attr("val_maxdd", metrics_val.max_drawdown_pct)
        trial.set_user_attr("val_trades", metrics_val.total_trades)

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"Best Mean Reversion trial: {best.params}")

    return {
        "best_params": best.params,
        "best_score": best.value,
        "best_attrs": best.user_attrs,
        "study": study,
    }


def optimize_volatility_breakout(
    df_raw: pd.DataFrame,
    n_trials: int = 200,
    spread_pips: float = 1.0,
) -> dict:
    """Optimize Strategy C: Volatility Breakout."""

    def objective(trial: optuna.Trial) -> float:
        squeeze_lookback = trial.suggest_int("squeeze_lookback", 3, 10)
        atr_sl_mult = trial.suggest_float("atr_sl_mult", 1.0, 3.0)
        atr_tp_mult = trial.suggest_float("atr_tp_mult", 1.5, 4.0)
        vol_ratio_threshold = trial.suggest_float("vol_ratio_threshold", 0.5, 1.2)
        bb_period = trial.suggest_int("bb_period", 15, 30)
        kc_period = trial.suggest_int("kc_period", 15, 30)
        bb_mult = trial.suggest_float("bb_mult", 1.5, 2.5)
        kc_mult = trial.suggest_float("kc_mult", 1.0, 2.0)

        params = {
            "bb_period": bb_period,
            "kc_period": kc_period,
            "bb_mult": bb_mult,
            "kc_mult": kc_mult,
        }
        df = prepare_strategy_data(df_raw, "volatility", params)

        train, val, _ = chronological_split(df)

        strategy = VolatilityBreakoutStrategy(
            squeeze_lookback=squeeze_lookback,
            atr_sl_mult=atr_sl_mult,
            atr_tp_mult=atr_tp_mult,
            vol_ratio_threshold=vol_ratio_threshold,
            spread_pips=spread_pips,
        )

        trades_train, _ = strategy.backtest(train)
        trades_val, _ = strategy.backtest(val)

        rets_train = strategy.get_trade_returns(trades_train)
        rets_val = strategy.get_trade_returns(trades_val)

        if len(rets_val) < 10:
            return -10.0

        if hasattr(val.index, 'date'):
            n_val_days = len(set(val.index.date))
            n_train_days = len(set(train.index.date))
        else:
            n_val_days = len(val)
            n_train_days = len(train)

        metrics_val = compute_metrics(rets_val, n_trading_days=n_val_days)
        metrics_train = compute_metrics(rets_train, n_trading_days=n_train_days)

        overfit_penalty = 0
        if metrics_train.sharpe_ratio > 0:
            degradation = (
                (metrics_train.sharpe_ratio - metrics_val.sharpe_ratio)
                / metrics_train.sharpe_ratio
            )
            if degradation > 0.5:
                overfit_penalty = degradation * 2

        score = metrics_val.sharpe_ratio
        if metrics_val.max_drawdown_pct > 15:
            score -= (metrics_val.max_drawdown_pct - 15) * 0.5
        if metrics_val.profit_factor < 1.5:
            score -= (1.5 - metrics_val.profit_factor) * 2

        score -= overfit_penalty

        trial.set_user_attr("val_sharpe", metrics_val.sharpe_ratio)
        trial.set_user_attr("val_pf", metrics_val.profit_factor)
        trial.set_user_attr("val_maxdd", metrics_val.max_drawdown_pct)
        trial.set_user_attr("val_trades", metrics_val.total_trades)

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"Best Volatility Breakout trial: {best.params}")

    return {
        "best_params": best.params,
        "best_score": best.value,
        "best_attrs": best.user_attrs,
        "study": study,
    }
