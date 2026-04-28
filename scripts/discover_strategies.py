#!/usr/bin/env python3
"""Strategy discovery and optimization pipeline.

Runs Optuna optimization for all 3 strategy types, validates on OOS data,
and performs walk-forward analysis + Monte Carlo stress tests.

Usage:
    python scripts/discover_strategies.py [--trials 200] [--symbol sp500]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetcher import load_data
from src.features.engineering import prepare_strategy_data
from src.optimization.optuna_optimizer import (
    optimize_trend_following,
    optimize_mean_reversion,
    optimize_volatility_breakout,
)
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.validation.pipeline import (
    chronological_split,
    compute_metrics,
    walk_forward_analysis,
    monte_carlo_test,
    check_overfitting,
    check_look_ahead_bias,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def validate_strategy(
    name: str,
    strategy,
    df_raw: pd.DataFrame,
    strategy_type: str,
    feature_params: dict,
) -> dict:
    """Run full validation pipeline on a strategy."""
    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATING: {name}")
    logger.info(f"{'='*60}")

    df = prepare_strategy_data(df_raw, strategy_type, feature_params)
    train, val, test = chronological_split(df)

    # Backtest on each split
    trades_train, eq_train = strategy.backtest(train)
    trades_val, eq_val = strategy.backtest(val)
    trades_test, eq_test = strategy.backtest(test)

    rets_train = strategy.get_trade_returns(trades_train)
    rets_val = strategy.get_trade_returns(trades_val)
    rets_test = strategy.get_trade_returns(trades_test)

    metrics_train = compute_metrics(rets_train)
    metrics_val = compute_metrics(rets_val)
    metrics_test = compute_metrics(rets_test)

    logger.info(f"  TRAIN: {metrics_train.summary()}")
    logger.info(f"  VAL:   {metrics_val.summary()}")
    logger.info(f"  TEST:  {metrics_test.summary()}")

    # Walk-Forward Analysis
    def wf_fn(train_df, test_df):
        _, _ = strategy.backtest(train_df)
        trades, _ = strategy.backtest(test_df)
        return strategy.get_trade_returns(trades)

    wf_results = walk_forward_analysis(df, wf_fn, n_splits=5)
    wf_sharpes = [r.sharpe_ratio for r in wf_results]
    logger.info(f"  Walk-Forward Sharpes: {wf_sharpes}")

    # Monte Carlo
    mc_results = monte_carlo_test(rets_test, n_simulations=1000)
    logger.info(f"  Monte Carlo: median_sharpe={mc_results['median_sharpe']:.2f}, "
                f"p95_maxdd={mc_results['p95_max_dd']:.1f}%, "
                f"prob_profitable={mc_results['prob_profitable']:.1f}%")

    # Overfitting check
    overfit = check_overfitting(metrics_train, metrics_val, metrics_test)
    logger.info(f"  Overfitting: {overfit['verdict']} "
                f"(val_deg={overfit['val_degradation_pct']:.1f}%, "
                f"test_deg={overfit['test_degradation_pct']:.1f}%)")

    # Look-ahead bias check
    signal_col = {
        "trend": "ema_cross",
        "mean_reversion": "mr_signal",
        "volatility": "breakout_signal",
    }.get(strategy_type, "signal")

    if signal_col in df.columns:
        lab = check_look_ahead_bias(df, signal_col)
        logger.info(f"  Look-Ahead: {lab['verdict']} "
                    f"(future_corr={lab['future_return_correlation']:.4f})")
    else:
        lab = {"verdict": "N/A"}

    # OOS criteria check
    meets_criteria = metrics_test.meets_criteria(
        min_sharpe=1.2, min_pf=1.5, max_dd=15.0
    )
    logger.info(f"  MEETS OOS CRITERIA: {'YES' if meets_criteria else 'NO'}")

    return {
        "name": name,
        "strategy_type": strategy_type,
        "params": strategy.get_params(),
        "metrics": {
            "train": metrics_train.__dict__,
            "val": metrics_val.__dict__,
            "test": metrics_test.__dict__,
        },
        "walk_forward": {
            "sharpes": wf_sharpes,
            "mean_sharpe": float(np.mean(wf_sharpes)),
        },
        "monte_carlo": mc_results,
        "overfitting": overfit,
        "look_ahead": lab,
        "meets_criteria": meets_criteria,
    }


def main():
    parser = argparse.ArgumentParser(description="Strategy Discovery Pipeline")
    parser.add_argument("--trials", type=int, default=200, help="Optuna trials per strategy")
    parser.add_argument("--symbol", type=str, default="sp500", help="Symbol to optimize on")
    parser.add_argument("--spread", type=float, default=1.0, help="Spread in pips")
    args = parser.parse_args()

    import pandas as pd

    # Load data
    logger.info(f"Loading {args.symbol} daily data...")
    df_raw = load_data(args.symbol, "1d")
    logger.info(f"  Loaded {len(df_raw)} bars")

    results = {}

    # ═══════════════════════════════════════════════════
    # Strategy A: Trend Following
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZING STRATEGY A: TREND FOLLOWING")
    logger.info("=" * 60)

    opt_a = optimize_trend_following(df_raw, n_trials=args.trials, spread_pips=args.spread)
    best_a = opt_a["best_params"]

    strategy_a = TrendFollowingStrategy(
        ema_fast=best_a["ema_fast"],
        ema_slow=best_a["ema_slow"],
        adx_threshold=best_a["adx_threshold"],
        atr_sl_mult=best_a["atr_sl_mult"],
        atr_tp_mult=best_a["atr_tp_mult"],
        spread_pips=args.spread,
    )
    results["trend_following"] = validate_strategy(
        "Trend Following (EMA Cross + ADX)",
        strategy_a, df_raw, "trend",
        {"ema_fast": best_a["ema_fast"], "ema_slow": best_a["ema_slow"]},
    )

    # ═══════════════════════════════════════════════════
    # Strategy B: Mean Reversion
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZING STRATEGY B: MEAN REVERSION")
    logger.info("=" * 60)

    opt_b = optimize_mean_reversion(df_raw, n_trials=args.trials, spread_pips=args.spread)
    best_b = opt_b["best_params"]

    strategy_b = MeanReversionStrategy(
        rsi_period=best_b["rsi_period"],
        rsi_oversold=best_b["rsi_oversold"],
        rsi_overbought=best_b["rsi_overbought"],
        bb_period=best_b["bb_period"],
        bb_std=best_b["bb_std"],
        atr_sl_mult=best_b["atr_sl_mult"],
        atr_tp_mult=best_b["atr_tp_mult"],
        min_bb_width=best_b["min_bb_width"],
        spread_pips=args.spread,
    )
    results["mean_reversion"] = validate_strategy(
        "Mean Reversion (RSI + BB)",
        strategy_b, df_raw, "mean_reversion",
        {"rsi_period": best_b["rsi_period"], "bb_period": best_b["bb_period"],
         "bb_std": best_b["bb_std"]},
    )

    # ═══════════════════════════════════════════════════
    # Strategy C: Volatility Breakout
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZING STRATEGY C: VOLATILITY BREAKOUT")
    logger.info("=" * 60)

    opt_c = optimize_volatility_breakout(df_raw, n_trials=args.trials, spread_pips=args.spread)
    best_c = opt_c["best_params"]

    strategy_c = VolatilityBreakoutStrategy(
        squeeze_lookback=best_c["squeeze_lookback"],
        atr_sl_mult=best_c["atr_sl_mult"],
        atr_tp_mult=best_c["atr_tp_mult"],
        vol_ratio_threshold=best_c["vol_ratio_threshold"],
        spread_pips=args.spread,
    )
    results["volatility_breakout"] = validate_strategy(
        "Volatility Breakout (Squeeze + Donchian)",
        strategy_c, df_raw, "volatility",
        {"bb_period": best_c["bb_period"], "kc_period": best_c["kc_period"],
         "bb_mult": best_c["bb_mult"], "kc_mult": best_c["kc_mult"]},
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "discovery_results.json"

    # Make serializable
    serializable = {}
    for k, v in results.items():
        sv = {}
        for k2, v2 in v.items():
            if isinstance(v2, dict):
                sv[k2] = {
                    k3: (float(v3) if isinstance(v3, (np.floating, np.integer)) else v3)
                    for k3, v3 in v2.items()
                    if not callable(v3)
                }
            else:
                sv[k2] = v2
        serializable[k] = sv

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(f"\nResults saved to {results_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DISCOVERY SUMMARY")
    logger.info("=" * 60)
    for name, res in results.items():
        test_m = res["metrics"]["test"]
        logger.info(
            f"  {name}: Sharpe={test_m['sharpe_ratio']:.2f} | "
            f"PF={test_m['profit_factor']:.2f} | "
            f"MaxDD={test_m['max_drawdown_pct']:.1f}% | "
            f"Meets criteria: {res['meets_criteria']}"
        )


if __name__ == "__main__":
    main()
