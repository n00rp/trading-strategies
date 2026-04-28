#!/usr/bin/env python3
"""Multi-timeframe strategy optimization across M5, M10, M15, M30, H1.

Run this on your local machine for best performance.
Requires MT5 data exported via the MQL5 ExportData script (see mt5/ExportData.mq5).

Usage:
    python scripts/run_multi_timeframe.py --trials 200
    python scripts/run_multi_timeframe.py --trials 500 --timeframes 15m 1h
    python scripts/run_multi_timeframe.py --trials 200 --symbols sp500 nasdaq
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

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
from src.validation.pipeline import chronological_split, compute_metrics


SPREAD_PIPS = 1.0

# Target: ~10 trades/week on the test set
TARGET_TRADES_PER_WEEK = 10


def evaluate_on_test(strategy, df_raw, strategy_type, feat_params):
    """Backtest a strategy on the OOS test set with corrected metrics."""
    df = prepare_strategy_data(df_raw, strategy_type, feat_params)
    train, val, test = chronological_split(df)

    if hasattr(test.index, "date"):
        n_test_days = len(set(test.index.date))
    else:
        n_test_days = len(test)

    trades, equity = strategy.backtest(test)
    rets = strategy.get_trade_returns(trades)
    m = compute_metrics(rets, n_trading_days=n_test_days)

    n_weeks = n_test_days / 5.0
    tpw = m.total_trades / max(n_weeks, 0.1)

    return {
        "sharpe": m.sharpe_ratio,
        "pf": m.profit_factor,
        "maxdd": m.max_drawdown_pct,
        "trades": m.total_trades,
        "trades_per_week": tpw,
        "win_rate": m.win_rate,
        "annual_return": m.annual_return,
        "meets_criteria": m.meets_criteria(),
        "test_days": n_test_days,
        "test_bars": len(test),
    }


def build_trend_strategy(params):
    return TrendFollowingStrategy(
        ema_fast=params["ema_fast"],
        ema_slow=params["ema_slow"],
        adx_threshold=params["adx_threshold"],
        atr_sl_mult=params["atr_sl_mult"],
        atr_tp_mult=params["atr_tp_mult"],
        spread_pips=SPREAD_PIPS,
    )


def build_mr_strategy(params):
    return MeanReversionStrategy(
        rsi_period=params.get("rsi_period", 14),
        rsi_oversold=params.get("rsi_oversold", 35.0),
        rsi_overbought=params.get("rsi_overbought", 70.0),
        bb_period=params["bb_period"],
        bb_std=params["bb_std"],
        atr_sl_mult=params["atr_sl_mult"],
        atr_tp_mult=params["atr_tp_mult"],
        min_bb_width=params["min_bb_width"],
        use_stochastic=params.get("use_stochastic", False),
        use_zscore=params.get("use_zscore", False),
        zscore_threshold=params.get("zscore_threshold", 2.0),
        max_adx=params.get("max_adx", 100.0),
        long_only=params.get("long_only", True),
        require_bb=params.get("require_bb", False),
        spread_pips=SPREAD_PIPS,
    )


def build_vol_strategy(params):
    return VolatilityBreakoutStrategy(
        squeeze_lookback=params["squeeze_lookback"],
        atr_sl_mult=params["atr_sl_mult"],
        atr_tp_mult=params["atr_tp_mult"],
        vol_ratio_threshold=params["vol_ratio_threshold"],
        spread_pips=SPREAD_PIPS,
    )


def run_optimization(timeframe, symbol, n_trials):
    """Run full optimization for all 3 strategies on one timeframe + symbol."""
    try:
        df_raw = load_data(symbol, timeframe)
    except FileNotFoundError:
        print(f"  No data for {symbol} @ {timeframe}. Run MT5 ExportData first.")
        return None

    print(f"  Loaded {len(df_raw)} bars for {symbol} @ {timeframe}")

    if len(df_raw) < 500:
        print(f"  Too few bars ({len(df_raw)}), skipping")
        return None

    results = {}

    # ═══ Strategy A: Trend Following ═══
    print(f"  [A] Trend Following ({n_trials} trials)...")
    try:
        opt = optimize_trend_following(df_raw, n_trials=n_trials, spread_pips=SPREAD_PIPS)
        params = opt["best_params"]
        strat = build_trend_strategy(params)
        feat = {"ema_fast": params["ema_fast"], "ema_slow": params["ema_slow"]}
        test_res = evaluate_on_test(strat, df_raw, "trend", feat)
        test_res["params"] = params
        test_res["val_score"] = opt["best_score"]
        results["trend"] = test_res
        flag = " <<< MEETS >>>" if test_res["meets_criteria"] else ""
        print(
            f"    -> Sharpe={test_res['sharpe']:.2f}, PF={test_res['pf']:.2f}, "
            f"MaxDD={test_res['maxdd']:.1f}%, Trades={test_res['trades']}, "
            f"T/wk={test_res['trades_per_week']:.1f}, "
            f"AnnRet={test_res['annual_return']:.1f}%{flag}"
        )
    except Exception as e:
        print(f"    FAILED: {e}")

    # ═══ Strategy B: Mean Reversion ═══
    print(f"  [B] Mean Reversion ({n_trials} trials)...")
    try:
        opt = optimize_mean_reversion(df_raw, n_trials=n_trials, spread_pips=SPREAD_PIPS)
        params = opt["best_params"]
        strat = build_mr_strategy(params)
        feat = {
            "rsi_period": params.get("rsi_period", 14),
            "bb_period": params["bb_period"],
            "bb_std": params["bb_std"],
        }
        test_res = evaluate_on_test(strat, df_raw, "mean_reversion", feat)
        test_res["params"] = params
        test_res["val_score"] = opt["best_score"]
        results["mean_reversion"] = test_res
        flag = " <<< MEETS >>>" if test_res["meets_criteria"] else ""
        print(
            f"    -> Sharpe={test_res['sharpe']:.2f}, PF={test_res['pf']:.2f}, "
            f"MaxDD={test_res['maxdd']:.1f}%, Trades={test_res['trades']}, "
            f"T/wk={test_res['trades_per_week']:.1f}, "
            f"AnnRet={test_res['annual_return']:.1f}%{flag}"
        )
    except Exception as e:
        print(f"    FAILED: {e}")

    # ═══ Strategy C: Volatility Breakout ═══
    print(f"  [C] Volatility Breakout ({n_trials} trials)...")
    try:
        opt = optimize_volatility_breakout(df_raw, n_trials=n_trials, spread_pips=SPREAD_PIPS)
        params = opt["best_params"]
        strat = build_vol_strategy(params)
        feat = {
            "bb_period": params["bb_period"],
            "kc_period": params["kc_period"],
            "bb_mult": params["bb_mult"],
            "kc_mult": params["kc_mult"],
        }
        test_res = evaluate_on_test(strat, df_raw, "volatility", feat)
        test_res["params"] = params
        test_res["val_score"] = opt["best_score"]
        results["volatility"] = test_res
        flag = " <<< MEETS >>>" if test_res["meets_criteria"] else ""
        print(
            f"    -> Sharpe={test_res['sharpe']:.2f}, PF={test_res['pf']:.2f}, "
            f"MaxDD={test_res['maxdd']:.1f}%, Trades={test_res['trades']}, "
            f"T/wk={test_res['trades_per_week']:.1f}, "
            f"AnnRet={test_res['annual_return']:.1f}%{flag}"
        )
    except Exception as e:
        print(f"    FAILED: {e}")

    return results


def cross_index_validation(timeframe, n_trials, symbols):
    """Optimize on sp500, then validate across all indices."""
    print(f"\n{'='*70}")
    print(f"CROSS-INDEX VALIDATION @ {timeframe}")
    print(f"{'='*70}")

    # Optimize on S&P 500
    print(f"\nPhase 1: Optimize on sp500 ({n_trials} trials)...")
    sp500_results = run_optimization(timeframe, "sp500", n_trials)
    if not sp500_results:
        return None

    # Validate on other indices using same params
    print(f"\nPhase 2: Cross-index validation...")
    all_results = {"sp500": sp500_results}

    for sym in symbols:
        if sym == "sp500":
            continue
        try:
            df_raw = load_data(sym, timeframe)
        except FileNotFoundError:
            print(f"  No data for {sym} @ {timeframe}")
            continue

        print(f"\n  Validating on {sym} ({len(df_raw)} bars)...")
        sym_results = {}

        for strat_name in ["trend", "mean_reversion", "volatility"]:
            if strat_name not in sp500_results:
                continue
            params = sp500_results[strat_name]["params"]

            if strat_name == "trend":
                strat = build_trend_strategy(params)
                feat = {"ema_fast": params["ema_fast"], "ema_slow": params["ema_slow"]}
            elif strat_name == "mean_reversion":
                strat = build_mr_strategy(params)
                feat = {
                    "rsi_period": params.get("rsi_period", 14),
                    "bb_period": params["bb_period"],
                    "bb_std": params["bb_std"],
                }
            else:
                strat = build_vol_strategy(params)
                feat = {
                    "bb_period": params["bb_period"],
                    "kc_period": params["kc_period"],
                    "bb_mult": params["bb_mult"],
                    "kc_mult": params["kc_mult"],
                }

            try:
                test_res = evaluate_on_test(strat, df_raw, strat_name, feat)
                test_res["params"] = params
                sym_results[strat_name] = test_res
                flag = " <<< MEETS >>>" if test_res["meets_criteria"] else ""
                print(
                    f"    [{strat_name[:5].upper():>5s}] Sharpe={test_res['sharpe']:.2f}, "
                    f"PF={test_res['pf']:.2f}, MaxDD={test_res['maxdd']:.1f}%, "
                    f"Trades={test_res['trades']}, T/wk={test_res['trades_per_week']:.1f}{flag}"
                )
            except Exception as e:
                print(f"    [{strat_name[:5].upper():>5s}] FAILED: {e}")

        all_results[sym] = sym_results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-timeframe strategy optimization")
    parser.add_argument("--trials", type=int, default=200, help="Optuna trials per strategy")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["5m", "10m", "15m", "30m", "1h"],
        help="Timeframes to test",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["sp500", "nasdaq", "dowjones", "dax"],
        help="Symbols to validate on",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run cross-index validation (optimize on sp500, validate on all)",
    )
    args = parser.parse_args()

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    all_results = {}

    if args.cross_validate:
        for tf in args.timeframes:
            result = cross_index_validation(tf, args.trials, args.symbols)
            if result:
                all_results[tf] = result
    else:
        for tf in args.timeframes:
            print(f"\n{'='*70}")
            print(f"TIMEFRAME: {tf}")
            print(f"{'='*70}")
            result = run_optimization(tf, "sp500", args.trials)
            if result:
                all_results[tf] = result

    # ═══ Final summary ═══
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nOOS Criteria: Sharpe > 1.2, PF > 1.5, MaxDD < 15%")
    print(f"Target: ~{TARGET_TRADES_PER_WEEK} trades/week\n")

    for strat_name in ["trend", "mean_reversion", "volatility"]:
        print(f"  {strat_name.upper()}:")
        for tf in args.timeframes:
            if tf not in all_results:
                continue

            if args.cross_validate:
                # Show per-index results
                for sym in args.symbols:
                    if sym in all_results[tf] and strat_name in all_results[tf][sym]:
                        r = all_results[tf][sym][strat_name]
                        flag = " <<< MEETS >>>" if r["meets_criteria"] else ""
                        print(
                            f"    {tf:>4s} {sym:>10s}: Sharpe={r['sharpe']:6.2f}, "
                            f"PF={r['pf']:6.2f}, MaxDD={r['maxdd']:5.1f}%, "
                            f"Trades={r['trades']:5d}, T/wk={r['trades_per_week']:6.1f}, "
                            f"AnnRet={r['annual_return']:6.1f}%{flag}"
                        )
            else:
                if strat_name in all_results[tf]:
                    r = all_results[tf][strat_name]
                    flag = " <<< MEETS >>>" if r["meets_criteria"] else ""
                    print(
                        f"    {tf:>4s}: Sharpe={r['sharpe']:6.2f}, PF={r['pf']:6.2f}, "
                        f"MaxDD={r['maxdd']:5.1f}%, Trades={r['trades']:5d}, "
                        f"T/wk={r['trades_per_week']:6.1f}, "
                        f"AnnRet={r['annual_return']:6.1f}%{flag}"
                    )
        print()

    # Save results
    out_path = out_dir / "multi_timeframe_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
