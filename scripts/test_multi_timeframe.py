#!/usr/bin/env python3
"""Multi-timeframe strategy evaluation with epsilon tolerance + missed trades log.

Tests all 3 strategies across M5, M15, M30, H1 using existing optimized params.
Outputs a comprehensive report with:
  - OOS metrics (Sharpe, PF, MaxDD) with epsilon tolerance
  - Missed trades analysis per strategy
  - Trades/week frequency

Usage:
    python scripts/test_multi_timeframe.py
    python scripts/test_multi_timeframe.py --epsilon 0.05
    python scripts/test_multi_timeframe.py --timeframes 15m 1h
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from src.data.fetcher import load_data
from src.features.engineering import prepare_strategy_data
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.strategies.base import MissedTrade
from src.validation.pipeline import chronological_split, compute_metrics

SPREAD_PIPS = 1.0

# OOS criteria
MIN_SHARPE = 1.2
MIN_PF = 1.5
MAX_DD = 15.0

# Default strategy params from previous optimization (from multi_timeframe_results.json)
DEFAULT_TREND_PARAMS = {
    "ema_fast": 12, "ema_slow": 37,
    "adx_threshold": 16.24, "atr_sl_mult": 2.73, "atr_tp_mult": 1.73,
}
DEFAULT_MR_PARAMS = {
    "rsi_period": 14, "rsi_oversold": 38.0, "rsi_overbought": 70.0,
    "bb_period": 20, "bb_std": 2.0, "atr_sl_mult": 2.0, "atr_tp_mult": 2.5,
    "min_bb_width": 0.01, "use_stochastic": False, "use_zscore": False,
    "zscore_threshold": 2.0, "max_adx": 100.0, "long_only": False, "require_bb": True,
}
DEFAULT_VOL_PARAMS = {
    "squeeze_lookback": 5, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0,
    "vol_ratio_threshold": 0.8, "bb_period": 20, "kc_period": 20,
    "bb_mult": 2.0, "kc_mult": 1.5,
}

SYMBOLS = ["sp500", "nasdaq", "dowjones", "dax"]
TIMEFRAMES = ["5m", "15m", "30m", "1h"]


def load_optimized_params(results_file: Path) -> dict:
    """Load best params from previous optimization results if available."""
    if not results_file.exists():
        return {}
    with open(results_file) as f:
        return json.load(f)


def build_strategies(params_override: dict | None = None):
    """Build all 3 strategies with given or default params."""
    tp = params_override.get("trend", DEFAULT_TREND_PARAMS) if params_override else DEFAULT_TREND_PARAMS
    mp = params_override.get("mean_reversion", DEFAULT_MR_PARAMS) if params_override else DEFAULT_MR_PARAMS
    vp = params_override.get("volatility", DEFAULT_VOL_PARAMS) if params_override else DEFAULT_VOL_PARAMS

    trend = TrendFollowingStrategy(
        ema_fast=tp.get("ema_fast", 12), ema_slow=tp.get("ema_slow", 37),
        adx_threshold=tp.get("adx_threshold", 16.24),
        atr_sl_mult=tp.get("atr_sl_mult", 2.73), atr_tp_mult=tp.get("atr_tp_mult", 1.73),
        spread_pips=SPREAD_PIPS,
    )
    mr = MeanReversionStrategy(
        rsi_period=mp.get("rsi_period", 14),
        rsi_oversold=mp.get("rsi_oversold", 38.0),
        rsi_overbought=mp.get("rsi_overbought", 70.0),
        bb_period=mp.get("bb_period", 20), bb_std=mp.get("bb_std", 2.0),
        atr_sl_mult=mp.get("atr_sl_mult", 2.0), atr_tp_mult=mp.get("atr_tp_mult", 2.5),
        min_bb_width=mp.get("min_bb_width", 0.01),
        use_stochastic=mp.get("use_stochastic", False),
        use_zscore=mp.get("use_zscore", False),
        zscore_threshold=mp.get("zscore_threshold", 2.0),
        max_adx=mp.get("max_adx", 100.0),
        long_only=mp.get("long_only", False),
        require_bb=mp.get("require_bb", True),
        spread_pips=SPREAD_PIPS,
    )
    vol = VolatilityBreakoutStrategy(
        squeeze_lookback=vp.get("squeeze_lookback", 5),
        atr_sl_mult=vp.get("atr_sl_mult", 1.5),
        atr_tp_mult=vp.get("atr_tp_mult", 3.0),
        vol_ratio_threshold=vp.get("vol_ratio_threshold", 0.8),
        spread_pips=SPREAD_PIPS,
    )

    return {
        "trend": (trend, "trend", {"ema_fast": tp.get("ema_fast", 12), "ema_slow": tp.get("ema_slow", 37)}),
        "mean_reversion": (mr, "mean_reversion", {"rsi_period": mp.get("rsi_period", 14), "bb_period": mp.get("bb_period", 20), "bb_std": mp.get("bb_std", 2.0), "max_adx": mp.get("max_adx", 100.0)}),
        "volatility": (vol, "volatility", {"bb_period": vp.get("bb_period", 20), "kc_period": vp.get("kc_period", 20), "bb_mult": vp.get("bb_mult", 2.0), "kc_mult": vp.get("kc_mult", 1.5)}),
    }


def evaluate_strategy(strategy, df_raw, strat_type, feat_params, epsilon, detect_missed=True):
    """Evaluate a strategy on the test set with epsilon tolerance + missed trades."""
    df = prepare_strategy_data(df_raw, strat_type, feat_params)
    train, val, test = chronological_split(df)

    n_test_days = len(set(test.index.date)) if hasattr(test.index, "date") else len(test)

    # Backtest on test set
    trades, equity = strategy.backtest(test)
    rets = strategy.get_trade_returns(trades)
    m = compute_metrics(rets, n_trading_days=n_test_days)

    n_weeks = n_test_days / 5.0
    tpw = m.total_trades / max(n_weeks, 0.1)

    meets_strict = m.meets_criteria(MIN_SHARPE, MIN_PF, MAX_DD, epsilon=0.0)
    meets_eps = m.meets_criteria(MIN_SHARPE, MIN_PF, MAX_DD, epsilon=epsilon)

    result = {
        "sharpe": m.sharpe_ratio,
        "pf": m.profit_factor,
        "maxdd": m.max_drawdown_pct,
        "trades": m.total_trades,
        "trades_per_week": tpw,
        "win_rate": m.win_rate,
        "annual_return": m.annual_return,
        "sortino": m.sortino_ratio,
        "meets_strict": meets_strict,
        "meets_epsilon": meets_eps,
        "test_days": n_test_days,
        "test_bars": len(test),
    }

    # Detect missed trades
    if detect_missed:
        signals = strategy.generate_signals(df)
        test_signals = signals.iloc[len(train) + len(val):]
        missed = strategy.detect_missed_trades(
            test, test_signals,
            lookahead_bars=min(20, len(test) // 10),
            min_move_pct=0.005,
        )

        # Summarize missed trades
        if missed:
            reason_counts = Counter(mt.reason for mt in missed)
            top_reasons = reason_counts.most_common(5)
            avg_move = np.mean([mt.peak_favorable_move_pct for mt in missed])
            result["missed_trades"] = len(missed)
            result["missed_avg_move_pct"] = float(avg_move)
            result["missed_top_reasons"] = [
                {"reason": r, "count": c} for r, c in top_reasons
            ]
        else:
            result["missed_trades"] = 0
            result["missed_avg_move_pct"] = 0.0
            result["missed_top_reasons"] = []

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-timeframe strategy evaluation")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon tolerance (default 0.05 = 5%%)")
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES, help="Timeframes to test")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS, help="Symbols to test")
    parser.add_argument("--no-missed", action="store_true", help="Skip missed trades detection (faster)")
    args = parser.parse_args()

    # Try to load optimized params
    results_file = Path("reports/multi_timeframe_results.json")
    opt_results = load_optimized_params(results_file)

    all_results = {}

    for tf in args.timeframes:
        print(f"\n{'='*80}")
        print(f"TIMEFRAME: {tf}")
        print(f"{'='*80}")

        all_results[tf] = {}

        for sym in args.symbols:
            try:
                df_raw = load_data(sym, tf)
            except FileNotFoundError:
                print(f"  {sym}: No data for {tf}")
                continue

            print(f"\n  {sym} ({len(df_raw)} bars)")

            # Try to get optimized params for this tf+symbol
            sym_params = None
            if tf in opt_results and sym in opt_results[tf]:
                sym_params = {}
                for strat_name in ["trend", "mean_reversion", "volatility"]:
                    if strat_name in opt_results[tf][sym]:
                        sym_params[strat_name] = opt_results[tf][sym][strat_name].get("params", {})

            strategies = build_strategies(sym_params)
            all_results[tf][sym] = {}

            for strat_name, (strategy, strat_type, feat_params) in strategies.items():
                try:
                    res = evaluate_strategy(
                        strategy, df_raw, strat_type, feat_params,
                        epsilon=args.epsilon,
                        detect_missed=not args.no_missed,
                    )
                    all_results[tf][sym][strat_name] = res

                    # Format output
                    strict_flag = " [STRICT]" if res["meets_strict"] else ""
                    eps_flag = f" [eps={args.epsilon}]" if res["meets_epsilon"] and not res["meets_strict"] else ""
                    missed_info = ""
                    if "missed_trades" in res and res["missed_trades"] > 0:
                        missed_info = f", Missed={res['missed_trades']} (avg {res['missed_avg_move_pct']:.2f}%)"

                    print(
                        f"    {strat_name:20s}: Sharpe={res['sharpe']:6.2f}, PF={res['pf']:5.2f}, "
                        f"MaxDD={res['maxdd']:5.1f}%, Trades={res['trades']:5d}, "
                        f"T/wk={res['trades_per_week']:5.1f}, WR={res['win_rate']:4.1f}%"
                        f"{strict_flag}{eps_flag}{missed_info}"
                    )

                    # Print top missed trade reasons
                    if res.get("missed_top_reasons"):
                        for reason_info in res["missed_top_reasons"][:3]:
                            print(f"      Missed reason: {reason_info['reason']} ({reason_info['count']}x)")

                except Exception as e:
                    print(f"    {strat_name:20s}: FAILED - {e}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY (epsilon={args.epsilon})")
    print(f"{'='*80}")
    print(f"\nOOS Criteria: Sharpe >= {MIN_SHARPE}, PF >= {MIN_PF}, MaxDD <= {MAX_DD}%")
    print(f"With epsilon={args.epsilon}: Sharpe >= {MIN_SHARPE*(1-args.epsilon):.2f}, "
          f"PF >= {MIN_PF*(1-args.epsilon):.2f}, MaxDD <= {MAX_DD*(1+args.epsilon):.1f}%")
    print(f"Target: ~10 trades/week\n")

    meets_strict_list = []
    meets_eps_list = []

    for tf in args.timeframes:
        if tf not in all_results:
            continue
        for sym in args.symbols:
            if sym not in all_results.get(tf, {}):
                continue
            for strat_name, res in all_results[tf][sym].items():
                entry = f"{strat_name}@{tf}/{sym}"
                if res.get("meets_strict"):
                    meets_strict_list.append((entry, res))
                elif res.get("meets_epsilon"):
                    meets_eps_list.append((entry, res))

    if meets_strict_list:
        print("MEETS STRICT CRITERIA:")
        for entry, r in meets_strict_list:
            print(f"  {entry:40s}: Sharpe={r['sharpe']:.2f}, PF={r['pf']:.2f}, "
                  f"MaxDD={r['maxdd']:.1f}%, T/wk={r['trades_per_week']:.1f}")

    if meets_eps_list:
        print(f"\nMEETS WITH EPSILON={args.epsilon} TOLERANCE:")
        for entry, r in meets_eps_list:
            print(f"  {entry:40s}: Sharpe={r['sharpe']:.2f}, PF={r['pf']:.2f}, "
                  f"MaxDD={r['maxdd']:.1f}%, T/wk={r['trades_per_week']:.1f}")

    if not meets_strict_list and not meets_eps_list:
        print("NO strategies meet criteria (even with epsilon tolerance).")
        print("\nClosest to meeting criteria:")
        best = []
        for tf in args.timeframes:
            if tf not in all_results:
                continue
            for sym in args.symbols:
                if sym not in all_results.get(tf, {}):
                    continue
                for strat_name, res in all_results[tf][sym].items():
                    score = res["sharpe"] - max(0, res["maxdd"] - MAX_DD) * 0.5
                    best.append((score, f"{strat_name}@{tf}/{sym}", res))
        best.sort(reverse=True)
        for score, entry, r in best[:10]:
            print(f"  {entry:40s}: Sharpe={r['sharpe']:.2f}, PF={r['pf']:.2f}, "
                  f"MaxDD={r['maxdd']:.1f}%, T/wk={r['trades_per_week']:.1f}")

    # Missed trades summary
    print(f"\n{'='*80}")
    print("MISSED TRADES SUMMARY")
    print(f"{'='*80}")
    for tf in args.timeframes:
        if tf not in all_results:
            continue
        for sym in args.symbols:
            if sym not in all_results.get(tf, {}):
                continue
            for strat_name, res in all_results[tf][sym].items():
                missed = res.get("missed_trades", 0)
                if missed > 0:
                    print(f"  {strat_name}@{tf}/{sym}: {missed} missed trades "
                          f"(avg move {res['missed_avg_move_pct']:.2f}%)")
                    for reason_info in res.get("missed_top_reasons", [])[:3]:
                        print(f"    -> {reason_info['reason']} ({reason_info['count']}x)")

    # Save results
    out_path = Path("reports/evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
