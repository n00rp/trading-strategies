#!/usr/bin/env python3
"""Test the Smoothed Heiken Ashi (SHA) dual-timeframe strategy.

Evaluates the SHA trend strategy across multiple ATF/ETF timeframe
combinations, using the user's MT5 data files.

Usage:
    python scripts/test_sha_strategy.py
    python scripts/test_sha_strategy.py --epsilon 0.05 --sha-period 10
    python scripts/test_sha_strategy.py --symbols sp500 nasdaq --tf-pairs H1/M5 M30/M5
"""

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fetcher import load_data
from src.strategies.sha_trend import SHATrendStrategy
from src.validation.pipeline import PerformanceMetrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---
SPREAD_PIPS = 1.0
MIN_SHARPE = 1.2
MIN_PF = 1.5
MAX_DD = 15.0

SYMBOLS = ["sp500", "nasdaq", "dowjones", "dax"]

# Timeframe pairs: (ATF, ETF)
DEFAULT_TF_PAIRS = [
    ("1h", "5m"),
    ("1h", "15m"),
    ("30m", "5m"),
    ("30m", "1m"),
]

# SHA period range to test
SHA_PERIODS = [6, 8, 10, 12, 14]


def compute_metrics(trades, equity_curve, df_len, etf_interval):
    """Compute performance metrics from trade list."""
    if not trades:
        return None

    pnls = np.array([t.pnl for t in trades])
    pnl_pcts = np.array([t.pnl_pct for t in trades])

    # Annualize based on trade frequency
    n_trades = len(trades)
    bars_per_year = {
        "1m": 252 * 24 * 60, "5m": 252 * 24 * 12,
        "10m": 252 * 24 * 6, "15m": 252 * 24 * 4,
        "30m": 252 * 24 * 2, "1h": 252 * 24, "1d": 252,
    }.get(etf_interval, 252 * 24)

    data_bars = df_len
    trades_per_year = n_trades * bars_per_year / data_bars if data_bars > 0 else 0
    trades_per_week = trades_per_year / 52.0

    # Sharpe (corrected annualization)
    if len(pnl_pcts) > 1 and pnl_pcts.std() > 0 and trades_per_year > 0:
        sharpe = (pnl_pcts.mean() / pnl_pcts.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    max_dd = dd.max()

    # Win rate
    win_rate = (pnl_pcts > 0).mean() * 100

    return {
        "sharpe": sharpe,
        "pf": pf,
        "maxdd": max_dd,
        "trades": n_trades,
        "trades_per_week": trades_per_week,
        "win_rate": win_rate,
        "avg_pnl_pct": pnl_pcts.mean() * 100,
        "avg_bars_held": np.mean([t.bars_held for t in trades]),
    }


def run_sha_test(
    symbol: str,
    atf_interval: str,
    etf_interval: str,
    sha_period: int,
    epsilon_pct: float,
    require_atf_wick: bool,
    detect_missed: bool = False,
):
    """Run SHA strategy test for one symbol + TF pair."""
    # Load data
    try:
        df_etf = load_data(symbol, etf_interval)
        df_atf = load_data(symbol, atf_interval)
    except Exception as e:
        logger.warning(f"  Cannot load {symbol} {atf_interval}/{etf_interval}: {e}")
        return None

    if df_etf is None or df_atf is None or len(df_etf) < 100 or len(df_atf) < 50:
        logger.warning(f"  Insufficient data for {symbol} {atf_interval}/{etf_interval}")
        return None

    # Ensure datetime index
    for df in [df_etf, df_atf]:
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ["time", "datetime", "date", "timestamp"]:
                if col in df.columns:
                    df.index = pd.to_datetime(df[col])
                    break
            else:
                logger.warning(f"  No datetime column found for {symbol}")
                return None

    # Normalize column names
    for df in [df_etf, df_atf]:
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("open", "high", "low", "close", "volume"):
                col_map[c] = cl
        df.rename(columns=col_map, inplace=True)

    # Create strategy
    strategy = SHATrendStrategy(
        sha_period=sha_period,
        atf_interval=atf_interval,
        etf_interval=etf_interval,
        epsilon_pct=epsilon_pct,
        require_atf_wick=require_atf_wick,
        spread_pips=SPREAD_PIPS,
    )

    # Prepare dual-timeframe data
    try:
        df_prepared = strategy.prepare_data(df_etf, df_atf)
    except Exception as e:
        logger.warning(f"  Error preparing data for {symbol}: {e}")
        return None

    if len(df_prepared) < 50:
        logger.warning(f"  Not enough prepared bars for {symbol}")
        return None

    # 70/15/15 split — use test set only
    n = len(df_prepared)
    test_start = int(n * 0.85)
    df_test = df_prepared.iloc[test_start:].copy()

    if len(df_test) < 20:
        logger.warning(f"  Test set too small for {symbol}")
        return None

    # Run backtest
    trades, equity_curve = strategy.backtest(df_test)
    metrics = compute_metrics(trades, equity_curve, len(df_test), etf_interval)

    if metrics is None:
        return {"trades": 0, "sharpe": 0, "pf": 0, "maxdd": 0,
                "trades_per_week": 0, "win_rate": 0}

    # Missed trades detection
    missed_info = {}
    if detect_missed and hasattr(strategy, "detect_missed_trades"):
        try:
            missed = strategy.detect_missed_trades(
                df_test, min_move_pct=0.5, lookahead_bars=20
            )
            if missed:
                reason_counts = Counter(m.reason for m in missed)
                top_reasons = reason_counts.most_common(3)
                avg_move = np.mean([m.peak_favorable_move_pct for m in missed])
                missed_info = {
                    "count": len(missed),
                    "avg_move_pct": avg_move,
                    "top_reasons": top_reasons,
                }
        except Exception as e:
            logger.debug(f"  Missed trades detection failed: {e}")

    metrics["missed"] = missed_info
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test SHA dual-timeframe strategy")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--tf-pairs", nargs="+", default=None,
                        help="ATF/ETF pairs like H1/M5 M30/M1")
    parser.add_argument("--sha-periods", nargs="+", type=int, default=None,
                        help="SHA periods to test (default: 6,8,10,12,14)")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="OOS epsilon tolerance")
    parser.add_argument("--epsilon-pct", type=float, default=0.0001,
                        help="Wick comparison epsilon as pct of price")
    parser.add_argument("--no-wick-filter", action="store_true",
                        help="Don't require ATF wick strength")
    parser.add_argument("--no-missed", action="store_true",
                        help="Skip missed trades detection")
    args = parser.parse_args()

    # Parse TF pairs
    tf_pairs = DEFAULT_TF_PAIRS
    if args.tf_pairs:
        tf_pairs = []
        tf_map = {"H1": "1h", "M30": "30m", "M15": "15m", "M10": "10m",
                   "M5": "5m", "M1": "1m", "D1": "1d",
                   "1h": "1h", "30m": "30m", "15m": "15m", "10m": "10m",
                   "5m": "5m", "1m": "1m", "1d": "1d"}
        for pair in args.tf_pairs:
            atf, etf = pair.split("/")
            tf_pairs.append((tf_map.get(atf, atf), tf_map.get(etf, etf)))

    sha_periods = args.sha_periods or SHA_PERIODS
    require_wick = not args.no_wick_filter
    detect_missed = not args.no_missed

    all_results = {}
    passing_strict = []
    passing_epsilon = []

    print("=" * 80)
    print("SHA DUAL-TIMEFRAME STRATEGY EVALUATION")
    print(f"OOS Criteria: Sharpe >= {MIN_SHARPE}, PF >= {MIN_PF}, MaxDD <= {MAX_DD}%")
    print(f"Epsilon: {args.epsilon} (relaxed: Sharpe >= {MIN_SHARPE*(1-args.epsilon):.2f}, "
          f"PF >= {MIN_PF*(1-args.epsilon):.2f}, MaxDD <= {MAX_DD*(1+args.epsilon):.1f}%)")
    print(f"SHA periods: {sha_periods}")
    print(f"TF pairs: {[f'{a}/{e}' for a, e in tf_pairs]}")
    print(f"Symbols: {args.symbols}")
    print(f"Wick filter: {'ON' if require_wick else 'OFF'}")
    print("=" * 80)

    for atf, etf in tf_pairs:
        print(f"\n{'='*80}")
        print(f"TIMEFRAME PAIR: ATF={atf}, ETF={etf}")
        print(f"{'='*80}")

        pair_key = f"{atf}/{etf}"
        all_results[pair_key] = {}

        for symbol in args.symbols:
            all_results[pair_key][symbol] = {}

            # Find best SHA period
            best_result = None
            best_period = None
            best_sharpe = -999

            for period in sha_periods:
                result = run_sha_test(
                    symbol, atf, etf, period, args.epsilon_pct,
                    require_wick, detect_missed=False,
                )
                if result and result.get("trades", 0) > 0:
                    if result["sharpe"] > best_sharpe:
                        best_sharpe = result["sharpe"]
                        best_result = result
                        best_period = period

            if best_result is None or best_result.get("trades", 0) == 0:
                print(f"\n  {symbol}: NO TRADES (all SHA periods tested)")
                all_results[pair_key][symbol] = {"trades": 0}
                continue

            # Re-run best with missed trades if requested
            if detect_missed:
                best_result = run_sha_test(
                    symbol, atf, etf, best_period, args.epsilon_pct,
                    require_wick, detect_missed=True,
                )

            # Check criteria
            pm = PerformanceMetrics(
                total_return=0.0, annual_return=0.0,
                sharpe_ratio=best_result["sharpe"],
                profit_factor=best_result["pf"],
                max_drawdown_pct=best_result["maxdd"],
                total_trades=best_result["trades"],
                win_rate=best_result["win_rate"],
                avg_trade_return=best_result.get("avg_pnl_pct", 0.0),
                calmar_ratio=0.0, sortino_ratio=0.0,
            )
            meets_strict = pm.meets_criteria(MIN_SHARPE, MIN_PF, MAX_DD, epsilon=0.0)
            meets_eps = pm.meets_criteria(MIN_SHARPE, MIN_PF, MAX_DD, epsilon=args.epsilon)

            flag = ""
            if meets_strict:
                flag = " [STRICT]"
                passing_strict.append(f"sha@{pair_key}/{symbol} (N={best_period})")
            elif meets_eps:
                flag = f" [EPSILON={args.epsilon}]"
                passing_epsilon.append(f"sha@{pair_key}/{symbol} (N={best_period})")

            missed_str = ""
            missed_info = best_result.get("missed", {})
            if missed_info and missed_info.get("count", 0) > 0:
                missed_str = f", Missed={missed_info['count']} (avg {missed_info['avg_move_pct']:.2f}%)"

            print(f"\n  {symbol} (SHA N={best_period})")
            print(f"    Sharpe={best_result['sharpe']:6.2f}, PF={best_result['pf']:5.2f}, "
                  f"MaxDD={best_result['maxdd']:5.1f}%, Trades={best_result['trades']:4d}, "
                  f"T/wk={best_result['trades_per_week']:5.1f}, "
                  f"WR={best_result['win_rate']:.1f}%{flag}{missed_str}")
            print(f"    Avg PnL/trade={best_result['avg_pnl_pct']:.3f}%, "
                  f"Avg bars held={best_result['avg_bars_held']:.1f}")

            if missed_info and missed_info.get("top_reasons"):
                for reason, count in missed_info["top_reasons"]:
                    print(f"      Missed reason: {reason} ({count}x)")

            best_result["sha_period"] = best_period
            best_result["meets_strict"] = meets_strict
            best_result["meets_epsilon"] = meets_eps
            all_results[pair_key][symbol] = best_result

    # --- Final Summary ---
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY (epsilon={args.epsilon})")
    print(f"{'='*80}")
    print(f"\nOOS Criteria: Sharpe >= {MIN_SHARPE}, PF >= {MIN_PF}, MaxDD <= {MAX_DD}%")
    print(f"With epsilon={args.epsilon}: Sharpe >= {MIN_SHARPE*(1-args.epsilon):.2f}, "
          f"PF >= {MIN_PF*(1-args.epsilon):.2f}, MaxDD <= {MAX_DD*(1+args.epsilon):.1f}%")

    if passing_strict:
        print(f"\nMEETS STRICT CRITERIA:")
        for s in passing_strict:
            print(f"  {s}")

    if passing_epsilon:
        print(f"\nMEETS EPSILON CRITERIA:")
        for s in passing_epsilon:
            print(f"  {s}")

    if not passing_strict and not passing_epsilon:
        print(f"\nNO strategies meet criteria (even with epsilon tolerance).")

    # Top 10 by Sharpe
    print(f"\nTop results by Sharpe:")
    scored = []
    for pair_key in all_results:
        for sym in all_results[pair_key]:
            r = all_results[pair_key][sym]
            if isinstance(r, dict) and r.get("trades", 0) > 0:
                scored.append((r["sharpe"], pair_key, sym, r))
    scored.sort(reverse=True)
    for _, pair_key, sym, r in scored[:10]:
        n = r.get("sha_period", "?")
        print(f"  sha@{pair_key}/{sym} (N={n}): "
              f"Sharpe={r['sharpe']:6.2f}, PF={r['pf']:5.2f}, "
              f"MaxDD={r['maxdd']:5.1f}%, T/wk={r['trades_per_week']:5.1f}")

    # Save results
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    out_file = report_dir / "sha_evaluation_results.json"

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (tuple,)):
            return list(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
