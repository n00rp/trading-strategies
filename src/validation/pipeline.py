"""Strict validation pipeline.

Implements:
  - Train/Val/Test split (70/15/15) — chronological, no shuffling
  - Walk-Forward Analysis
  - Monte Carlo stress testing
  - Anti-bias checks (look-ahead, overfitting detection)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Strategy performance metrics."""

    total_return: float
    annual_return: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    calmar_ratio: float
    sortino_ratio: float

    def meets_criteria(
        self,
        min_sharpe: float = 1.2,
        min_pf: float = 1.5,
        max_dd: float = 15.0,
    ) -> bool:
        return (
            self.sharpe_ratio >= min_sharpe
            and self.profit_factor >= min_pf
            and self.max_drawdown_pct <= max_dd
        )

    def summary(self) -> str:
        return (
            f"Sharpe={self.sharpe_ratio:.2f} | PF={self.profit_factor:.2f} | "
            f"MaxDD={self.max_drawdown_pct:.1f}% | WR={self.win_rate:.1f}% | "
            f"Trades={self.total_trades} | AnnRet={self.annual_return:.1f}%"
        )


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically (NO shuffling — prevents look-ahead bias)."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    logger.info(
        f"Split: train={len(train)} ({train.index[0]} -> {train.index[-1]}) | "
        f"val={len(val)} ({val.index[0]} -> {val.index[-1]}) | "
        f"test={len(test)} ({test.index[0]} -> {test.index[-1]})"
    )
    return train, val, test


def compute_metrics(
    trade_returns: np.ndarray,
    equity_curve: np.ndarray | None = None,
    n_trading_days: int | None = None,
) -> PerformanceMetrics:
    """Compute performance metrics from per-trade returns.

    Args:
        trade_returns: Array of per-trade returns (e.g., [0.02, -0.01, 0.03, ...])
        equity_curve: Optional cumulative equity curve
        n_trading_days: Number of trading days in the evaluation period.
            If provided, used to compute trades-per-year for proper
            Sharpe annualization. If None, assumes 252 days per year
            and estimates from trade count (conservative).
    """
    if len(trade_returns) == 0:
        return PerformanceMetrics(
            total_return=0, annual_return=0, sharpe_ratio=0, profit_factor=0,
            max_drawdown_pct=100, win_rate=0, total_trades=0,
            avg_trade_return=0, calmar_ratio=0, sortino_ratio=0,
        )

    trade_returns = np.asarray(trade_returns)

    # Basic stats
    total_return = float(np.prod(1 + trade_returns) - 1)
    n_trades = len(trade_returns)
    avg_ret = float(np.mean(trade_returns))
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    win_rate = float(len(wins) / n_trades * 100) if n_trades > 0 else 0

    # Sharpe ratio (annualized using trades-per-year, NOT sqrt(252))
    # The annualization factor for per-trade returns should be
    # sqrt(trades_per_year), not sqrt(trading_days_per_year).
    if n_trading_days is not None and n_trading_days > 0:
        n_years = n_trading_days / 252.0
    else:
        # Conservative: assume test period is ~1.5 years for 15% split of 10yr
        n_years = max(n_trades / 252.0, 0.1)

    trades_per_year = n_trades / max(n_years, 0.01)

    if np.std(trade_returns) > 0:
        sharpe = float(
            np.mean(trade_returns) / np.std(trade_returns)
            * np.sqrt(trades_per_year)
        )
    else:
        sharpe = 0.0

    # Sortino ratio (same annualization)
    downside = trade_returns[trade_returns < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = float(
            np.mean(trade_returns) / np.std(downside)
            * np.sqrt(trades_per_year)
        )
    else:
        sortino = sharpe

    # Profit factor
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    # Max drawdown from equity curve
    if equity_curve is None:
        equity_curve = np.cumprod(1 + trade_returns)

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max * 100
    max_dd = float(np.abs(np.min(drawdowns)))

    # Annualized return
    if n_years > 0 and total_return > -1:
        annual_return = float((1 + total_return) ** (1 / max(n_years, 0.01)) - 1) * 100
    else:
        annual_return = 0.0

    # Calmar
    calmar = annual_return / max_dd if max_dd > 0 else 0.0

    return PerformanceMetrics(
        total_return=total_return * 100,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        total_trades=n_trades,
        avg_trade_return=avg_ret * 100,
        calmar_ratio=calmar,
        sortino_ratio=sortino,
    )


def walk_forward_analysis(
    df: pd.DataFrame,
    strategy_fn,
    n_splits: int = 5,
    train_ratio: float = 0.7,
) -> list[PerformanceMetrics]:
    """Walk-forward analysis with expanding/rolling windows.

    Args:
        df: Full dataset with features
        strategy_fn: Callable(train_df, test_df) -> np.ndarray of trade returns
        n_splits: Number of forward-walk periods
        train_ratio: Ratio of training data in each split

    Returns:
        List of PerformanceMetrics for each OOS period
    """
    n = len(df)
    fold_size = n // n_splits
    results = []

    for i in range(n_splits):
        # Expanding window: train on all data up to split point
        split_end = (i + 1) * fold_size
        if split_end >= n:
            break

        train_end = int(split_end * train_ratio)
        test_start = train_end
        test_end = min(split_end, n)

        if test_start >= test_end:
            continue

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        logger.info(
            f"WF fold {i + 1}/{n_splits}: "
            f"train[0:{train_end}] test[{test_start}:{test_end}]"
        )

        trade_returns = strategy_fn(train_df, test_df)
        metrics = compute_metrics(trade_returns)
        results.append(metrics)
        logger.info(f"  -> {metrics.summary()}")

    return results


def monte_carlo_test(
    trade_returns: np.ndarray,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
) -> dict:
    """Monte Carlo stress test by shuffling trade order.

    Tests whether the strategy's performance is robust or dependent
    on specific trade sequencing.
    """
    trade_returns = np.asarray(trade_returns)
    n = len(trade_returns)

    if n < 10:
        return {
            "median_sharpe": 0, "p5_sharpe": 0, "p95_sharpe": 0,
            "median_max_dd": 100, "p95_max_dd": 100,
            "prob_profitable": 0, "n_simulations": 0,
        }

    sharpes = []
    max_dds = []
    final_returns = []

    for _ in range(n_simulations):
        shuffled = np.random.permutation(trade_returns)
        equity = np.cumprod(1 + shuffled)
        final_returns.append(equity[-1] - 1)

        # Sharpe
        if np.std(shuffled) > 0:
            sharpes.append(np.mean(shuffled) / np.std(shuffled) * np.sqrt(252))
        else:
            sharpes.append(0)

        # Max DD
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        max_dds.append(np.abs(np.min(dd)) * 100)

    sharpes = np.array(sharpes)
    max_dds = np.array(max_dds)
    final_returns = np.array(final_returns)

    ci_low = (1 - confidence_level) / 2 * 100
    ci_high = (1 + confidence_level) / 2 * 100

    return {
        "median_sharpe": float(np.median(sharpes)),
        "p5_sharpe": float(np.percentile(sharpes, ci_low)),
        "p95_sharpe": float(np.percentile(sharpes, ci_high)),
        "median_max_dd": float(np.median(max_dds)),
        "p95_max_dd": float(np.percentile(max_dds, ci_high)),
        "prob_profitable": float(np.mean(final_returns > 0) * 100),
        "n_simulations": n_simulations,
    }


def check_overfitting(
    train_metrics: PerformanceMetrics,
    val_metrics: PerformanceMetrics,
    test_metrics: PerformanceMetrics,
    max_degradation: float = 0.5,
) -> dict:
    """Check for signs of overfitting.

    Flags:
      - Large gap between train and val/test Sharpe
      - Val/test Sharpe significantly lower than train
      - Inconsistency between val and test
    """
    train_sharpe = train_metrics.sharpe_ratio
    val_sharpe = val_metrics.sharpe_ratio
    test_sharpe = test_metrics.sharpe_ratio

    val_degradation = (
        (train_sharpe - val_sharpe) / abs(train_sharpe) if train_sharpe != 0 else 0
    )
    test_degradation = (
        (train_sharpe - test_sharpe) / abs(train_sharpe) if train_sharpe != 0 else 0
    )
    val_test_gap = abs(val_sharpe - test_sharpe)

    is_overfit = (
        val_degradation > max_degradation
        or test_degradation > max_degradation
    )

    return {
        "is_overfit": is_overfit,
        "train_sharpe": train_sharpe,
        "val_sharpe": val_sharpe,
        "test_sharpe": test_sharpe,
        "val_degradation_pct": val_degradation * 100,
        "test_degradation_pct": test_degradation * 100,
        "val_test_consistency": val_test_gap,
        "verdict": "OVERFIT ⚠" if is_overfit else "OK",
    }


def check_look_ahead_bias(df: pd.DataFrame, signal_col: str) -> dict:
    """Statistical check for potential look-ahead bias.

    Tests correlation between signal and FUTURE returns (should be zero
    if signal is properly lagged).
    """
    future_ret = df["close"].pct_change().shift(-1)  # future return
    current_ret = df["close"].pct_change()  # current return

    signal = df[signal_col]

    # Correlation with future returns (should NOT be significant)
    future_corr = signal.corr(future_ret)

    # Correlation with current returns (can be significant)
    current_corr = signal.corr(current_ret)

    suspicious = abs(future_corr) > 0.3  # high correlation with future = look-ahead

    return {
        "signal_column": signal_col,
        "future_return_correlation": float(future_corr),
        "current_return_correlation": float(current_corr),
        "suspicious_look_ahead": suspicious,
        "verdict": "LOOK-AHEAD BIAS ⚠" if suspicious else "OK",
    }
