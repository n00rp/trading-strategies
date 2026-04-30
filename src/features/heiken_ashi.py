"""Heiken Ashi and Smoothed Heiken Ashi (SHA) indicator computations.

Heiken Ashi (HA):
    C_HA = (O + H + L + C) / 4
    O_HA = (O_HA[t-1] + C_HA[t-1]) / 2  (recursive)
    H_HA = max(H, O_HA, C_HA)
    L_HA = min(L, O_HA, C_HA)

Smoothed Heiken Ashi (SHA) — "Double Smoothing":
    1. Compute EMA(O, N), EMA(H, N), EMA(L, N), EMA(C, N)
    2. Apply HA formulas to the EMA-smoothed values
    Result: O_SHA, H_SHA, L_SHA, C_SHA
"""

import numpy as np
import pandas as pd


def compute_heiken_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heiken Ashi candles from raw OHLC data.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns.

    Returns:
        DataFrame with ha_open, ha_high, ha_low, ha_close columns added.
    """
    df = df.copy()
    n = len(df)

    ha_close = (df["open"].values + df["high"].values +
                df["low"].values + df["close"].values) / 4.0

    ha_open = np.empty(n)
    ha_open[0] = (df["open"].values[0] + df["close"].values[0]) / 2.0

    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum(df["high"].values, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(df["low"].values, np.minimum(ha_open, ha_close))

    df["ha_open"] = ha_open
    df["ha_high"] = ha_high
    df["ha_low"] = ha_low
    df["ha_close"] = ha_close

    return df


def compute_smoothed_heiken_ashi(
    df: pd.DataFrame, period: int = 10
) -> pd.DataFrame:
    """Compute Smoothed Heiken Ashi (SHA) using double EMA smoothing.

    Steps:
        1. EMA-smooth each of O, H, L, C with given period
        2. Apply HA formulas to the smoothed values

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns.
        period: EMA smoothing period (default 10, optimize 6-14).

    Returns:
        DataFrame with sha_open, sha_high, sha_low, sha_close columns added.
    """
    df = df.copy()

    # Step 1: EMA smooth raw OHLC
    o_ema = df["open"].ewm(span=period, adjust=False).mean().values
    h_ema = df["high"].ewm(span=period, adjust=False).mean().values
    l_ema = df["low"].ewm(span=period, adjust=False).mean().values
    c_ema = df["close"].ewm(span=period, adjust=False).mean().values

    # Step 2: Apply HA formulas to EMA-smoothed values
    n = len(df)
    sha_close = (o_ema + h_ema + l_ema + c_ema) / 4.0

    sha_open = np.empty(n)
    sha_open[0] = (o_ema[0] + c_ema[0]) / 2.0

    for i in range(1, n):
        sha_open[i] = (sha_open[i - 1] + sha_close[i - 1]) / 2.0

    sha_high = np.maximum(h_ema, np.maximum(sha_open, sha_close))
    sha_low = np.minimum(l_ema, np.minimum(sha_open, sha_close))

    df["sha_open"] = sha_open
    df["sha_high"] = sha_high
    df["sha_low"] = sha_low
    df["sha_close"] = sha_close

    return df


def is_bullish(open_val: float, close_val: float) -> bool:
    """Check if a candle is bullish (green): Close > Open."""
    return close_val > open_val


def is_bearish(open_val: float, close_val: float) -> bool:
    """Check if a candle is bearish (red): Close < Open."""
    return close_val < open_val


def has_no_lower_wick(
    low_val: float, open_val: float, epsilon: float = 0.0
) -> bool:
    """Check if candle has no lower wick (strong bullish).

    Mathematically: L_HA >= O_HA (within epsilon tolerance).
    """
    return low_val >= open_val - epsilon


def has_no_upper_wick(
    high_val: float, open_val: float, epsilon: float = 0.0
) -> bool:
    """Check if candle has no upper wick (strong bearish).

    Mathematically: H_HA <= O_HA (within epsilon tolerance).
    """
    return high_val <= open_val + epsilon
