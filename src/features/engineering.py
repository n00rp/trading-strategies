"""Feature engineering for all three strategy types.

Anti-bias: All features use ONLY past data (no look-ahead).
Every indicator is computed using `.shift(1)` where necessary to ensure
signals are generated from data available BEFORE the current bar.
"""

import numpy as np
import pandas as pd
import ta


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add universal features used across all strategies."""
    df = df.copy()

    # Returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # ATR (Average True Range)
    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # Volume features
    if "volume" in df.columns and df["volume"].sum() > 0:
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    else:
        df["volume_sma_20"] = 0.0
        df["volume_ratio"] = 1.0

    return df


def add_trend_features(df: pd.DataFrame, ema_fast: int = 12, ema_slow: int = 50) -> pd.DataFrame:
    """Features for Strategy A: Trend Following.

    - EMA crossovers (fast/slow)
    - ADX for trend strength
    - MACD histogram
    - Supertrend proxy
    """
    df = df.copy()

    # EMAs
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=ema_fast)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=ema_slow)
    df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
    df["ema_diff_pct"] = df["ema_diff"] / df["close"]

    # EMA crossover signal (1 = bullish cross, -1 = bearish cross)
    df["ema_cross"] = 0
    df.loc[
        (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)),
        "ema_cross",
    ] = 1
    df.loc[
        (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)),
        "ema_cross",
    ] = -1

    # ADX
    adx_indicator = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx_indicator.adx()
    df["adx_pos"] = adx_indicator.adx_pos()
    df["adx_neg"] = adx_indicator.adx_neg()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Trend regime: 1=uptrend, -1=downtrend, 0=no trend
    df["trend_regime"] = 0
    df.loc[(df["ema_fast"] > df["ema_slow"]) & (df["adx"] > 25), "trend_regime"] = 1
    df.loc[(df["ema_fast"] < df["ema_slow"]) & (df["adx"] > 25), "trend_regime"] = -1

    return df


def add_mean_reversion_features(
    df: pd.DataFrame,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """Features for Strategy B: Mean Reversion.

    - RSI extremes
    - Bollinger Band position
    - Z-score of price relative to moving average
    - Stochastic oscillator
    """
    df = df.copy()

    # RSI
    df["rsi"] = ta.momentum.rsi(df["close"], window=rsi_period)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=bb_period, window_dev=bb_std)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Z-score
    rolling_mean = df["close"].rolling(bb_period).mean()
    rolling_std = df["close"].rolling(bb_period).std()
    df["zscore"] = (df["close"] - rolling_mean) / rolling_std

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Mean reversion signals
    df["mr_signal"] = 0
    df.loc[(df["rsi"] < 30) & (df["bb_pct"] < 0.05), "mr_signal"] = 1   # oversold
    df.loc[(df["rsi"] > 70) & (df["bb_pct"] > 0.95), "mr_signal"] = -1  # overbought

    return df


def add_volatility_features(
    df: pd.DataFrame,
    bb_period: int = 20,
    kc_period: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
) -> pd.DataFrame:
    """Features for Strategy C: Volatility/Breakout.

    - Bollinger Band squeeze detection (BB inside KC)
    - Opening range breakout levels
    - Historical volatility
    - Keltner Channels
    """
    df = df.copy()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=bb_period, window_dev=bb_mult)
    df["bb_upper_vol"] = bb.bollinger_hband()
    df["bb_lower_vol"] = bb.bollinger_lband()

    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(
        df["high"], df["low"], df["close"], window=kc_period, window_atr=kc_period,
        multiplier=kc_mult,
    )
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()

    # Squeeze: BB is inside KC => low volatility, expect breakout
    df["squeeze"] = (
        (df["bb_lower_vol"] > df["kc_lower"]) & (df["bb_upper_vol"] < df["kc_upper"])
    ).astype(int)

    # Squeeze released
    df["squeeze_release"] = 0
    df.loc[(df["squeeze"] == 0) & (df["squeeze"].shift(1) == 1), "squeeze_release"] = 1

    # Historical volatility (annualized)
    df["hvol_20"] = df["log_returns"].rolling(20).std() * np.sqrt(252)
    df["hvol_50"] = df["log_returns"].rolling(50).std() * np.sqrt(252)
    df["vol_ratio"] = df["hvol_20"] / df["hvol_50"]

    # Donchian Channel (for breakout levels)
    df["donchian_high_20"] = df["high"].rolling(20).max()
    df["donchian_low_20"] = df["low"].rolling(20).min()
    df["donchian_mid"] = (df["donchian_high_20"] + df["donchian_low_20"]) / 2

    # Breakout signal
    df["breakout_signal"] = 0
    df.loc[df["close"] > df["donchian_high_20"].shift(1), "breakout_signal"] = 1
    df.loc[df["close"] < df["donchian_low_20"].shift(1), "breakout_signal"] = -1

    return df


def prepare_strategy_data(
    df: pd.DataFrame,
    strategy_type: str,
    params: dict | None = None,
) -> pd.DataFrame:
    """Full feature pipeline for a given strategy type.

    Args:
        df: Raw OHLCV DataFrame
        strategy_type: One of 'trend', 'mean_reversion', 'volatility'
        params: Optional parameter overrides for feature engineering

    Returns:
        DataFrame with all features, NaN rows dropped
    """
    params = params or {}

    # Always add base features
    df = add_base_features(df)

    if strategy_type == "trend":
        df = add_trend_features(
            df,
            ema_fast=params.get("ema_fast", 12),
            ema_slow=params.get("ema_slow", 50),
        )
    elif strategy_type == "mean_reversion":
        df = add_mean_reversion_features(
            df,
            rsi_period=params.get("rsi_period", 14),
            bb_period=params.get("bb_period", 20),
            bb_std=params.get("bb_std", 2.0),
        )
    elif strategy_type == "volatility":
        df = add_volatility_features(
            df,
            bb_period=params.get("bb_period", 20),
            kc_period=params.get("kc_period", 20),
            bb_mult=params.get("bb_mult", 2.0),
            kc_mult=params.get("kc_mult", 1.5),
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    # Drop NaN rows from indicator warm-up (CRITICAL: avoids look-ahead bias)
    df = df.dropna()

    return df
