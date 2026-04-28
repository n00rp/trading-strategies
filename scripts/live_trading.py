#!/usr/bin/env python3
"""Live/Paper trading script for MT5.

Connects to MetaTrader 5 (IC Markets/IG) and executes signals
from trained models in real-time.

Prerequisites:
  - Windows OS with MetaTrader 5 installed
  - pip install MetaTrader5
  - Trained models in models/ directory

Usage:
    python scripts/live_trading.py --strategy trend --symbol sp500 --mode paper
"""

import argparse
import logging
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

MT5_CONFIG = {
    "server": "ICMarketsSC-Demo",  # Change for live: "ICMarketsSC-Live"
    "login": 0,          # Your MT5 account number
    "password": "",      # Your MT5 password
    "path": r"C:\Program Files\MetaTrader 5\terminal64.exe",
}

STRATEGY_CONFIGS = {
    "trend": {
        "model_path": "models/model_A_trend_following",
        "strategy_type": "trend",
        "feature_params": {"ema_fast": 12, "ema_slow": 50},
        "lot_size": 0.1,
        "sl_pips": 50,
        "tp_pips": 100,
    },
    "mean_reversion": {
        "model_path": "models/model_B_mean_reversion",
        "strategy_type": "mean_reversion",
        "feature_params": {"rsi_period": 14, "bb_period": 20, "bb_std": 2.0},
        "lot_size": 0.1,
        "sl_pips": 30,
        "tp_pips": 60,
    },
    "volatility": {
        "model_path": "models/model_C_volatility_breakout",
        "strategy_type": "volatility",
        "feature_params": {"bb_period": 20, "kc_period": 20, "bb_mult": 2.0, "kc_mult": 1.5},
        "lot_size": 0.1,
        "sl_pips": 40,
        "tp_pips": 100,
    },
}

SYMBOL_MAP = {
    "sp500": "US500",
    "nasdaq": "USTEC",
    "dowjones": "US30",
    "dax": "DE40",
}


def get_live_features(mt5, symbol: str, strategy_type: str, feature_params: dict, n_bars: int = 100):
    """Fetch recent bars from MT5 and compute features."""
    from src.features.engineering import prepare_strategy_data

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})

    return prepare_strategy_data(df, strategy_type, feature_params)


def main():
    parser = argparse.ArgumentParser(description="MT5 Live Trading")
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["trend", "mean_reversion", "volatility"])
    parser.add_argument("--symbol", type=str, default="sp500",
                        choices=["sp500", "nasdaq", "dowjones", "dax"])
    parser.add_argument("--mode", type=str, default="paper", choices=["paper", "live"])
    parser.add_argument("--interval", type=int, default=3600, help="Check interval in seconds")
    args = parser.parse_args()

    config = STRATEGY_CONFIGS[args.strategy]
    mt5_symbol = SYMBOL_MAP[args.symbol]

    logger.info(f"Starting {args.mode} trading: {args.strategy} on {args.symbol} ({mt5_symbol})")

    # Initialize MT5
    from src.mt5.connector import MT5Connector, MT5Config, InferenceEngine

    mt5_config = MT5Config(**MT5_CONFIG)
    connector = MT5Connector(mt5_config)

    if not connector.connect():
        logger.error("Failed to connect to MT5. Check configuration.")
        sys.exit(1)

    # Load inference engine
    engine = InferenceEngine(
        strategy_type=config["strategy_type"],
        model_path=config["model_path"],
        params=config.get("rule_params", {}),
    )
    engine.load_model()

    logger.info("Starting trading loop...")
    try:
        while True:
            try:
                # Get features
                features = get_live_features(
                    connector.mt5,
                    mt5_symbol,
                    config["strategy_type"],
                    config["feature_params"],
                )

                if features is None or features.empty:
                    logger.warning("No data received, retrying...")
                    time.sleep(60)
                    continue

                # Get signal
                exclude_cols = {"open", "high", "low", "close", "volume", "symbol",
                                "returns", "log_returns"}
                feature_cols = [c for c in features.columns if c not in exclude_cols]
                signal = engine.predict(features[feature_cols])

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"[{timestamp}] Signal: {signal} "
                            f"({'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'})")

                # Execute (only in live/paper mode with actual MT5)
                if signal != 0 and args.mode in ("live", "paper"):
                    result = connector.execute_signal(
                        args.symbol,
                        signal=signal,
                        lot_size=config["lot_size"],
                        sl_pips=config["sl_pips"],
                        tp_pips=config["tp_pips"],
                    )
                    if result:
                        logger.info(f"  Order executed: {result}")

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")

            logger.info(f"Sleeping {args.interval}s until next check...")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("\nStopping trading...")
    finally:
        connector.disconnect()
        logger.info("Disconnected from MT5")


if __name__ == "__main__":
    main()
