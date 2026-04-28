#!/usr/bin/env python3
"""Fetch all market data for the trading strategies project.

Run this first to populate data/raw/ with historical OHLCV data.
Uses yfinance for free data access.

Usage:
    python scripts/fetch_data.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetcher import fetch_all, load_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("FETCHING MARKET DATA")
    logger.info("=" * 60)

    # Fetch daily data (10+ years)
    logger.info("\n--- Daily Data (10+ years) ---")
    daily_data = fetch_all(
        intervals=["1d"],
        start="2014-01-01",
        end="2024-12-31",
        save=True,
    )

    for name, df in daily_data.get("1d", {}).items():
        logger.info(f"  {name}: {len(df)} bars | {df.index[0]} -> {df.index[-1]}")

    # Fetch hourly data (available period ~2 years)
    logger.info("\n--- Hourly Data (available period) ---")
    hourly_data = fetch_all(
        intervals=["1h"],
        start="2014-01-01",  # will be clamped to available
        end="2024-12-31",
        save=True,
    )

    for name, df in hourly_data.get("1h", {}).items():
        logger.info(f"  {name}: {len(df)} bars | {df.index[0]} -> {df.index[-1]}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATA FETCH COMPLETE")
    logger.info("=" * 60)

    # Verify saved data loads correctly
    logger.info("\nVerifying saved data...")
    try:
        all_daily = load_all("1d")
        for name, df in all_daily.items():
            logger.info(f"  Loaded {name} (1d): {len(df)} bars")
    except Exception as e:
        logger.error(f"Verification failed: {e}")


if __name__ == "__main__":
    main()
