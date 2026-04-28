"""Data fetching module using yfinance.

For M15/H1 data, yfinance has limitations:
  - 15m: max ~60 days of history
  - 1h:  max ~730 days of history
  - 1d:  full history available

Strategy:
  1. Fetch daily data for full 10+ year history (primary for backtesting).
  2. Fetch hourly data for available period (~2 years) for higher-res validation.
  3. Fetch 15m data for recent period (~60 days) for granular analysis.

For production CFD trading, use MT5 API to get full intraday history.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SYMBOLS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dowjones": "^DJI",
    "dax": "^GDAXI",
}

# yfinance interval constraints
INTERVAL_MAX_PERIOD = {
    "15m": 60,     # days
    "1h": 730,     # days
    "1d": 365 * 30,  # effectively unlimited
}


def fetch_symbol(
    symbol: str,
    name: str,
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data for a single symbol."""
    max_days = INTERVAL_MAX_PERIOD.get(interval, 365 * 30)

    if start is None:
        start_dt = datetime.now() - timedelta(days=max_days)
        start = start_dt.strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    # Clamp start date based on interval limitation
    earliest = datetime.now() - timedelta(days=max_days)
    start_dt = max(datetime.strptime(start, "%Y-%m-%d"), earliest)
    start = start_dt.strftime("%Y-%m-%d")

    # Also clamp end date to now (can't fetch future data)
    end_dt = min(datetime.strptime(end, "%Y-%m-%d"), datetime.now())
    end = end_dt.strftime("%Y-%m-%d")

    if start_dt >= end_dt:
        logger.warning(f"Start date >= end date for {name} interval={interval}, skipping")
        return pd.DataFrame()

    logger.info(f"Fetching {name} ({symbol}) | interval={interval} | {start} -> {end}")

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)

    if df.empty:
        logger.warning(f"No data returned for {name} ({symbol}) interval={interval}")
        return df

    # Standardize columns
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "datetime"
    df["symbol"] = name

    logger.info(f"  -> {len(df)} bars fetched for {name}")
    return df


def fetch_all(
    intervals: list[str] | None = None,
    start: str = "2014-01-01",
    end: str = "2024-12-31",
    save: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch data for all symbols and intervals.

    Returns:
        Nested dict: {interval: {symbol_name: DataFrame}}
    """
    if intervals is None:
        intervals = ["1d", "1h"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    result: dict[str, dict[str, pd.DataFrame]] = {}

    for interval in intervals:
        result[interval] = {}
        for name, symbol in SYMBOLS.items():
            df = fetch_symbol(symbol, name, interval=interval, start=start, end=end)
            if not df.empty:
                result[interval][name] = df
                if save:
                    out_path = RAW_DIR / f"{name}_{interval}.parquet"
                    df.to_parquet(out_path)
                    logger.info(f"  -> Saved to {out_path}")

    return result


def load_data(name: str, interval: str = "1d") -> pd.DataFrame:
    """Load previously saved data from parquet."""
    path = RAW_DIR / f"{name}_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}. Run fetch_all() first.")
    return pd.read_parquet(path)


def load_all(interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Load all symbols for a given interval."""
    data = {}
    for name in SYMBOLS:
        try:
            data[name] = load_data(name, interval)
        except FileNotFoundError:
            logger.warning(f"No data file for {name} at interval {interval}")
    return data
