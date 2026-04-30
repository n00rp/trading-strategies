"""Data fetching module.

Supports loading from:
  1. MT5-exported parquet files (preferred — full intraday history)
  2. yfinance (fallback — limited intraday history)

MT5 file naming conventions supported:
  - {SYMBOL}_{TF}_{start}_{end}.parquet  (e.g. US500_M15_201705100100_202604212345.parquet)
  - {name}_{interval}.parquet  (e.g. sp500_15m.parquet)
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

# Map MT5 symbol names to our internal names
MT5_SYMBOL_MAP = {
    "US500": "sp500",
    "USTEC": "nasdaq",
    "DOW.NYSE": "dowjones",
    "DE40": "dax",
}

# Map MT5 timeframe codes to our interval names
MT5_TF_MAP = {
    "M1": "1m",
    "M5": "5m",
    "M10": "10m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "D1": "1d",
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
    """Load previously saved data from parquet.

    Searches for data files in this order:
      1. MT5-style naming: {MT5_SYMBOL}_{MT5_TF}_{dates}.parquet
      2. Simple naming: {name}_{interval}.parquet
    """
    import glob

    # Reverse-map our name to MT5 symbol
    mt5_symbol = None
    for mt5_sym, our_name in MT5_SYMBOL_MAP.items():
        if our_name == name:
            mt5_symbol = mt5_sym
            break

    # Reverse-map our interval to MT5 timeframe
    mt5_tf = None
    for mt5_code, our_interval in MT5_TF_MAP.items():
        if our_interval == interval:
            mt5_tf = mt5_code
            break

    # Try MT5-style files first
    if mt5_symbol and mt5_tf:
        pattern = str(RAW_DIR / f"{mt5_symbol}_{mt5_tf}_*.parquet")
        matches = sorted(glob.glob(pattern))
        if matches:
            path = Path(matches[-1])  # latest file
            logger.info(f"Loading MT5 data: {path.name}")
            return pd.read_parquet(path)

    # Fall back to simple naming
    path = RAW_DIR / f"{name}_{interval}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    raise FileNotFoundError(
        f"No data found for {name} @ {interval}. "
        f"Searched: {mt5_symbol}_{mt5_tf}_*.parquet and {name}_{interval}.parquet "
        f"in {RAW_DIR}"
    )


def load_all(interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Load all symbols for a given interval."""
    data = {}
    for name in SYMBOLS:
        try:
            data[name] = load_data(name, interval)
        except FileNotFoundError:
            logger.warning(f"No data file for {name} at interval {interval}")
    return data
