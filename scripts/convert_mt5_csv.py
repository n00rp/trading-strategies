#!/usr/bin/env python3
"""Convert MT5 exported CSV files to parquet format for the pipeline.

Usage:
    python scripts/convert_mt5_csv.py --input-dir "/path/to/MQL5/Files"
    python scripts/convert_mt5_csv.py  # uses default Wine path
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def convert_mt5_csvs(input_dir: Path, output_dir: Path):
    """Convert all MT5 CSV files to parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for f in csv_files:
        df = pd.read_csv(f)

        if "time" not in df.columns:
            print(f"  Skipping {f.name} (no 'time' column)")
            continue

        df["datetime"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("datetime", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        parts = f.stem.split("_")
        name = parts[0]
        tf = "_".join(parts[1:])

        df["symbol"] = name
        df = df[["open", "high", "low", "close", "volume", "symbol"]].copy()

        out = output_dir / f"{name}_{tf}.parquet"
        df.to_parquet(out)
        print(f"  {f.name}: {len(df)} bars ({df.index[0]} -> {df.index[-1]}) -> {out.name}")


def main():
    parser = argparse.ArgumentParser(description="Convert MT5 CSV exports to parquet")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing MT5 CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for parquet files",
    )
    args = parser.parse_args()

    if args.input_dir is None:
        # Try common MT5 paths
        candidates = [
            Path.home() / ".wine_mt5/drive_c/Program Files/MetaTrader 5/MQL5/Files",
            Path.home() / ".wine/drive_c/Program Files/MetaTrader 5/MQL5/Files",
            Path(r"C:\Program Files\MetaTrader 5\MQL5\Files"),
            Path.home() / "AppData/Roaming/MetaQuotes/Terminal" / "*/MQL5/Files",
        ]
        input_dir = None
        for c in candidates:
            if c.exists():
                input_dir = c
                break
        if input_dir is None:
            print("Could not find MT5 Files directory. Use --input-dir to specify.")
            sys.exit(1)
    else:
        input_dir = Path(args.input_dir)

    output_dir = Path(args.output_dir)
    print(f"Converting CSVs from: {input_dir}")
    print(f"Output to: {output_dir}\n")
    convert_mt5_csvs(input_dir, output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
