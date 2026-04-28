# MT5 Data Export

## Setup

1. Install MetaTrader 5 and connect to IC Markets (or your broker)
2. Copy `ExportData.mq5` to your MT5 `MQL5/Scripts/` directory
3. Open MetaEditor (F4 in MT5), open the script, and click Compile (F7)
4. In MT5, drag `ExportData` from Navigator → Scripts onto any chart
5. Click OK to run — it exports all symbols × all timeframes

## Exported Data

The script exports CSV files to `MQL5/Files/`:

| Symbol | MT5 Name | Indices |
|--------|----------|---------|
| sp500 | US500 | S&P 500 |
| nasdaq | USTEC | Nasdaq 100 |
| dowjones | US30 | Dow Jones |
| dax | DE40 | DAX 40 |

Timeframes: M1, M5, M10, M15, M30, H1, D1 (up to 100,000 bars each)

## Converting to Parquet

After exporting, run:

```bash
python scripts/convert_mt5_csv.py --input-dir "/path/to/MQL5/Files" --output-dir "data/raw"
```

Or if your MT5 Files directory is at the default Wine path:
```bash
python scripts/convert_mt5_csv.py
```

## Data Availability (IC Markets)

| Timeframe | Typical History | Bars |
|-----------|----------------|------|
| M1 | ~3 months | 100K |
| M5 | ~5 months | 100K |
| M10 | ~3 years | 100K |
| M15 | ~4 years | 100K |
| M30 | ~8 years | 100K |
| H1 | 14 years | ~60K |
| D1 | 14 years | ~3.5K |
