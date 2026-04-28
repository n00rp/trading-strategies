# Strategy Comparison Report

## ⚠️ Status: Pending Multi-Timeframe Optimization

Previous results used a **buggy Sharpe calculation** (`sqrt(252)` on per-trade returns instead of `sqrt(trades_per_year)`). This inflated Sharpe ratios by ~3x.

Additionally, strategies B and C generated **too few trades on daily data** (2-5 per index over ~1.5 year test period). The solution is to use **intraday data** (M5-H1) from MT5 to get ~10 trades/week.

**What's been done:**
- Fixed Sharpe annualization bug in `src/validation/pipeline.py`
- Exported 14 years of data from IC Markets MT5 across 7 timeframes (M1-D1)
- Created `scripts/run_multi_timeframe.py` for local optimization across all timeframes
- Updated optimizer to pass `n_trading_days` for correct Sharpe calculation

**What needs to happen next:**
- Run `python scripts/run_multi_timeframe.py --trials 500 --cross-validate` on your local rig
- Identify best timeframe per strategy (targeting ~10 trades/week)
- Update this report with real, validated results

---

## Corrected Results on Daily Data (for reference)

After fixing the Sharpe bug, daily data results on S&P 500 test set:

| Strategy | Sharpe | PF | MaxDD | Trades | T/wk | Meets? |
|----------|--------|-----|-------|--------|------|--------|
| **A: Trend Following** | 1.83 | 2.12 | 8.0% | 45 | 0.6 | YES (but few trades/wk) |
| **B: Mean Reversion** | 0.80 | 2.68 | 2.5% | 4 | 0.05 | NO (too few trades) |
| **C: Volatility Breakout** | 0.00 | — | 0.0% | 1 | 0.01 | NO (too few trades) |

Strategy A meets OOS criteria on daily S&P 500 and Dow Jones, but trade frequency is too low (~0.6/week). Strategies B and C are not viable on daily data.

---

## OOS Criteria

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 1.2 | Annualized using `sqrt(trades_per_year)` |
| Profit Factor | > 1.5 | Gross profit / gross loss |
| Max Drawdown | < 15% | Peak-to-trough from equity curve |
| Trade Frequency | ~10/week | Target for meaningful statistical significance |
| Spread | 1.0 pip | Applied on entry and exit |

---

## System A: Trend Following (EMA Cross + ADX + MACD)

**Market Regime**: Trending markets (sustained directional moves)

### Logic
- **Long**: EMA(fast) crosses above EMA(slow) + ADX > threshold + MACD histogram > 0
- **Short**: EMA(fast) crosses below EMA(slow) + ADX > threshold + MACD histogram < 0
- **Exit**: ATR-based SL/TP

### Key Characteristics
- Highest trade count of all strategies
- Most consistent across indices
- Works best on trending equity indices (S&P 500, Dow Jones)

---

## System B: Mean Reversion (RSI Dip-Buying)

**Market Regime**: Range-bound / pullback opportunities in uptrending markets

### Logic
- **Long**: RSI < oversold + optional Stochastic confirmation + ADX < max threshold
- **Short**: RSI > overbought + optional Stochastic confirmation + ADX < max threshold
- **Exit**: ATR-based SL/TP

### Key Characteristics
- Trades quality over quantity on daily data
- Needs intraday timeframes (M15-H1) for sufficient trade frequency
- Optional Z-score mode as alternative to RSI+BB

---

## System C: Volatility/Breakout (Squeeze + Donchian)

**Market Regime**: Volatility compression → expansion (breakouts)

### Logic
- **Setup**: BB squeeze (BB inside KC) OR HV ratio compression OR ATR below average
- **Long**: Compression + close > Donchian high
- **Short**: Compression + close < Donchian low
- **Exit**: Tight SL, wide TP (favorable risk:reward)

### Key Characteristics
- Best risk:reward ratio (1:2.6+)
- Lowest drawdown
- Needs intraday data for sufficient squeeze events

---

## Strategy Diversity Matrix

| Aspect | System A (Trend) | System B (Mean Rev) | System C (Vol/Breakout) |
|--------|-----------------|--------------------|-----------------------|
| Market Regime | Trending | Pullbacks/Range | Compression→Expansion |
| Hold Period | Medium | Medium | Short-Medium |
| Signal Type | Momentum | Contrarian | Structural |
| Risk:Reward | ~1:0.6 (wide SL) | ~1:1 (symmetric) | ~1:2.6 (tight SL, wide TP) |
| Best Timeframe | TBD (run optimization) | TBD | TBD |

### Correlation Benefits
- Trend Following profits in sustained moves → underperforms in choppy markets
- Mean Reversion profits in pullbacks → underperforms in strong trends without dips
- Volatility Breakout profits at regime changes → uncorrelated with trend duration

Running all 3 simultaneously provides diversification across market regimes.

---

## Validation Pipeline

| Step | Method | Detail |
|------|--------|--------|
| Data Split | 70/15/15 chronological | Time-based, no shuffling |
| Walk-Forward | 5-fold expanding window | Sharpe consistency across time periods |
| Monte Carlo | 1000 trade-order shuffles | Median Sharpe, p5/p95 CI |
| Overfitting | Train/Val/Test degradation | Flag if >50% Sharpe degradation |
| Look-Ahead | Future-return correlation | Flag if correlation > 0.3 |
| Spread | 1.0 pip per trade | Applied on both entry and exit |

### Anti-Bias Measures
- All indicators use `.shift(1)` — no look-ahead
- Entry at open of bar AFTER signal bar
- Chronological splits only (no shuffling)
- Spread applied symmetrically
- Overfitting penalty in Optuna objective

---

## How to Run

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Export data from MT5 (see mt5/README.md)
#    Then convert CSVs to parquet:
python scripts/convert_mt5_csv.py --input-dir "/path/to/MQL5/Files"

# 3. Run multi-timeframe optimization (on your rig)
python scripts/run_multi_timeframe.py --trials 500 --cross-validate

# 4. Or optimize specific timeframes:
python scripts/run_multi_timeframe.py --trials 500 --timeframes 15m 1h

# 5. Train RL models locally (optional — uses rule-based as baseline)
python scripts/train_model_A.py --timesteps 500000
python scripts/train_model_B.py --timesteps 500000
python scripts/train_model_C.py --timesteps 500000

# 6. Deploy to MT5 (paper trading first)
python scripts/live_trading.py --strategy trend --symbol sp500 --mode paper
```

## Sharpe Calculation Fix

The Sharpe ratio is now calculated correctly:

```
trades_per_year = n_trades / (n_trading_days / 252)
sharpe = mean(returns) / std(returns) * sqrt(trades_per_year)
```

Previously it used `sqrt(252)` which is only correct for daily bar returns,
not for per-trade returns. This inflated Sharpe by ~3x when trades were
infrequent (e.g., 30 trades/year → sqrt(30) ≈ 5.5 vs sqrt(252) ≈ 15.9).
