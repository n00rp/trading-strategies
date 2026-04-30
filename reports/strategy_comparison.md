# Strategy Comparison Report — Multi-Timeframe Evaluation

## OOS Criteria

| Metric | Strict | With ε=0.05 | With ε=0.10 |
|--------|--------|-------------|-------------|
| Sharpe | ≥ 1.20 | ≥ 1.14 | ≥ 1.08 |
| Profit Factor | ≥ 1.50 | ≥ 1.42 | ≥ 1.35 |
| Max Drawdown | ≤ 15.0% | ≤ 15.75% | ≤ 16.5% |

**Sharpe annualization**: `sqrt(trades_per_year)` (corrected — NOT `sqrt(252)`).
**Spread**: 1.0 pip included on all entries/exits.
**Test set**: 15% of data (chronological split, no shuffling).

---

## Results by Timeframe

### 5-Minute (M5)

| Symbol | Strategy | Sharpe | PF | MaxDD | Trades | T/wk | WR |
|--------|----------|--------|-----|-------|--------|------|----|
| SP500 | Trend | **7.91** | 1.33 | 9.6% | 8281 | 120.0 | 72.1% |
| SP500 | MeanRev | -3.58 | 0.59 | 32.4% | 577 | 8.4 | 15.1% |
| SP500 | Volatility | -0.39 | 0.96 | 6.9% | 699 | 10.1 | 41.1% |
| Nasdaq | Trend | **7.61** | 1.30 | 10.5% | 8337 | 120.5 | 71.6% |
| Nasdaq | MeanRev | -3.69 | 0.66 | 42.5% | 1004 | 14.5 | 14.8% |
| Nasdaq | Volatility | -1.37 | 0.86 | 13.8% | 712 | 10.3 | 40.2% |
| DowJones | Trend | 0.40 | 1.03 | 37.7% | 1631 | 34.6 | 70.1% |
| DowJones | MeanRev | -15.04 | 0.37 | 97.0% | 1895 | 40.1 | 14.8% |
| DowJones | Volatility | -0.47 | 0.92 | 9.2% | 133 | 2.8 | 43.6% |
| DAX | Trend | **6.09** | 1.22 | 7.8% | 8524 | 125.7 | 69.8% |
| DAX | MeanRev | -3.25 | 0.59 | 32.9% | 602 | 8.9 | 13.1% |
| DAX | Volatility | 1.06 | 1.12 | 6.0% | 689 | 10.2 | 43.3% |

**Notes**: Trend@5m has very high Sharpe but PF < 1.5 and trade frequency is extreme (~120/wk). Mean reversion at 5m is consistently negative — too much noise.

### 15-Minute (M15)

| Symbol | Strategy | Sharpe | PF | MaxDD | Trades | T/wk | WR |
|--------|----------|--------|-----|-------|--------|------|----|
| SP500 | Trend | 2.54 | 1.19 | 16.3% | 2184 | 31.6 | 67.6% |
| SP500 | MeanRev | -0.02 | 0.99 | 13.8% | 152 | 2.2 | 44.1% |
| SP500 | Volatility | 0.52 | 1.14 | 3.7% | 288 | 4.2 | 25.0% |
| Nasdaq | Trend | 2.30 | 1.16 | 14.7% | 2162 | 31.2 | 66.6% |
| Nasdaq | MeanRev | -0.02 | 1.00 | 13.5% | 209 | 3.0 | 45.0% |
| Nasdaq | Volatility | **1.45** | 1.34 | 3.2% | 317 | 4.6 | 29.7% |
| DowJones | Trend | 1.44 | 1.17 | 24.5% | 453 | 9.6 | 67.3% |
| DowJones | MeanRev | -0.88 | 0.85 | 32.6% | 145 | 3.1 | 46.2% |
| DowJones | Volatility | -0.83 | 0.77 | 13.8% | 62 | 1.3 | 19.4% |
| DAX | Trend | **2.96** | 1.20 | 13.4% | 2145 | 31.6 | 67.2% |
| DAX | MeanRev | -1.27 | 0.77 | 17.6% | 168 | 2.5 | 39.9% |
| DAX | Volatility | 0.46 | 1.11 | 3.8% | 268 | 4.0 | 23.9% |

### 30-Minute (M30)

| Symbol | Strategy | Sharpe | PF | MaxDD | Trades | T/wk | WR |
|--------|----------|--------|-----|-------|--------|------|----|
| SP500 | Trend | 1.57 | 1.16 | 17.4% | 939 | 13.5 | 63.0% |
| SP500 | MeanRev | 0.76 | 1.11 | 8.1% | 797 | 11.5 | 16.4% |
| SP500 | Volatility | -0.05 | 0.99 | 12.0% | 244 | 3.5 | 41.4% |
| Nasdaq | Trend | 1.49 | 1.14 | 18.8% | 1002 | 14.4 | 62.2% |
| Nasdaq | MeanRev | 1.11 | 1.15 | 7.6% | 959 | 13.8 | 17.6% |
| Nasdaq | Volatility | **1.18** | 1.23 | 8.2% | 249 | 3.6 | 48.2% |
| DowJones | Trend | 0.15 | 1.02 | 32.5% | 236 | 5.0 | 60.6% |
| DowJones | MeanRev | 0.49 | 1.12 | 17.0% | 151 | 3.2 | 17.9% |
| DowJones | Volatility | -0.00 | 1.00 | 27.1% | 44 | 0.9 | 45.5% |
| DAX | Trend | 2.56 | 1.25 | 27.5% | 992 | 14.5 | 65.6% |
| DAX | MeanRev | 0.31 | 1.04 | 10.7% | 814 | 11.9 | 16.8% |
| DAX | Volatility | 0.96 | 1.19 | 6.3% | 238 | 3.5 | 46.2% |

### Hourly (H1)

| Symbol | Strategy | Sharpe | PF | MaxDD | Trades | T/wk | WR |
|--------|----------|--------|-----|-------|--------|------|----|
| SP500 | Trend | 0.24 | 1.02 | 11.3% | 988 | 14.2 | 40.1% |
| SP500 | MeanRev | 0.73 | 1.22 | 3.7% | 194 | 2.8 | 22.9% |
| SP500 | Volatility | 0.71 | 1.15 | 11.4% | 208 | 3.0 | 38.0% |
| **Nasdaq** | **Trend** | 0.29 | 1.03 | 20.6% | 981 | 14.1 | 39.9% |
| **Nasdaq** | **MeanRev** | **1.97** | **1.54** | **3.9%** | **219** | **3.1** | **28.8%** |
| Nasdaq | Volatility | 0.26 | 1.05 | 15.3% | 189 | 2.7 | 36.0% |
| DowJones | Trend | 0.88 | 1.13 | 29.2% | 248 | 5.3 | 47.6% |
| DowJones | MeanRev | 0.77 | 1.42 | 3.5% | 29 | 0.6 | 31.0% |
| DowJones | Volatility | -0.24 | 0.90 | 13.2% | 25 | 0.5 | 32.0% |
| DAX | Trend | -0.05 | 1.00 | 14.1% | 978 | 14.3 | 41.8% |
| DAX | MeanRev | 0.83 | 1.24 | 3.4% | 177 | 2.6 | 24.3% |
| DAX | Volatility | 0.26 | 1.05 | 14.0% | 204 | 3.0 | 37.7% |

---

## Strategies Meeting OOS Criteria

### Strict (ε=0)
| Combo | Sharpe | PF | MaxDD | T/wk |
|-------|--------|-----|-------|------|
| **MeanRev@H1/Nasdaq** | **1.97** | **1.54** | **3.9%** | 3.1 |

### Close to criteria (top 10 by composite score)
| Combo | Sharpe | PF | MaxDD | T/wk | Limiting factor |
|-------|--------|-----|-------|------|-----------------|
| Trend@5m/SP500 | 7.91 | 1.33 | 9.6% | 120.0 | PF < 1.5, too many trades |
| Trend@5m/Nasdaq | 7.61 | 1.30 | 10.5% | 120.5 | PF < 1.5, too many trades |
| Trend@5m/DAX | 6.09 | 1.22 | 7.8% | 125.7 | PF < 1.5, too many trades |
| Trend@15m/DAX | 2.96 | 1.20 | 13.4% | 31.6 | PF < 1.5 |
| Trend@15m/SP500 | 2.54 | 1.19 | 16.3% | 31.6 | PF < 1.5, MaxDD > 15% |
| Trend@30m/DAX | 2.56 | 1.25 | 27.5% | 14.5 | PF < 1.5, MaxDD > 15% |
| Vol@15m/Nasdaq | 1.45 | 1.34 | 3.2% | 4.6 | PF < 1.5 |
| Vol@30m/Nasdaq | 1.18 | 1.23 | 8.2% | 3.6 | Sharpe < 1.2, PF < 1.5 |
| MR@30m/Nasdaq | 1.11 | 1.15 | 7.6% | 13.8 | Sharpe < 1.2, PF < 1.5 |

---

## Missed Trades Analysis

For strategies with the most potential (30m and 1h timeframes):

### Key Findings:
- **Trend Following** misses trades primarily due to ADX filter being too strict (ADX threshold filtering out many entries where the trend later materialized)
- **Mean Reversion** misses trades because z-score thresholds are too conservative and long-only mode eliminates short opportunities
- **Volatility Breakout** misses the most trades — main reason is no Donchian breakout + no squeeze detected (the compression detection is too strict)

### Missed Trade Counts (1h timeframe, test period):
| Symbol | Trend | MeanRev | Volatility |
|--------|-------|---------|------------|
| Nasdaq | 3,422 (1.28% avg) | 6,059 (1.32% avg) | 5,814 (1.32% avg) |
| DAX | 3,202 (1.20% avg) | 5,533 (1.20% avg) | 4,634 (1.26% avg) |
| SP500 | 2,759 (1.10% avg) | 5,167 (1.15% avg) | 4,384 (1.19% avg) |

### Top Reasons for Missed Trades:
1. **Volatility**: `no_breakout + no_squeeze_detected` — the compression filter + Donchian breakout requirement is too restrictive
2. **MeanRev**: `long_only_mode` — missing profitable short trades; `zscore` threshold too conservative
3. **Trend**: `adx <= threshold` — ADX filter too strict, missing trends that develop after a low-ADX environment

---

## Recommendations

1. **MeanRev@H1/Nasdaq** is the only strict pass. Consider:
   - Allowing short trades (removing `long_only`)
   - Relaxing z-score threshold slightly to increase trade frequency from 3.1/wk
   - Re-optimizing with Optuna on the other symbols

2. **Trend Following** has excellent Sharpe on shorter timeframes (5m, 15m) but PF is consistently below 1.5. The high win rate (~70%) with low PF suggests the winners are small. Consider:
   - Widening take-profit targets
   - Tighter stop losses to improve PF
   - Targeting 10-20 trades/week instead of 120

3. **Volatility Breakout** shows promise on 15m/30m Nasdaq. Consider:
   - Relaxing compression detection
   - Using pure ATR expansion without requiring Donchian breakout
   - Adding momentum confirmation instead of squeeze

4. **Re-optimize with Optuna** targeting PF ≥ 1.5 as primary objective instead of Sharpe.
