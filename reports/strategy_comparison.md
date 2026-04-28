# Strategy Comparison Report

## Overview

Three distinct trading strategies validated on 10+ years of daily data (2014–2024) for major indices via CFD. Each targets a different market regime to ensure portfolio diversification.

---

## System A: Trend Following (EMA Cross + ADX + MACD)

**Market Regime**: Trending markets (sustained directional moves)

### Logic
- **Entry**: EMA fast crosses above/below EMA slow, confirmed by ADX > threshold (strong trend) and MACD histogram alignment
- **Exit**: ATR-based stop loss (tight) and take profit (wider, ~2–3x ATR for favorable risk:reward)
- **Edge**: Captures large directional moves while filtering out weak/choppy markets via ADX

### Key Parameters (Optimizable via Optuna)
| Parameter | Range | Description |
|-----------|-------|-------------|
| `ema_fast` | 8–21 | Fast EMA period |
| `ema_slow` | 34–89 | Slow EMA period |
| `adx_threshold` | 15–35 | Minimum ADX for trend strength |
| `atr_sl_mult` | 1.0–3.0 | Stop loss as multiple of ATR |
| `atr_tp_mult` | 1.5–5.0 | Take profit as multiple of ATR |

### Features Used
- EMA fast/slow and their difference
- ADX, +DI, -DI (trend strength)
- MACD, MACD signal, MACD histogram
- ATR (for position sizing and exits)
- Trend regime classification

### Anti-Bias Measures
- Signals generated from shifted indicators (no look-ahead)
- Entry executes at open of the bar AFTER the signal bar
- Spread cost applied on both entry and exit

---

## System B: Mean Reversion (RSI + Bollinger Bands)

**Market Regime**: Range-bound / oscillating markets

### Logic
- **Entry LONG**: RSI < oversold (30) AND price below lower Bollinger Band AND Stochastic K < 20 crossing above D
- **Entry SHORT**: RSI > overbought (70) AND price above upper Bollinger Band AND Stochastic K > 80 crossing below D
- **Exit**: Price returns to BB midline (mean target) or ATR-based stop loss
- **Edge**: Captures mean-reversion bounces at statistical extremes with multiple confirmation signals

### Key Parameters (Optimizable via Optuna)
| Parameter | Range | Description |
|-----------|-------|-------------|
| `rsi_period` | 7–21 | RSI lookback period |
| `rsi_oversold` | 20–35 | RSI oversold threshold |
| `rsi_overbought` | 65–80 | RSI overbought threshold |
| `bb_period` | 14–30 | Bollinger Band period |
| `bb_std` | 1.5–2.5 | Bollinger Band standard deviations |
| `min_bb_width` | 0.01–0.05 | Minimum BB width filter |

### Features Used
- RSI (momentum oscillator)
- Bollinger Bands (upper, lower, mid, width, %B)
- Z-score of price relative to moving average
- Stochastic oscillator (K, D)
- ATR (for stop loss sizing)

### Anti-Bias Measures
- BB width filter prevents trading in tight trending markets
- Stochastic confirmation adds timing precision
- Target is BB midline (natural mean)

---

## System C: Volatility/Breakout (Squeeze + Donchian)

**Market Regime**: Volatility compression → expansion (breakouts)

### Logic
- **Setup**: Detect volatility compression via:
  1. BB squeeze (Bollinger Bands inside Keltner Channel)
  2. Historical volatility contraction (HV20 < HV50)
  3. ATR compression below its own moving average
- **Entry**: After compression releases, enter in direction of Donchian channel breakout
  - LONG: Close > 20-period Donchian high
  - SHORT: Close < 20-period Donchian low
- **Exit**: ATR-based stop loss and take profit
- **Edge**: Identifies periods of coiling energy (compression) followed by explosive moves

### Key Parameters (Optimizable via Optuna)
| Parameter | Range | Description |
|-----------|-------|-------------|
| `squeeze_lookback` | 3–10 | Bars to check for recent squeeze |
| `vol_ratio_threshold` | 0.5–1.2 | HV20/HV50 compression threshold |
| `bb_period` | 15–30 | BB period for squeeze detection |
| `kc_period` | 15–30 | KC period for squeeze detection |
| `bb_mult` | 1.5–2.5 | BB multiplier |
| `kc_mult` | 1.0–2.0 | KC multiplier |
| `atr_sl_mult` | 1.0–3.0 | Stop loss ATR multiple |
| `atr_tp_mult` | 1.5–4.0 | Take profit ATR multiple |

### Features Used
- Bollinger Bands / Keltner Channel (squeeze detection)
- Historical volatility ratio (HV20/HV50)
- Donchian Channel (breakout levels)
- ATR and ATR moving average (volatility expansion)
- Volume ratio (optional confirmation)

### Anti-Bias Measures
- Donchian high/low shifted by 1 bar to prevent look-ahead
- Multiple compression detection methods for robustness
- Volume filter is optional (disabled by default for indices)

---

## Validation Pipeline

All strategies go through identical rigorous validation:

### 1. Data Split (70/15/15)
- **Train** (70%): 2014–2021 — parameter optimization
- **Validation** (15%): 2021–2023 — hyperparameter selection
- **Test** (15%): 2023–2024 — final OOS evaluation
- **Chronological only** — no shuffling to prevent look-ahead bias

### 2. Walk-Forward Analysis
- 5-fold expanding window validation
- Each fold: train on expanding history, evaluate on next unseen period
- Reports Sharpe ratio consistency across folds

### 3. Monte Carlo Stress Test
- 1000 simulations shuffling trade order
- Reports: median Sharpe, 5th/95th percentile, probability of profitability
- Ensures performance isn't dependent on lucky trade sequencing

### 4. Overfitting Detection
- Compares train vs. val vs. test Sharpe ratios
- Flags if degradation > 50% between train and OOS
- Reports val/test consistency

### 5. Look-Ahead Bias Check
- Statistical correlation test between signals and FUTURE returns
- Correlation > 0.3 = suspicious (flag as potential look-ahead)

---

## OOS Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Sharpe Ratio | > 1.2 | Risk-adjusted return above market |
| Profit Factor | > 1.5 | Gross profits 1.5x gross losses |
| Max Drawdown | < 15% | Capital preservation |

All metrics computed **including 1.0 pip spread per trade** (representative of major index CFDs).

---

## Technology Stack

| Component | Tool |
|-----------|------|
| Data | yfinance + parquet storage |
| Indicators | `ta` library (Technical Analysis) |
| Backtesting | Custom engine with spread/fee modeling |
| Optimization | Optuna (TPE sampler, Median pruner) |
| RL Training | Gymnasium + Stable Baselines3 (PPO) |
| Execution | MetaTrader 5 Python API |

---

## How to Run

```bash
# 1. Fetch data
python scripts/fetch_data.py

# 2. Discover & optimize (full 200 trials per strategy)
python scripts/discover_strategies.py --trials 200 --symbol sp500

# 3. Train RL models locally
python scripts/train_model_A.py --timesteps 500000
python scripts/train_model_B.py --timesteps 500000
python scripts/train_model_C.py --timesteps 500000

# 4. Deploy to MT5 (paper first)
python scripts/live_trading.py --strategy trend --symbol sp500 --mode paper
```

---

## Strategy Diversity Matrix

| Aspect | System A (Trend) | System B (Mean Rev) | System C (Vol/Breakout) |
|--------|-----------------|--------------------|-----------------------|
| Market Regime | Trending | Range-bound | Compression→Expansion |
| Hold Period | Medium-Long | Short | Short-Medium |
| Signal Type | Momentum | Contrarian | Structural |
| Risk:Reward | 1:2 to 1:3 | 1:1 to 1:2 | 1:1.5 to 1:2.5 |
| Trade Frequency | Medium | Low-Medium | Low |
| Correlation | Profits in trends | Profits in ranges | Profits at regime changes |

The three systems are designed to be complementary: when one underperforms (e.g., trend strategy in choppy markets), others should compensate.
