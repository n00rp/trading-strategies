# Trading Strategies — CFD Index Trading Systems

Three validated trading strategies for major indices (S&P 500, Nasdaq, Dow Jones, DAX) via CFD, combining rule-based systems with reinforcement learning.

## Strategy Overview

| System | Type | Market Regime | Core Logic |
|--------|------|--------------|------------|
| **A** | Trend Following | Trending | EMA crossover + ADX filter + MACD confirmation |
| **B** | Mean Reversion | Range-bound | RSI extremes + Bollinger Band reversals + Stochastic |
| **C** | Volatility/Breakout | Compression→Expansion | BB/KC squeeze + Donchian breakout |

## Project Structure

```
trading-strategies/
├── configs/
│   └── default.yaml          # All strategy parameters & training config
├── scripts/
│   ├── fetch_data.py          # Download market data (run first)
│   ├── discover_strategies.py # Optuna optimization + validation
│   ├── train_model_A.py       # Standalone RL training: Trend Following
│   ├── train_model_B.py       # Standalone RL training: Mean Reversion
│   ├── train_model_C.py       # Standalone RL training: Volatility Breakout
│   └── live_trading.py        # MT5 live/paper trading
├── src/
│   ├── data/fetcher.py        # yfinance data fetching
│   ├── features/engineering.py # Feature engineering (anti-bias)
│   ├── strategies/
│   │   ├── base.py            # Base strategy + backtester
│   │   ├── trend_following.py # Strategy A implementation
│   │   ├── mean_reversion.py  # Strategy B implementation
│   │   └── volatility_breakout.py # Strategy C implementation
│   ├── validation/pipeline.py # Train/Val/Test, WFA, Monte Carlo
│   ├── optimization/optuna_optimizer.py # Optuna HPO
│   ├── environments/trading_env.py # Gymnasium RL environment
│   ├── mt5/connector.py       # MetaTrader 5 integration
│   └── utils/config.py        # Configuration loader
├── data/                      # Market data (gitignored)
├── models/                    # Trained models (gitignored)
├── reports/                   # Validation reports
└── tests/                     # Unit tests
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Fetch Data

```bash
python scripts/fetch_data.py
```

Downloads 10+ years of daily data and available hourly data for all four indices.

### 3. Discover & Optimize Strategies

```bash
python scripts/discover_strategies.py --trials 200 --symbol sp500
```

Runs Optuna optimization for all 3 strategy types with full validation:
- 70/15/15 chronological split
- Walk-forward analysis (5 folds)
- Monte Carlo stress test (1000 simulations)
- Overfitting detection
- Look-ahead bias check

### 4. Train RL Models (Local)

Each training script is standalone with hardcoded hyperparameters:

```bash
python scripts/train_model_A.py --timesteps 500000
python scripts/train_model_B.py --timesteps 500000
python scripts/train_model_C.py --timesteps 500000
```

### 5. Deploy to MT5

```bash
# Paper trading
python scripts/live_trading.py --strategy trend --symbol sp500 --mode paper

# Live trading (update MT5_CONFIG in script first)
python scripts/live_trading.py --strategy trend --symbol sp500 --mode live
```

## Validation Criteria (OOS)

All strategies must meet on the **test set** (with 1.0 pip spread):

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 1.2 |
| Profit Factor | > 1.5 |
| Max Drawdown | < 15% |

## Anti-Bias Measures

1. **No Look-Ahead**: All indicators use `.shift(1)` where needed; signals from bar `t` use only data ≤ `t`
2. **Chronological Splits**: No shuffling — train/val/test are in temporal order
3. **Walk-Forward Analysis**: Expanding window validation across 5 time folds
4. **Monte Carlo**: 1000 trade-order shuffles to verify robustness
5. **Overfitting Detection**: Sharpe degradation check between train/val/test
6. **Statistical Look-Ahead Test**: Correlation between signals and future returns

## Tech Stack

- **Data**: yfinance, pandas, parquet
- **Indicators**: ta (Technical Analysis Library)
- **RL**: Gymnasium, Stable Baselines3 (PPO)
- **Optimization**: Optuna (TPE sampler)
- **Backtesting**: Custom engine with spread/fee modeling
- **Execution**: MetaTrader 5 Python API
- **Validation**: scipy, numpy (Monte Carlo, statistical tests)

## MT5 Broker Setup

Configured for **IC Markets** (demo). Update `MT5_CONFIG` in `scripts/live_trading.py`:

```python
MT5_CONFIG = {
    "server": "ICMarketsSC-Demo",  # or "ICMarketsSC-Live"
    "login": YOUR_ACCOUNT,
    "password": "YOUR_PASSWORD",
    "path": r"C:\Program Files\MetaTrader 5\terminal64.exe",
}
```

Symbol mapping (IC Markets):
- S&P 500 → `US500`
- Nasdaq → `USTEC`
- Dow Jones → `US30`
- DAX → `DE40`
