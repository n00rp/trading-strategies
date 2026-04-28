"""MetaTrader 5 integration for live/paper trading.

Provides:
  - Connection management (IC Markets / IG broker setup)
  - Order execution with spread-aware pricing
  - Signal-to-order translation from trained models
  - Position management

NOTE: MT5 Python API only works on Windows. For Linux deployment,
use the MT5 REST API bridge or Wine-based wrapper.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"


@dataclass
class MT5Config:
    """MT5 connection configuration."""

    server: str = "ICMarketsSC-Demo"  # or "IG-Demo"
    login: int = 0
    password: str = ""
    path: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    symbol_map: dict | None = None

    def __post_init__(self):
        if self.symbol_map is None:
            # Default symbol mapping for IC Markets
            self.symbol_map = {
                "sp500": "US500",
                "nasdaq": "USTEC",
                "dowjones": "US30",
                "dax": "DE40",
            }


class MT5Connector:
    """MT5 connection and order management.

    Usage:
        connector = MT5Connector(config)
        connector.connect()
        connector.execute_signal("sp500", signal=1, lot_size=0.1)
        connector.disconnect()
    """

    def __init__(self, config: MT5Config):
        self.config = config
        self.mt5 = None
        self.connected = False

    def connect(self) -> bool:
        """Initialize MT5 connection."""
        try:
            import MetaTrader5 as mt5

            self.mt5 = mt5

            if not mt5.initialize(
                path=self.config.path,
                login=self.config.login,
                server=self.config.server,
                password=self.config.password,
            ):
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False

            info = mt5.account_info()
            if info is None:
                logger.error("Failed to get account info")
                return False

            logger.info(
                f"Connected to MT5: {info.server} | "
                f"Account: {info.login} | Balance: {info.balance}"
            )
            self.connected = True
            return True

        except ImportError:
            logger.warning(
                "MetaTrader5 package not available. "
                "Install on Windows: pip install MetaTrader5"
            )
            return False

    def disconnect(self):
        if self.mt5 and self.connected:
            self.mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")

    def get_symbol(self, name: str) -> str:
        """Map internal symbol name to MT5 symbol."""
        return self.config.symbol_map.get(name, name)

    def execute_signal(
        self,
        symbol_name: str,
        signal: int,
        lot_size: float = 0.1,
        sl_pips: float = 50,
        tp_pips: float = 100,
    ) -> dict | None:
        """Execute a trading signal.

        Args:
            symbol_name: Internal symbol name (e.g., 'sp500')
            signal: 1=buy, -1=sell, 0=close
            lot_size: Position size in lots
            sl_pips: Stop loss in pips
            tp_pips: Take profit in pips
        """
        if not self.connected or self.mt5 is None:
            logger.error("Not connected to MT5")
            return None

        mt5 = self.mt5
        symbol = self.get_symbol(symbol_name)

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None

        point = mt5.symbol_info(symbol).point

        if signal == 1:  # Buy
            price = tick.ask
            sl = price - sl_pips * point
            tp = price + tp_pips * point
            order_type = mt5.ORDER_TYPE_BUY
        elif signal == -1:  # Sell
            price = tick.bid
            sl = price + sl_pips * point
            tp = price - tp_pips * point
            order_type = mt5.ORDER_TYPE_SELL
        elif signal == 0:  # Close all positions for symbol
            return self._close_positions(symbol)
        else:
            return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "trading_strategy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None

        logger.info(
            f"Order executed: {symbol} {'BUY' if signal == 1 else 'SELL'} "
            f"{lot_size} lots @ {price:.2f} | SL={sl:.2f} TP={tp:.2f}"
        )

        return {
            "ticket": result.order,
            "symbol": symbol,
            "direction": signal,
            "price": price,
            "sl": sl,
            "tp": tp,
            "lot_size": lot_size,
        }

    def _close_positions(self, symbol: str) -> dict | None:
        """Close all open positions for a symbol."""
        mt5 = self.mt5
        positions = mt5.positions_get(symbol=symbol)

        if positions is None or len(positions) == 0:
            logger.info(f"No open positions for {symbol}")
            return {"closed": 0}

        closed = 0
        for pos in positions:
            if pos.type == mt5.ORDER_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": "close_position",
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1

        return {"closed": closed}


class InferenceEngine:
    """Run trained model inference and generate MT5 signals.

    This bridges the trained RL/rule-based models with MT5 execution.
    """

    def __init__(
        self,
        strategy_type: str,
        model_path: str | None = None,
        params: dict | None = None,
    ):
        self.strategy_type = strategy_type
        self.model_path = model_path
        self.params = params or {}
        self.model = None
        self.strategy = None

    def load_model(self):
        """Load trained model (RL) or initialize rule-based strategy."""
        if self.model_path:
            # Load SB3 model
            from stable_baselines3 import PPO

            self.model = PPO.load(self.model_path)
            logger.info(f"Loaded RL model from {self.model_path}")
        else:
            # Use rule-based strategy
            if self.strategy_type == "trend":
                from src.strategies.trend_following import TrendFollowingStrategy

                self.strategy = TrendFollowingStrategy(**self.params)
            elif self.strategy_type == "mean_reversion":
                from src.strategies.mean_reversion import MeanReversionStrategy

                self.strategy = MeanReversionStrategy(**self.params)
            elif self.strategy_type == "volatility":
                from src.strategies.volatility_breakout import VolatilityBreakoutStrategy

                self.strategy = VolatilityBreakoutStrategy(**self.params)

    def predict(self, features: pd.DataFrame) -> int:
        """Generate trading signal from current features.

        Returns: 1=buy, -1=sell, 0=hold
        """
        if self.model is not None:
            # RL model inference
            obs = features.values[-1].astype(np.float32)
            # Append dummy position info
            obs = np.concatenate([obs, [0.0, 0.0]])
            action, _ = self.model.predict(obs, deterministic=True)
            return {0: 0, 1: 1, 2: -1}.get(int(action), 0)

        elif self.strategy is not None:
            # Rule-based strategy
            signals = self.strategy.generate_signals(features)
            return int(signals.iloc[-1])

        return 0
