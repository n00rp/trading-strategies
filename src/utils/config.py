"""Configuration loader utility."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Load YAML configuration file."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class CostModel:
    """Trading cost model for CFDs."""

    spread_pips: float = 1.0
    commission_per_trade: float = 0.0
    pip_value: float = 1.0  # varies per instrument

    def total_cost_per_trade(self) -> float:
        return self.spread_pips * self.pip_value + self.commission_per_trade

    def apply_spread(self, price: float, direction: int) -> float:
        """Adjust entry price for spread. direction: 1=buy, -1=sell."""
        half_spread = (self.spread_pips * self.pip_value) / 2
        return price + direction * half_spread
