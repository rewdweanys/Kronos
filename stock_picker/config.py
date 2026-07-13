from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchConfig:
    """Defaults for the first research version.

    All horizons are trading days. Nothing in this configuration places orders.
    """

    horizons: tuple[int, ...] = (5, 10, 20)
    lookback: int = 252
    minimum_history: int = 300
    minimum_edge: float = 0.03
    round_trip_cost_bps: float = 10.0
    output_directory: Path = Path("outputs/stock_picker")


DEFAULT_CONFIG = ResearchConfig()
