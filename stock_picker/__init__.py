"""Research-first stock ranking tools built around the Kronos model."""

from .backtest import BacktestObservation, summarize_backtest, walk_forward_backtest
from .baselines import MeanReversionForecaster, MomentumForecaster
from .forecasts import (
    Forecast,
    KronosEnsembleForecaster,
    KronosForecaster,
    TrendBaselineForecaster,
    aggregate_ensemble_forecasts,
)
from .scoring import Recommendation, aggregate_forecasts

__all__ = [
    "Forecast",
    "KronosForecaster",
    "KronosEnsembleForecaster",
    "TrendBaselineForecaster",
    "MomentumForecaster",
    "MeanReversionForecaster",
    "aggregate_ensemble_forecasts",
    "Recommendation",
    "aggregate_forecasts",
    "BacktestObservation",
    "walk_forward_backtest",
    "summarize_backtest",
]
