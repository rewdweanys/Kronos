from __future__ import annotations

from math import exp, sqrt
from typing import Iterable

import numpy as np
import pandas as pd

from .forecasts import Forecast, _validate_history


def _forecast_from_log_return(
    *,
    ticker: str,
    history: pd.DataFrame,
    horizon: int,
    model_name: str,
    expected_log_return: float,
    volatility_lookback: int = 126,
) -> Forecast:
    clean = _validate_history(history)
    if horizon <= 0:
        raise ValueError("Horizon must be positive.")

    close = clean["close"].astype(float)
    current_price = float(close.iloc[-1])
    expected_log_return = float(np.clip(expected_log_return, -0.25, 0.25))
    expected_close = float(current_price * exp(expected_log_return))

    log_returns = np.log(close).diff().dropna().tail(volatility_lookback)
    daily_volatility = float(log_returns.std(ddof=1)) if len(log_returns) > 1 else 0.0
    interval_move = 1.2816 * daily_volatility * sqrt(horizon)
    range_low = float(expected_close * exp(-interval_move))
    range_high = float(expected_close * exp(interval_move))
    expected_return = float(expected_close / current_price - 1.0)

    noise = daily_volatility * sqrt(horizon) + 1e-9
    confidence = float(np.clip(abs(expected_log_return) / (2.0 * noise), 0.0, 1.0))

    return Forecast(
        ticker=ticker,
        as_of=pd.Timestamp(clean.index[-1]),
        horizon=horizon,
        model=model_name,
        current_price=current_price,
        expected_close=expected_close,
        range_low=range_low,
        range_high=range_high,
        expected_return=expected_return,
        confidence=confidence,
    )


class MomentumForecaster:
    """Transparent volatility-aware momentum benchmark over several lookback windows."""

    name = "momentum-baseline"

    def __init__(
        self,
        lookbacks: Iterable[int] = (21, 63, 126),
        weights: Iterable[float] = (0.50, 0.30, 0.20),
    ):
        self.lookbacks = tuple(int(value) for value in lookbacks)
        self.weights = tuple(float(value) for value in weights)
        if not self.lookbacks or len(self.lookbacks) != len(self.weights):
            raise ValueError("lookbacks and weights must be non-empty and have equal length.")
        if any(value <= 0 for value in self.lookbacks):
            raise ValueError("Momentum lookbacks must be positive.")
        if any(value < 0 for value in self.weights) or sum(self.weights) <= 0:
            raise ValueError("Momentum weights must be non-negative and sum to a positive value.")

    def forecast(self, ticker: str, history: pd.DataFrame, horizon: int) -> Forecast:
        clean = _validate_history(history)
        close = clean["close"].astype(float)
        current_price = float(close.iloc[-1])

        drifts: list[float] = []
        usable_weights: list[float] = []
        for lookback, weight in zip(self.lookbacks, self.weights):
            if len(close) <= lookback:
                continue
            prior_price = float(close.iloc[-lookback - 1])
            if prior_price <= 0:
                continue
            drifts.append(float(np.log(current_price / prior_price) / lookback))
            usable_weights.append(weight)

        if not drifts:
            raise ValueError("History is too short for the configured momentum lookbacks.")

        normalized_weights = np.asarray(usable_weights, dtype=float)
        normalized_weights /= normalized_weights.sum()
        daily_drift = float(np.dot(np.asarray(drifts, dtype=float), normalized_weights))
        expected_log_return = daily_drift * horizon
        return _forecast_from_log_return(
            ticker=ticker,
            history=clean,
            horizon=horizon,
            model_name=self.name,
            expected_log_return=expected_log_return,
        )


class MeanReversionForecaster:
    """Transparent short-term benchmark that predicts partial reversion to a log-price mean."""

    name = "mean-reversion-baseline"

    def __init__(self, lookback: int = 20, reversion_strength: float = 0.50):
        if lookback <= 1:
            raise ValueError("lookback must be greater than one.")
        if not 0.0 <= reversion_strength <= 1.0:
            raise ValueError("reversion_strength must be between zero and one.")
        self.lookback = int(lookback)
        self.reversion_strength = float(reversion_strength)

    def forecast(self, ticker: str, history: pd.DataFrame, horizon: int) -> Forecast:
        clean = _validate_history(history)
        close = clean["close"].astype(float)
        if len(close) < self.lookback:
            raise ValueError(f"At least {self.lookback} observations are required.")

        recent_log_prices = np.log(close.tail(self.lookback).to_numpy())
        current_log_price = float(recent_log_prices[-1])
        anchor_log_price = float(np.mean(recent_log_prices))
        horizon_fraction = min(float(horizon) / float(self.lookback), 1.0)
        expected_log_return = (
            anchor_log_price - current_log_price
        ) * self.reversion_strength * horizon_fraction

        return _forecast_from_log_return(
            ticker=ticker,
            history=clean,
            horizon=horizon,
            model_name=self.name,
            expected_log_return=expected_log_return,
        )
