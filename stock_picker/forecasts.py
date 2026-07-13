from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Forecast:
    ticker: str
    as_of: pd.Timestamp
    horizon: int
    model: str
    current_price: float
    expected_close: float
    range_low: float
    range_high: float
    expected_return: float
    confidence: float


class Forecaster(Protocol):
    name: str

    def forecast(self, ticker: str, history: pd.DataFrame, horizon: int) -> Forecast: ...


def _validate_history(history: pd.DataFrame, minimum_rows: int = 40) -> pd.DataFrame:
    if len(history) < minimum_rows:
        raise ValueError(f"At least {minimum_rows} observations are required; received {len(history)}.")
    if "close" not in history.columns:
        raise ValueError("History must contain a close column.")
    clean = history.sort_index().dropna(subset=["close"])
    if clean.empty or float(clean["close"].iloc[-1]) <= 0:
        raise ValueError("History does not contain a valid positive closing price.")
    return clean


def _stable_forecast_seed(base_seed: int, ticker: str, as_of: pd.Timestamp, horizon: int) -> int:
    """Create a repeatable per-symbol, per-date seed independent of Python hash randomization."""
    payload = f"{base_seed}|{ticker.upper()}|{pd.Timestamp(as_of).isoformat()}|{horizon}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


class TrendBaselineForecaster:
    """A transparent baseline using log-price trend and realized volatility.

    Kronos must beat this and a no-change baseline before it is considered useful.
    """

    name = "trend-baseline"

    def __init__(self, lookback: int = 126, range_probability: float = 0.80):
        self.lookback = lookback
        self.range_probability = range_probability

    def forecast(self, ticker: str, history: pd.DataFrame, horizon: int) -> Forecast:
        clean = _validate_history(history)
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")

        close = clean["close"].astype(float).tail(self.lookback)
        x = np.arange(len(close), dtype=float)
        log_prices = np.log(close.to_numpy())
        slope, intercept = np.polyfit(x, log_prices, 1)
        target_log_price = intercept + slope * (len(close) - 1 + horizon)
        expected_close = float(np.exp(target_log_price))
        current_price = float(close.iloc[-1])

        daily_returns = np.diff(log_prices)
        daily_volatility = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
        interval_move = 1.2816 * daily_volatility * sqrt(horizon)
        range_low = expected_close * exp(-interval_move)
        range_high = expected_close * exp(interval_move)
        expected_return = expected_close / current_price - 1.0

        trend_signal = abs(slope) * sqrt(horizon)
        noise = daily_volatility + 1e-9
        confidence = float(np.clip(trend_signal / noise, 0.0, 1.0))

        return Forecast(
            ticker=ticker,
            as_of=pd.Timestamp(clean.index[-1]),
            horizon=horizon,
            model=self.name,
            current_price=current_price,
            expected_close=expected_close,
            range_low=float(range_low),
            range_high=float(range_high),
            expected_return=float(expected_return),
            confidence=confidence,
        )


class KronosForecaster:
    """Adapter from stock-picker research code to the upstream Kronos predictor."""

    MODEL_SETTINGS = {
        "kronos-mini": {
            "model": "NeoQuasar/Kronos-mini",
            "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",
            "max_context": 2048,
        },
        "kronos-small": {
            "model": "NeoQuasar/Kronos-small",
            "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
            "max_context": 512,
        },
        "kronos-base": {
            "model": "NeoQuasar/Kronos-base",
            "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
            "max_context": 512,
        },
    }

    def __init__(
        self,
        variant: str = "kronos-mini",
        *,
        device: str | None = None,
        lookback: int = 400,
        sample_count: int = 5,
        temperature: float = 1.0,
        top_p: float = 0.9,
        seed: int = 42,
    ):
        if variant not in self.MODEL_SETTINGS:
            raise ValueError(f"Unsupported Kronos variant: {variant}")
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        self.variant = variant
        self.name = variant
        self.device = device
        self.lookback = lookback
        self.sample_count = sample_count
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        try:
            from model import Kronos, KronosPredictor, KronosTokenizer
        except ImportError as exc:
            raise RuntimeError("Run this command from the Kronos repository root.") from exc

        settings = self.MODEL_SETTINGS[self.variant]
        tokenizer = KronosTokenizer.from_pretrained(settings["tokenizer"])
        model = Kronos.from_pretrained(settings["model"])
        tokenizer.eval()
        model.eval()
        self._predictor = KronosPredictor(
            model,
            tokenizer,
            device=self.device,
            max_context=settings["max_context"],
        )
        return self._predictor

    def warmup(self) -> None:
        """Download model files and initialize the predictor."""
        self._load()

    def forecast(self, ticker: str, history: pd.DataFrame, horizon: int) -> Forecast:
        clean = _validate_history(history)
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")
        predictor = self._load()
        settings = self.MODEL_SETTINGS[self.variant]
        lookback = min(self.lookback, settings["max_context"], len(clean))
        context = clean.tail(lookback).copy()

        required = ["open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in context.columns]
        if missing:
            raise ValueError(f"Kronos history is missing columns: {missing}")
        context["amount"] = context["volume"] * context[["open", "high", "low", "close"]].mean(axis=1)

        as_of = pd.Timestamp(context.index[-1])
        forecast_seed = _stable_forecast_seed(self.seed, ticker, as_of, horizon)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Kronos forecasts.") from exc
        np.random.seed(forecast_seed % (2**32 - 1))
        torch.manual_seed(forecast_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(forecast_seed)

        x_timestamp = pd.Series(pd.to_datetime(context.index), index=context.index)
        future_index = pd.bdate_range(as_of + pd.Timedelta(days=1), periods=horizon)
        y_timestamp = pd.Series(future_index)
        predicted = predictor.predict(
            df=context[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=x_timestamp.reset_index(drop=True),
            y_timestamp=y_timestamp,
            pred_len=horizon,
            T=self.temperature,
            top_p=self.top_p,
            sample_count=self.sample_count,
            verbose=False,
        )

        current_price = float(context["close"].iloc[-1])
        expected_close = float(predicted["close"].iloc[-1])
        range_low = float(predicted["low"].min())
        range_high = float(predicted["high"].max())
        expected_return = expected_close / current_price - 1.0

        realized_volatility = float(np.log(context["close"]).diff().dropna().std())
        signal_to_noise = abs(log(max(expected_close, 1e-9) / current_price)) / (
            realized_volatility * sqrt(horizon) + 1e-9
        )
        confidence = float(np.clip(signal_to_noise / 2.0, 0.0, 1.0))

        return Forecast(
            ticker=ticker,
            as_of=as_of,
            horizon=horizon,
            model=self.name,
            current_price=current_price,
            expected_close=expected_close,
            range_low=range_low,
            range_high=range_high,
            expected_return=expected_return,
            confidence=confidence,
        )
