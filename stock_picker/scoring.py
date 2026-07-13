from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .forecasts import Forecast


@dataclass(frozen=True)
class Recommendation:
    ticker: str
    as_of: pd.Timestamp
    action: str
    score: float
    confidence: float
    expected_return: float
    range_low: float
    range_high: float
    horizons: str


def aggregate_forecasts(
    forecasts: Iterable[Forecast],
    *,
    buy_threshold: float = 0.03,
    short_threshold: float = -0.03,
    minimum_confidence: float = 0.20,
) -> Recommendation:
    rows = list(forecasts)
    if not rows:
        raise ValueError("At least one forecast is required.")

    ticker = rows[0].ticker
    as_of = max(row.as_of for row in rows)
    if any(row.ticker != ticker for row in rows):
        raise ValueError("Forecasts must all belong to the same ticker.")

    weights = [max(row.confidence, 0.05) for row in rows]
    total_weight = sum(weights)
    expected_return = sum(row.expected_return * weight for row, weight in zip(rows, weights)) / total_weight
    confidence = sum(row.confidence * weight for row, weight in zip(rows, weights)) / total_weight
    range_low = min(row.range_low for row in rows)
    range_high = max(row.range_high for row in rows)

    score = expected_return * confidence
    if confidence < minimum_confidence:
        action = "NEUTRAL"
    elif expected_return >= buy_threshold:
        action = "BUY"
    elif expected_return <= short_threshold:
        action = "SHORT"
    else:
        action = "NEUTRAL"

    return Recommendation(
        ticker=ticker,
        as_of=as_of,
        action=action,
        score=float(score),
        confidence=float(confidence),
        expected_return=float(expected_return),
        range_low=float(range_low),
        range_high=float(range_high),
        horizons=",".join(str(row.horizon) for row in sorted(rows, key=lambda item: item.horizon)),
    )
