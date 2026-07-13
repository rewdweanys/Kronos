from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .forecasts import Forecaster


@dataclass(frozen=True)
class BacktestObservation:
    ticker: str
    as_of: pd.Timestamp
    target_date: pd.Timestamp
    horizon: int
    model: str
    current_price: float
    predicted_close: float
    actual_close: float
    predicted_return: float
    actual_return: float
    confidence: float
    range_low: float
    range_high: float
    range_hit: bool
    direction_correct: bool
    signal: int
    gross_strategy_return: float
    net_strategy_return: float


def walk_forward_backtest(
    ticker: str,
    history: pd.DataFrame,
    forecaster: Forecaster,
    horizon: int,
    *,
    minimum_history: int = 252,
    step: int = 20,
    max_points: int | None = None,
    signal_threshold: float = 0.03,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Evaluate forecasts using only data available at each historical decision date.

    The simulated strategy is long when predicted return is above ``signal_threshold``,
    short when it is below the negative threshold, and flat otherwise. Transaction cost
    is applied on both entry and exit.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if minimum_history < 40:
        raise ValueError("minimum_history must be at least 40.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be positive when supplied.")

    clean = history.sort_index().dropna(subset=["close"])
    end_positions = list(range(minimum_history - 1, len(clean) - horizon, step))
    if max_points is not None:
        end_positions = end_positions[-max_points:]

    round_trip_cost = 2.0 * transaction_cost_bps / 10_000.0
    rows: list[BacktestObservation] = []
    for end_position in end_positions:
        context = clean.iloc[: end_position + 1]
        forecast = forecaster.forecast(ticker, context, horizon)
        current_price = float(clean["close"].iloc[end_position])
        actual_close = float(clean["close"].iloc[end_position + horizon])
        actual_return = actual_close / current_price - 1.0

        if forecast.expected_return >= signal_threshold:
            signal = 1
        elif forecast.expected_return <= -signal_threshold:
            signal = -1
        else:
            signal = 0

        gross_strategy_return = signal * actual_return
        net_strategy_return = gross_strategy_return - (round_trip_cost if signal else 0.0)
        predicted_direction = int(np.sign(forecast.expected_return))
        actual_direction = int(np.sign(actual_return))

        rows.append(
            BacktestObservation(
                ticker=ticker,
                as_of=pd.Timestamp(clean.index[end_position]),
                target_date=pd.Timestamp(clean.index[end_position + horizon]),
                horizon=horizon,
                model=forecast.model,
                current_price=current_price,
                predicted_close=forecast.expected_close,
                actual_close=actual_close,
                predicted_return=forecast.expected_return,
                actual_return=actual_return,
                confidence=forecast.confidence,
                range_low=forecast.range_low,
                range_high=forecast.range_high,
                range_hit=forecast.range_low <= actual_close <= forecast.range_high,
                direction_correct=predicted_direction == actual_direction,
                signal=signal,
                gross_strategy_return=gross_strategy_return,
                net_strategy_return=net_strategy_return,
            )
        )

    return pd.DataFrame([asdict(row) for row in rows])


def summarize_backtest(observations: pd.DataFrame) -> pd.DataFrame:
    """Return one metrics row per ticker/model/horizon group."""
    if observations.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    group_columns = ["ticker", "model", "horizon"]
    for keys, group in observations.groupby(group_columns, sort=True):
        ticker, model, horizon = keys
        model_mae = float((group["predicted_return"] - group["actual_return"]).abs().mean())
        no_change_mae = float(group["actual_return"].abs().mean())
        signal_rows = group[group["signal"] != 0]
        if signal_rows.empty:
            signal_win_rate = np.nan
            average_net_return = 0.0
            cumulative_net_return = 0.0
        else:
            signal_win_rate = float((signal_rows["net_strategy_return"] > 0).mean())
            average_net_return = float(signal_rows["net_strategy_return"].mean())
            cumulative_net_return = float((1.0 + signal_rows["net_strategy_return"]).prod() - 1.0)

        correlation = float(group["predicted_return"].corr(group["actual_return"])) if len(group) > 1 else np.nan
        summary_rows.append(
            {
                "ticker": ticker,
                "model": model,
                "horizon": int(horizon),
                "observations": int(len(group)),
                "signals": int(len(signal_rows)),
                "directional_accuracy": float(group["direction_correct"].mean()),
                "range_coverage": float(group["range_hit"].mean()),
                "model_mae": model_mae,
                "no_change_mae": no_change_mae,
                "mae_improvement": no_change_mae - model_mae,
                "return_correlation": correlation,
                "signal_win_rate": signal_win_rate,
                "average_net_return": average_net_return,
                "cumulative_net_return": cumulative_net_return,
            }
        )

    return pd.DataFrame(summary_rows)
