import numpy as np
import pandas as pd

from stock_picker.backtest import summarize_backtest, walk_forward_backtest
from stock_picker.data import normalize_ohlcv
from stock_picker.forecasts import TrendBaselineForecaster, _stable_forecast_seed
from stock_picker.scoring import aggregate_forecasts


def make_history(rows: int = 180, daily_return: float = 0.001) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=rows)
    close = 100 * np.exp(np.arange(rows) * daily_return)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(rows, 1_000_000),
        },
        index=dates,
    )


def test_baseline_forecast_is_well_formed():
    forecast = TrendBaselineForecaster().forecast("TEST", make_history(), 10)
    assert forecast.expected_close > forecast.current_price
    assert forecast.range_low <= forecast.expected_close <= forecast.range_high
    assert 0 <= forecast.confidence <= 1


def test_aggregate_forecasts_returns_buy_for_positive_trend():
    model = TrendBaselineForecaster()
    history = make_history(daily_return=0.003)
    result = aggregate_forecasts([model.forecast("TEST", history, horizon) for horizon in (5, 10, 20)])
    assert result.action == "BUY"
    assert result.expected_return > 0


def test_normalize_ohlcv_accepts_single_ticker_multiindex_field_first():
    history = make_history(rows=5)
    raw = history.copy()
    raw.columns = pd.MultiIndex.from_tuples(
        [(column.title(), "AAPL") for column in raw.columns], names=["Price", "Ticker"]
    )

    normalized = normalize_ohlcv(raw)

    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert normalized.shape == history.shape


def test_normalize_ohlcv_accepts_single_ticker_multiindex_ticker_first():
    history = make_history(rows=5)
    raw = history.copy()
    raw.columns = pd.MultiIndex.from_tuples(
        [("AAPL", column.title()) for column in raw.columns], names=["Ticker", "Price"]
    )

    normalized = normalize_ohlcv(raw)

    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert normalized.shape == history.shape


def test_stable_forecast_seed_is_repeatable_and_horizon_specific():
    as_of = pd.Timestamp("2026-07-10")
    first = _stable_forecast_seed(42, "AAPL", as_of, 5)
    second = _stable_forecast_seed(42, "AAPL", as_of, 5)
    different_horizon = _stable_forecast_seed(42, "AAPL", as_of, 10)

    assert first == second
    assert first != different_horizon


def test_walk_forward_backtest_uses_future_target_and_summarizes():
    history = make_history(rows=180, daily_return=0.002)
    observations = walk_forward_backtest(
        "TEST",
        history,
        TrendBaselineForecaster(),
        horizon=10,
        minimum_history=80,
        step=20,
        max_points=3,
        transaction_cost_bps=10,
    )

    assert len(observations) == 3
    assert (observations["target_date"] > observations["as_of"]).all()
    assert (observations["actual_return"] > 0).all()

    summary = summarize_backtest(observations)
    assert len(summary) == 1
    assert summary.loc[0, "observations"] == 3
    assert summary.loc[0, "directional_accuracy"] == 1.0
