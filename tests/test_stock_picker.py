import numpy as np
import pandas as pd

from stock_picker.backtest import summarize_backtest, walk_forward_backtest
from stock_picker.data import normalize_ohlcv
from stock_picker.forecasts import (
    Forecast,
    TrendBaselineForecaster,
    _stable_forecast_seed,
    aggregate_ensemble_forecasts,
)
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


def member(expected_return: float, confidence: float = 0.6) -> Forecast:
    current = 100.0
    return Forecast(
        ticker="TEST",
        as_of=pd.Timestamp("2026-07-10"),
        horizon=10,
        model="kronos-mini",
        current_price=current,
        expected_close=current * (1 + expected_return),
        range_low=90 + expected_return * 10,
        range_high=110 + expected_return * 10,
        expected_return=expected_return,
        confidence=confidence,
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


def test_ensemble_uses_median_and_records_disagreement():
    ensemble = aggregate_ensemble_forecasts(
        [member(0.08), member(0.06), member(0.04), member(0.02), member(-0.12)],
        model_name="kronos-mini-ensemble",
    )
    assert np.isclose(ensemble.expected_return, 0.04)
    assert np.isclose(ensemble.direction_agreement, 0.8)
    assert ensemble.return_dispersion > 0
    assert ensemble.ensemble_size == 5
    assert ensemble.model == "kronos-mini-ensemble"


def test_backtest_rejects_low_agreement_signal():
    class LowAgreementForecaster:
        name = "low-agreement"

        def forecast(self, ticker, history, horizon):
            current = float(history["close"].iloc[-1])
            return Forecast(
                ticker=ticker,
                as_of=pd.Timestamp(history.index[-1]),
                horizon=horizon,
                model=self.name,
                current_price=current,
                expected_close=current * 1.10,
                range_low=current * 0.95,
                range_high=current * 1.15,
                expected_return=0.10,
                confidence=0.80,
                direction_agreement=0.60,
                return_dispersion=0.02,
                ensemble_size=5,
            )

    observations = walk_forward_backtest(
        "TEST",
        make_history(rows=120),
        LowAgreementForecaster(),
        horizon=5,
        minimum_history=80,
        step=20,
        max_points=2,
        minimum_direction_agreement=0.80,
    )
    assert (observations["signal"] == 0).all()
