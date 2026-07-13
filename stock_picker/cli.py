from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import summarize_backtest, walk_forward_backtest
from .baselines import MeanReversionForecaster, MomentumForecaster
from .data import download_daily, read_tickers
from .forecasts import KronosEnsembleForecaster, KronosForecaster, TrendBaselineForecaster
from .scoring import aggregate_forecasts


MODEL_CHOICES = [
    "baseline",
    "momentum",
    "mean-reversion",
    "kronos-mini",
    "kronos-small",
    "kronos-base",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research-first U.S. stock ranking prototype")
    parser.add_argument("--mode", choices=["scan", "backtest"], default="scan")
    parser.add_argument("--tickers-file", default="config/us_large_cap.txt")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Explicit ticker symbols; when supplied, overrides --tickers-file",
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default="baseline")
    parser.add_argument("--device", default=None, help="cpu, cuda:0, or omit for auto-detection")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Repeatable seed for single-seed Kronos")
    parser.add_argument(
        "--ensemble-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run a robust Kronos ensemble, for example: --ensemble-seeds 1 7 42 99 123",
    )
    parser.add_argument("--sample-count", type=int, default=5, help="Forecast paths averaged inside each Kronos seed")
    parser.add_argument("--output", default="outputs/latest_rankings.csv")
    parser.add_argument("--forecasts-output", default="outputs/latest_forecasts.csv")
    parser.add_argument("--backtest-output", default="outputs/backtest_observations.csv")
    parser.add_argument("--summary-output", default="outputs/backtest_summary.csv")
    parser.add_argument("--minimum-history", type=int, default=252)
    parser.add_argument("--step", type=int, default=20, help="Trading days between backtest decisions")
    parser.add_argument("--max-points", type=int, default=5, help="Most recent backtest decisions per ticker/horizon")
    parser.add_argument("--signal-threshold", type=float, default=0.03)
    parser.add_argument("--minimum-confidence", type=float, default=0.0)
    parser.add_argument(
        "--minimum-direction-agreement",
        type=float,
        default=0.80,
        help="Minimum seed-direction agreement required to open a simulated trade",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    return parser


def _build_forecaster(args: argparse.Namespace):
    if args.model == "baseline":
        if args.ensemble_seeds:
            raise ValueError("--ensemble-seeds is only valid with a Kronos model.")
        return TrendBaselineForecaster()
    if args.model == "momentum":
        if args.ensemble_seeds:
            raise ValueError("--ensemble-seeds is only valid with a Kronos model.")
        return MomentumForecaster()
    if args.model == "mean-reversion":
        if args.ensemble_seeds:
            raise ValueError("--ensemble-seeds is only valid with a Kronos model.")
        return MeanReversionForecaster()
    if args.ensemble_seeds:
        return KronosEnsembleForecaster(
            args.model,
            device=args.device,
            sample_count=args.sample_count,
            seeds=args.ensemble_seeds,
        )
    return KronosForecaster(
        args.model,
        device=args.device,
        sample_count=args.sample_count,
        seed=args.seed,
    )


def _write_csv(frame: pd.DataFrame, path_value: str) -> Path:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _resolve_tickers(args: argparse.Namespace) -> list[str]:
    if args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.tickers:
        values = [str(ticker).strip().upper() for ticker in args.tickers if str(ticker).strip()]
        tickers = list(dict.fromkeys(values))
    else:
        tickers = read_tickers(args.tickers_file)
    if not tickers:
        raise ValueError("No ticker symbols were supplied.")
    return tickers[: args.limit]


def _run_scan(args: argparse.Namespace, tickers: list[str], histories: dict[str, pd.DataFrame]) -> int:
    forecaster = _build_forecaster(args)
    recommendations = []
    forecast_rows = []
    errors = []

    for ticker in tickers:
        history = histories.get(ticker)
        if history is None or history.empty:
            errors.append((ticker, "no data"))
            continue
        try:
            forecasts = [forecaster.forecast(ticker, history, horizon) for horizon in args.horizons]
            forecast_rows.extend(item.__dict__ for item in forecasts)
            recommendations.append(aggregate_forecasts(forecasts))
        except Exception as exc:  # Keep a universe scan going when one symbol fails.
            errors.append((ticker, str(exc)))

    frame = pd.DataFrame([item.__dict__ for item in recommendations])
    if not frame.empty:
        frame = frame.sort_values(["score", "confidence"], ascending=False)
    forecast_frame = pd.DataFrame(forecast_rows)
    if not forecast_frame.empty:
        forecast_frame = forecast_frame.sort_values(["ticker", "horizon"])

    output = _write_csv(frame, args.output)
    forecasts_output = _write_csv(forecast_frame, args.forecasts_output)

    if frame.empty:
        print("No recommendations were produced.")
    else:
        display = frame.copy()
        for column in ["score", "expected_return", "confidence"]:
            display[column] = display[column].map(lambda value: f"{value:.2%}")
        print(display.to_string(index=False))
    if errors:
        print(f"\nSkipped {len(errors)} ticker(s). First errors: {errors[:5]}")
    print(f"\nSaved rankings: {output.resolve()}")
    print(f"Saved per-horizon forecasts: {forecasts_output.resolve()}")
    return 0 if not frame.empty else 1


def _run_backtest(args: argparse.Namespace, tickers: list[str], histories: dict[str, pd.DataFrame]) -> int:
    forecaster = _build_forecaster(args)
    observation_frames = []
    errors = []

    for ticker in tickers:
        history = histories.get(ticker)
        if history is None or history.empty:
            errors.append((ticker, "no data"))
            continue
        for horizon in args.horizons:
            try:
                result = walk_forward_backtest(
                    ticker,
                    history,
                    forecaster,
                    horizon,
                    minimum_history=args.minimum_history,
                    step=args.step,
                    max_points=args.max_points,
                    signal_threshold=args.signal_threshold,
                    minimum_confidence=args.minimum_confidence,
                    minimum_direction_agreement=args.minimum_direction_agreement,
                    transaction_cost_bps=args.transaction_cost_bps,
                )
                if not result.empty:
                    observation_frames.append(result)
            except Exception as exc:
                errors.append((f"{ticker}/{horizon}", str(exc)))

    observations = pd.concat(observation_frames, ignore_index=True) if observation_frames else pd.DataFrame()
    summary = summarize_backtest(observations)
    observations_output = _write_csv(observations, args.backtest_output)
    summary_output = _write_csv(summary, args.summary_output)

    if summary.empty:
        print("No backtest observations were produced.")
    else:
        display = summary.copy()
        percent_columns = [
            "directional_accuracy",
            "range_coverage",
            "model_mae",
            "no_change_mae",
            "mae_improvement",
            "return_correlation",
            "average_confidence",
            "average_direction_agreement",
            "average_return_dispersion",
            "signal_win_rate",
            "average_net_return",
            "cumulative_net_return",
        ]
        for column in percent_columns:
            display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{value:.2%}")
        print(display.to_string(index=False))
    if errors:
        print(f"\nSkipped {len(errors)} backtest segment(s). First errors: {errors[:5]}")
    print(f"\nSaved observations: {observations_output.resolve()}")
    print(f"Saved summary: {summary_output.resolve()}")
    return 0 if not summary.empty else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tickers = _resolve_tickers(args)
    histories = download_daily(tickers, period=args.period)
    if args.mode == "backtest":
        return _run_backtest(args, tickers, histories)
    return _run_scan(args, tickers, histories)
