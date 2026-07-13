from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import download_daily, read_tickers
from .forecasts import KronosForecaster, TrendBaselineForecaster
from .scoring import aggregate_forecasts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research-first U.S. stock ranking prototype")
    parser.add_argument("--tickers-file", default="config/us_large_cap.txt")
    parser.add_argument("--model", choices=["baseline", "kronos-mini", "kronos-small", "kronos-base"], default="baseline")
    parser.add_argument("--device", default=None, help="cpu, cuda:0, or omit for auto-detection")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="outputs/latest_rankings.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tickers = read_tickers(args.tickers_file)[: args.limit]
    histories = download_daily(tickers, period=args.period)

    if args.model == "baseline":
        forecaster = TrendBaselineForecaster()
    else:
        forecaster = KronosForecaster(args.model, device=args.device)

    recommendations = []
    errors = []
    for ticker in tickers:
        history = histories.get(ticker)
        if history is None or history.empty:
            errors.append((ticker, "no data"))
            continue
        try:
            forecasts = [forecaster.forecast(ticker, history, horizon) for horizon in args.horizons]
            recommendations.append(aggregate_forecasts(forecasts))
        except Exception as exc:  # Keep a universe scan going when one symbol fails.
            errors.append((ticker, str(exc)))

    frame = pd.DataFrame([item.__dict__ for item in recommendations])
    if not frame.empty:
        frame = frame.sort_values(["score", "confidence"], ascending=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    if frame.empty:
        print("No recommendations were produced.")
    else:
        display = frame.copy()
        for column in ["score", "expected_return", "confidence"]:
            display[column] = display[column].map(lambda value: f"{value:.2%}")
        print(display.to_string(index=False))
    if errors:
        print(f"\nSkipped {len(errors)} ticker(s). First errors: {errors[:5]}")
    print(f"\nSaved: {output.resolve()}")
    return 0 if not frame.empty else 1
