# Kronos Stock Picker Prototype

This branch preserves the upstream Kronos model and adds a research-first U.S. stock ranking layer for approximately 5, 10, and 20 trading-day horizons.

## What this version does

- Downloads free daily OHLCV data with `yfinance`.
- Runs transparent linear-trend, multi-window momentum, and short-term mean-reversion benchmarks.
- Adapts Kronos-mini, Kronos-small, or Kronos-base behind the same forecasting interface.
- Produces BUY, SHORT, or NEUTRAL research labels, expected return, confidence, and forecast ranges.
- Saves both aggregate rankings and separate 5-, 10-, and 20-day forecast rows.
- Uses repeatable per-symbol, per-date random seeds for Kronos sampling.
- Supports robust multi-seed ensembles using the median forecast, seed-direction agreement, and return dispersion.
- Runs walk-forward historical tests with directional accuracy, range coverage, forecast error, signal win rate, and transaction-cost-adjusted returns.
- Rejects simulated ensemble trades when seed-direction agreement is below the configured threshold.
- Includes a resumable PowerShell research matrix for comparing all benchmarks and the Kronos ensemble across multiple symbols and horizons.
- Does **not** place orders or connect to a broker.

Free Yahoo data is for prototyping and paper research, not a guaranteed production feed. Recommendations are experimental model output, not financial advice.

## Windows installation

Open PowerShell in the folder where you want the project, then clone your fork:

```powershell
git clone https://github.com/rewdweanys/Kronos.git
cd Kronos
git switch stock-picker-v1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\setup_stock_picker_windows.ps1 -SkipClone
```

The script installs Python 3.11 through `winget` when necessary, creates `.venv`, installs the upstream Kronos requirements plus prototype dependencies, verifies PyTorch, and runs the unit tests. It leaves Python 3.14 installed.

Activate later with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Current scan

Fast transparent models:

```powershell
python -m stock_picker --model baseline --limit 10
python -m stock_picker --model momentum --limit 10
python -m stock_picker --model mean-reversion --limit 10
```

Single-seed Kronos remains available for controlled experiments:

```powershell
python -m stock_picker --model kronos-mini --device cpu --limit 3 --seed 42 --sample-count 5
```

For an actual Kronos research scan, prefer the multi-seed ensemble:

```powershell
python -m stock_picker `
  --model kronos-mini `
  --device cpu `
  --tickers AAPL MSFT NVDA `
  --ensemble-seeds 1 7 42 99 123 `
  --sample-count 5
```

`--tickers` overrides the configured ticker file. The scan creates:

- `outputs/latest_rankings.csv` — one aggregate row per ticker.
- `outputs/latest_forecasts.csv` — separate rows for each ticker and horizon, including `direction_agreement`, `return_dispersion`, and `ensemble_size`.

Running the same model, data, seeds, and sample count should reproduce the same result. Change the ensemble seed list only to deliberately test a different sampling set.

## Walk-forward backtest

Start with the fast models because they do not require neural-model inference:

```powershell
python -m stock_picker --mode backtest --model baseline --limit 3 --max-points 20
python -m stock_picker --mode backtest --model momentum --limit 3 --max-points 20
python -m stock_picker --mode backtest --model mean-reversion --limit 3 --max-points 20
```

Then test a multi-seed Kronos ensemble on one ticker and one horizon. CPU inference is much slower because each historical decision runs every seed:

```powershell
python -m stock_picker `
  --mode backtest `
  --model kronos-mini `
  --device cpu `
  --tickers AAPL `
  --horizons 10 `
  --max-points 20 `
  --step 20 `
  --ensemble-seeds 1 7 42 99 123 `
  --sample-count 5 `
  --minimum-direction-agreement 0.80 `
  --backtest-output outputs\kronos_aapl_10_ensemble_observations.csv `
  --summary-output outputs\kronos_aapl_10_ensemble_summary.csv
```

Backtesting creates:

- `outputs/backtest_observations.csv` — every historical prediction and realized outcome.
- `outputs/backtest_summary.csv` — metrics grouped by ticker, model, and horizon.

Important metrics:

- `directional_accuracy`: whether predicted and realized returns had the same sign.
- `range_coverage`: whether the realized future close fell inside the forecast range.
- `model_mae`: average absolute error in predicted return.
- `no_change_mae`: error from simply predicting zero return.
- `mae_improvement`: positive means the model beat the no-change forecast on error.
- `average_direction_agreement`: average fraction of seeds agreeing with the ensemble direction.
- `average_return_dispersion`: median seed-to-seed return disagreement, averaged across observations.
- `signal_win_rate`: percentage of threshold-triggered simulated trades profitable after costs.
- `cumulative_net_return`: compounded return across those sampled signals; overlapping tests and small samples can make this misleading, so it is not sufficient evidence by itself.

The default backtest signal threshold is 3%, the default ensemble agreement threshold is 80%, and the default cost assumption is 10 basis points on both entry and exit. These can be changed with `--signal-threshold`, `--minimum-direction-agreement`, and `--transaction-cost-bps`.

## Resumable research matrix

Use the matrix runner to compare all three fast benchmarks and the Kronos ensemble across several stocks and all three horizons:

```powershell
.\scripts\run_stock_picker_matrix.ps1
```

The default matrix evaluates MSFT, NVDA, and GOOGL at 5, 10, and 20 trading days. Each ticker/model/horizon case is saved separately under `outputs\research_matrix`. Re-running the script skips completed cases unless `-Force` is supplied, so an interrupted CPU experiment can continue without starting over.

Customize the universe or test size with PowerShell parameters:

```powershell
.\scripts\run_stock_picker_matrix.ps1 `
  -Tickers MSFT,NVDA,GOOGL,AMZN,AVGO `
  -Horizons 5,10,20 `
  -MaxPoints 20
```

The script builds `outputs\research_matrix\combined_summary.csv` and adds `screen_pass`, an intentionally strict initial filter requiring at least 20 observations, positive improvement over no-change, positive return correlation, better-than-random direction and signal win rates, and positive average net returns. Passing that screen is not proof of a tradable strategy; it only identifies configurations worth deeper out-of-sample testing.

## GPU note

After the NVIDIA driver is repaired and `nvidia-smi` works, reinstall a CUDA-enabled PyTorch build in the virtual environment before using:

```powershell
python -m stock_picker --model kronos-small --device cuda:0 --limit 10
```

The current CPU PyTorch package cannot use the GPU merely by fixing the Windows driver.

## Research rules

Kronos is not accepted as useful merely because its forecast chart looks convincing. It must beat:

1. A no-change forecast.
2. The included transparent benchmarks.
3. Transaction-cost-aware walk-forward testing.
4. Paper trading on data unavailable during development.

## Next milestones

1. Complete the resumable multi-stock, multi-horizon research matrix.
2. Add historical index membership to reduce survivorship bias.
3. Calibrate ensemble confidence and forecast ranges against out-of-sample outcomes.
4. Add market and sector regime features.
5. Add earnings calendars, SEC filings, fundamentals, and news as separately testable feature groups.
6. Add paper-broker execution only after out-of-sample evidence.
7. Add ETFs, REITs, and crypto through provider and asset-class adapters.
