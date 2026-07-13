# Kronos Stock Picker Prototype

This branch preserves the upstream Kronos model and adds a research-first U.S. stock ranking layer for approximately 5, 10, and 20 trading-day horizons.

## What this version does

- Downloads free daily OHLCV data with `yfinance`.
- Runs a transparent trend/volatility baseline.
- Adapts Kronos-mini, Kronos-small, or Kronos-base behind the same forecasting interface.
- Produces BUY, SHORT, or NEUTRAL research labels, expected return, confidence, and forecast ranges.
- Saves both aggregate rankings and separate 5-, 10-, and 20-day forecast rows.
- Uses repeatable per-symbol, per-date random seeds for Kronos sampling.
- Runs walk-forward historical tests with directional accuracy, range coverage, forecast error, signal win rate, and transaction-cost-adjusted returns.
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

```powershell
python -m stock_picker --model baseline --limit 10
python -m stock_picker --model kronos-mini --device cpu --limit 3 --seed 42 --sample-count 5
```

The scan creates:

- `outputs/latest_rankings.csv` — one aggregate row per ticker.
- `outputs/latest_forecasts.csv` — separate rows for each ticker and horizon.

Running the same model, data, seed, and sample count should now reproduce the same result. Change `--seed` to deliberately test sensitivity to a different sampling stream.

## Walk-forward backtest

Start with the baseline because it is fast:

```powershell
python -m stock_picker --mode backtest --model baseline --limit 3 --max-points 20
```

Then test Kronos on one ticker and a small number of historical decision dates because CPU inference is much slower:

```powershell
python -m stock_picker --mode backtest --model kronos-mini --device cpu --limit 1 --horizons 5 10 20 --max-points 3 --seed 42 --sample-count 5
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
- `signal_win_rate`: percentage of threshold-triggered simulated trades profitable after costs.
- `cumulative_net_return`: compounded return across those sampled signals; overlapping tests and small samples can make this misleading, so it is not sufficient evidence by itself.

The default backtest signal threshold is 3%, and the default cost assumption is 10 basis points on both entry and exit. These can be changed with `--signal-threshold` and `--transaction-cost-bps`.

## GPU note

After the NVIDIA driver is repaired and `nvidia-smi` works, reinstall a CUDA-enabled PyTorch build in the virtual environment before using:

```powershell
python -m stock_picker --model kronos-small --device cuda:0 --limit 10
```

The current CPU PyTorch package cannot use the GPU merely by fixing the Windows driver.

## Research rules

Kronos is not accepted as useful merely because its forecast chart looks convincing. It must beat:

1. A no-change forecast.
2. The included trend/volatility baseline.
3. Transaction-cost-aware walk-forward testing.
4. Paper trading on data unavailable during development.

## Next milestones

1. Add historical index membership to reduce survivorship bias.
2. Add multi-seed stability reports and calibrated forecast probabilities.
3. Add market and sector regime features.
4. Add earnings calendars, SEC filings, fundamentals, and news as separately testable feature groups.
5. Add paper-broker execution only after out-of-sample evidence.
6. Add ETFs, REITs, and crypto through provider and asset-class adapters.
