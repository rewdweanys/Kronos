# Kronos Stock Picker Prototype

This branch preserves the upstream Kronos model and adds a research-first U.S. stock ranking layer for approximately 5, 10, and 20 trading-day horizons.

## What this first version does

- Downloads free daily OHLCV data with `yfinance`.
- Runs a transparent trend/volatility baseline immediately.
- Adapts Kronos-mini, Kronos-small, or Kronos-base behind the same forecasting interface.
- Produces BUY, SHORT, or NEUTRAL research labels, expected return, confidence, and a forecast range.
- Saves rankings to CSV.
- Does **not** place orders or connect to a broker.

Free Yahoo data is for prototyping and paper research, not a guaranteed production feed. Recommendations are experimental model output, not financial advice.

## Windows installation

Open PowerShell in the folder where you want the project, then clone your fork:

```powershell
git clone https://github.com/rewdweanys/Kronos.git
cd Kronos
git switch stock-picker-v1
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_stock_picker_windows.ps1 -SkipClone
```

The script installs Python 3.11 through `winget` when necessary, creates `.venv`, installs the upstream Kronos requirements plus prototype dependencies, verifies PyTorch, and runs the unit tests. It leaves Python 3.14 installed.

Activate later with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## First CPU test

```powershell
python -m pytest tests\test_stock_picker.py
python -m stock_picker --model baseline --limit 10
```

This creates `outputs/latest_rankings.csv`.

## First Kronos test

The first run downloads model files from Hugging Face and can take longer on CPU:

```powershell
python -m stock_picker --model kronos-mini --device cpu --limit 3
```

After the NVIDIA driver is repaired and `nvidia-smi` works:

```powershell
python -m stock_picker --model kronos-small --device cuda:0 --limit 10
```

## Research rules

Kronos is not accepted as useful merely because its forecast chart looks convincing. It must beat:

1. No-change and buy-and-hold assumptions.
2. The included trend/volatility baseline.
3. Transaction-cost-aware walk-forward testing.
4. Paper trading on data unavailable during development.

## Next milestones

1. Add a date-by-date walk-forward backtester and performance report.
2. Add historical S&P 500 membership to avoid survivorship bias.
3. Calibrate forecast ranges and probabilities.
4. Add market/sector regime features.
5. Add earnings calendars, SEC filings, fundamentals, and news as separately testable feature groups.
6. Add paper-broker execution only after out-of-sample evidence.
7. Add ETFs, REITs, and crypto through provider and asset-class adapters.
