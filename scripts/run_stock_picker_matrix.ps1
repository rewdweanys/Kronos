[CmdletBinding()]
param(
    [string[]]$Tickers = @("MSFT", "NVDA", "GOOGL"),
    [int[]]$Horizons = @(5, 10, 20),
    [int[]]$EnsembleSeeds = @(1, 7, 42, 99, 123),
    [ValidateSet("kronos-mini", "kronos-small", "kronos-base")]
    [string]$Model = "kronos-mini",
    [string]$Device = "cpu",
    [int]$MaxPoints = 20,
    [int]$Step = 20,
    [int]$SampleCount = 5,
    [double]$MinimumDirectionAgreement = 0.80,
    [double]$SignalThreshold = 0.03,
    [double]$TransactionCostBps = 10.0,
    [string]$OutputDir = "outputs\research_matrix",
    [switch]$SkipBaseline,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if ($MaxPoints -le 0) { throw "MaxPoints must be positive." }
if ($Step -le 0) { throw "Step must be positive." }
if ($SampleCount -le 0) { throw "SampleCount must be positive." }
if ($EnsembleSeeds.Count -lt 2) { throw "At least two EnsembleSeeds are required." }
if ($MinimumDirectionAgreement -lt 0 -or $MinimumDirectionAgreement -gt 1) {
    throw "MinimumDirectionAgreement must be between 0 and 1."
}

$null = Get-Command python -ErrorAction Stop
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Invoke-ResearchCase {
    param(
        [Parameter(Mandatory)] [string]$Ticker,
        [Parameter(Mandatory)] [int]$Horizon,
        [Parameter(Mandatory)] [ValidateSet("baseline", "ensemble")] [string]$Kind
    )

    $tickerName = $Ticker.Trim().ToUpperInvariant()
    $modelName = if ($Kind -eq "baseline") { "baseline" } else { $Model }
    $prefix = Join-Path $OutputDir ("{0}_{1}_{2}d" -f $tickerName, $Kind, $Horizon)
    $observationsPath = "${prefix}_observations.csv"
    $summaryPath = "${prefix}_summary.csv"

    if ((-not $Force) -and (Test-Path $summaryPath)) {
        Write-Host "Skipping completed case: $tickerName $Kind ${Horizon}d"
        return
    }

    Write-Host "`n==> Running $tickerName $Kind ${Horizon}d"
    $arguments = @(
        "-m", "stock_picker",
        "--mode", "backtest",
        "--model", $modelName,
        "--tickers", $tickerName,
        "--limit", "1",
        "--horizons", "$Horizon",
        "--max-points", "$MaxPoints",
        "--step", "$Step",
        "--signal-threshold", "$SignalThreshold",
        "--transaction-cost-bps", "$TransactionCostBps",
        "--backtest-output", $observationsPath,
        "--summary-output", $summaryPath
    )

    if ($Kind -eq "ensemble") {
        $arguments += @(
            "--device", $Device,
            "--sample-count", "$SampleCount",
            "--minimum-direction-agreement", "$MinimumDirectionAgreement",
            "--ensemble-seeds"
        )
        $arguments += ($EnsembleSeeds | ForEach-Object { "$_" })
    }

    & python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Research case failed: $tickerName $Kind ${Horizon}d"
    }
}

foreach ($ticker in $Tickers) {
    foreach ($horizon in $Horizons) {
        if (-not $SkipBaseline) {
            Invoke-ResearchCase -Ticker $ticker -Horizon $horizon -Kind baseline
        }
        Invoke-ResearchCase -Ticker $ticker -Horizon $horizon -Kind ensemble
    }
}

$summaryFiles = Get-ChildItem -Path $OutputDir -Filter "*_summary.csv" -File |
    Where-Object { $_.Name -ne "combined_summary.csv" }

$combined = foreach ($file in $summaryFiles) {
    foreach ($row in (Import-Csv $file.FullName)) {
        $observations = [int]$row.observations
        $maeImprovement = [double]$row.mae_improvement
        $correlation = [double]$row.return_correlation
        $directionalAccuracy = [double]$row.directional_accuracy
        $averageNetReturn = [double]$row.average_net_return
        $signalWinRate = if ([string]::IsNullOrWhiteSpace($row.signal_win_rate)) {
            [double]::NaN
        } else {
            [double]$row.signal_win_rate
        }

        $screenPass = (
            $observations -ge 20 -and
            $maeImprovement -gt 0 -and
            $correlation -gt 0 -and
            $directionalAccuracy -gt 0.50 -and
            (-not [double]::IsNaN($signalWinRate)) -and
            $signalWinRate -gt 0.50 -and
            $averageNetReturn -gt 0
        )

        $row | Add-Member -NotePropertyName screen_pass -NotePropertyValue $screenPass -PassThru
    }
}

$combinedPath = Join-Path $OutputDir "combined_summary.csv"
$combined = $combined | Sort-Object model, ticker, {[int]$_.horizon}
$combined | Export-Csv $combinedPath -NoTypeInformation

Write-Host "`n==> Combined research screen"
$combined |
    Select-Object ticker, model, horizon, observations, signals, directional_accuracy,
        model_mae, no_change_mae, mae_improvement, return_correlation,
        signal_win_rate, average_net_return, cumulative_net_return, screen_pass |
    Format-Table -AutoSize

Write-Host "`nSaved combined summary: $((Resolve-Path $combinedPath).Path)"
Write-Host "screen_pass is only an initial research filter, not evidence that a strategy is ready to trade."
