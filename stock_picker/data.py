from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, date-sorted OHLCV frame with lower-case columns."""
    if frame.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    result = frame.copy()
    result.columns = [str(column).strip().lower().replace(" ", "_") for column in result.columns]
    if "adj_close" in result.columns and "close" not in result.columns:
        result["close"] = result["adj_close"]

    missing = [column for column in REQUIRED_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {missing}")

    result = result.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    result.index = pd.to_datetime(result.index, utc=False).tz_localize(None)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result = result.dropna(subset=("open", "high", "low", "close"))
    result = result[result["close"] > 0]
    result["volume"] = result["volume"].fillna(0.0).clip(lower=0.0)
    return result


def download_daily(
    tickers: Iterable[str],
    *,
    period: str = "5y",
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download free prototype data through yfinance.

    yfinance is suitable for research and paper-trading prototypes, not a guaranteed
    production feed. Callers receive one normalized frame per successfully loaded ticker.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run the Windows setup script or install "
            "requirements-stock-picker.txt."
        ) from exc

    symbols = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    if not symbols:
        raise ValueError("At least one ticker is required.")

    kwargs: dict[str, object] = {
        "tickers": symbols,
        "auto_adjust": False,
        "actions": False,
        "group_by": "ticker",
        "threads": True,
        "progress": False,
    }
    if start or end:
        kwargs.update(start=start, end=end)
    else:
        kwargs["period"] = period

    raw = yf.download(**kwargs)
    output: dict[str, pd.DataFrame] = {}

    if len(symbols) == 1:
        if not raw.empty:
            output[symbols[0]] = normalize_ohlcv(raw)
        return output

    if not isinstance(raw.columns, pd.MultiIndex):
        return output

    first_level = {str(value).upper() for value in raw.columns.get_level_values(0)}
    for ticker in symbols:
        try:
            if ticker in first_level:
                frame = raw[ticker]
            else:
                frame = raw.xs(ticker, axis=1, level=1)
            normalized = normalize_ohlcv(frame)
            if not normalized.empty:
                output[ticker] = normalized
        except (KeyError, ValueError):
            continue
    return output


def read_tickers(path: str | Path) -> list[str]:
    values: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        clean = line.split("#", maxsplit=1)[0].strip().upper()
        if clean:
            values.append(clean)
    return list(dict.fromkeys(values))


def load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_column = next(
        (column for column in frame.columns if column.lower() in {"date", "datetime", "timestamp", "timestamps"}),
        None,
    )
    if date_column is None:
        raise ValueError("CSV must include a date, datetime, timestamp, or timestamps column.")
    frame = frame.set_index(pd.to_datetime(frame.pop(date_column), utc=False))
    return normalize_ohlcv(frame)
