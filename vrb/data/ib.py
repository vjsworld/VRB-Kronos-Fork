"""IB underlying bar loaders (1-sec ES/NQ/SPX/NDX, 1-min VIX) with NPZ cache.

Timestamps in the parquet are naive-CT bar-CLOSE instants (the downloader
already shifted bar-start -> bar-close), so a bar's close IS the price at its
timestamp and joins to ThetaData quote snapshots are exact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import IB_LAYOUT, IB_ROOT
from .cache import load_or_build


def day_path(symbol: str, date: str) -> Path:
    sec_type, interval, _ = IB_LAYOUT[symbol]
    return IB_ROOT / symbol / sec_type / interval / f"{date}.parquet"


def load_day(symbol: str, date: str) -> dict[str, np.ndarray]:
    """Load one day of bars -> {'ts': datetime64[s], 'open', 'high', 'low',
    'close': float64, 'volume': int64}, sorted by ts."""
    source = day_path(symbol, date)
    if not source.exists():
        raise FileNotFoundError(source)

    def build() -> dict[str, np.ndarray]:
        df = pd.read_parquet(source)
        ts = df["timestamp"].to_numpy().astype("datetime64[s]")
        order = np.argsort(ts, kind="stable")
        return {
            "ts": ts[order],
            "open": df["open"].to_numpy(np.float64)[order],
            "high": df["high"].to_numpy(np.float64)[order],
            "low": df["low"].to_numpy(np.float64)[order],
            "close": df["close"].to_numpy(np.float64)[order],
            "volume": df["volume"].to_numpy(np.int64)[order],
        }

    return load_or_build(f"IB/{symbol}/{date}.npz", source, build)


def load_range(symbol: str, dates: list[str]) -> dict[str, np.ndarray]:
    """Concatenate multiple days (skipping missing files) into one array set."""
    parts = []
    for d in dates:
        try:
            parts.append(load_day(symbol, d))
        except FileNotFoundError:
            continue
    if not parts:
        raise FileNotFoundError(f"no {symbol} data for {dates[0]}..{dates[-1]}")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def asof(bar_ts: np.ndarray, values: np.ndarray, grid_ts: np.ndarray) -> np.ndarray:
    """Latest bar value at-or-before each grid timestamp (NaN before first bar)."""
    idx = np.searchsorted(bar_ts, grid_ts, side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], np.nan)
    return out
