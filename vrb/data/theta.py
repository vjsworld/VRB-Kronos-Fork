"""ThetaData option-quote loader: 8M-row daily parquet -> dense numpy grid.

Each trade-date file is a dense 5-sec NBBO snapshot grid covering the full ET
day (converted to naive CT). We pivot it to arrays shaped (T, K, 2) where the
last axis is right: 0=CALL, 1=PUT. The pivot is expensive (~seconds), so the
result is NPZ-cached; warm loads are near-instant.

Only the trade date's own expiration (true 0DTE) is kept; files downloaded
with Max DTE > 0 may also contain next-day expirations, which we drop.
By default the grid is trimmed to RTH (08:30-15:00 CT) — the 0DTE session we
backtest — which also keeps the cache ~4x smaller than the raw 24h window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import THETA_ROOT
from .cache import load_or_build

CALL, PUT = 0, 1
RIGHT_NAMES = ("CALL", "PUT")

# Cached grid window, naive CT. End is inclusive of the 15:00:00 expiry print.
CACHE_WINDOW = ("08:30:00", "15:00:00")


def day_path(root: str, date: str) -> Path:
    return THETA_ROOT / root / "quotes" / f"{date}.parquet"


def load_chain_day(root: str, date: str) -> dict[str, np.ndarray]:
    """Load one trade date's 0DTE chain as a dense grid.

    Returns dict:
      ts        (T,)    datetime64[s], 5-sec grid, RTH window
      strikes   (K,)    float64, ascending
      bid, ask  (T,K,2) float32  (NaN = no snapshot row; 0.0 = real empty book)
      bid_size, ask_size (T,K,2) int32
    """
    source = day_path(root, date)
    if not source.exists():
        raise FileNotFoundError(source)

    def build() -> dict[str, np.ndarray]:
        iso = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        df = pd.read_parquet(
            source,
            columns=["expiration", "strike", "right_type", "timestamp",
                     "bid", "ask", "bid_size", "ask_size"],
        )
        # expiration stored as YYYY-MM-DD (current downloader) or YYYYMMDD (older)
        exp = df["expiration"].str.replace("-", "", regex=False)
        df = df[exp == date]
        if df.empty:
            raise ValueError(f"{source} has no {iso} (0DTE) expiration rows")

        lo, hi = f"{iso}T{CACHE_WINDOW[0]}", f"{iso}T{CACHE_WINDOW[1]}"
        df = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]

        ts = np.unique(df["timestamp"].to_numpy().astype("datetime64[s]"))
        strikes = np.unique(df["strike"].to_numpy(np.float64))
        t_idx = np.searchsorted(ts, df["timestamp"].to_numpy().astype("datetime64[s]"))
        k_idx = np.searchsorted(strikes, df["strike"].to_numpy(np.float64))
        r_idx = np.where(df["right_type"].to_numpy() == "CALL", CALL, PUT)

        shape = (len(ts), len(strikes), 2)
        out = {
            "ts": ts,
            "strikes": strikes,
            "bid": np.full(shape, np.nan, np.float32),
            "ask": np.full(shape, np.nan, np.float32),
            "bid_size": np.zeros(shape, np.int32),
            "ask_size": np.zeros(shape, np.int32),
        }
        for col in ("bid", "ask"):
            out[col][t_idx, k_idx, r_idx] = df[col].to_numpy(np.float32)
        for col in ("bid_size", "ask_size"):
            out[col][t_idx, k_idx, r_idx] = df[col].to_numpy(np.int32)
        return out

    return load_or_build(f"{root}/chain/{date}.npz", source, build)
