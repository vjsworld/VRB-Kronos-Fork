"""SuperTrend indicator and 0DTE reversal signals from a 1-min futures chart.

SuperTrend is an ATR-banded trend follower. Its direction flips when price
closes through the opposite band; each flip is a reversal signal:

  flip to +1 (uptrend)  -> bullish -> long signal  -> buy a call
  flip to -1 (downtrend) -> bearish -> short signal -> buy a put

`day_signals` loads the futures 1-min bars for a trade date, computes
SuperTrend over the full session (so the ATR is well warmed by RTH), and
returns the reversal events that fall inside a [start, end] CT window.
"""

from __future__ import annotations

import numpy as np

from ..data import ib
from ..options.chain import CALL, PUT


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int) -> np.ndarray:
    """Average True Range with Wilder's smoothing (RMA). NaN until warmed."""
    n = len(close)
    atr = np.full(n, np.nan)
    if n == 0:
        return atr
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close),
                                           np.abs(low - prev_close)))
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 10, multiplier: float = 3.0):
    """Return (st_line, direction, atr).

    direction is +1 (up), -1 (down), or 0 (before warmup). st_line is the
    active band (the line you plot under/over price).
    """
    n = len(close)
    st = np.full(n, np.nan)
    direction = np.zeros(n, np.int8)
    atr = wilder_atr(high, low, close, period)
    if n == 0 or n < period:
        return st, direction, atr

    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * atr      # mutated (carried forward) in the loop
    lower = hl2 - multiplier * atr
    w = period - 1                      # first finite-ATR bar
    direction[w] = 1
    st[w] = lower[w]
    for i in range(w + 1, n):
        if close[i] > upper[i - 1]:
            direction[i] = 1
        elif close[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            if direction[i] == 1 and lower[i] < lower[i - 1]:
                lower[i] = lower[i - 1]
            if direction[i] == -1 and upper[i] > upper[i - 1]:
                upper[i] = upper[i - 1]
        st[i] = lower[i] if direction[i] == 1 else upper[i]
    return st, direction, atr


def day_signals(date: str, symbol: str = "ES", period: int = 10,
                multiplier: float = 3.0, start: str = "14:00:00",
                end: str = "15:00:00") -> dict:
    """SuperTrend over one trade date's 1-min futures bars + windowed reversals.

    Returns dict with full-session arrays (ts, open, high, low, close, st,
    direction) for charting, plus `events`: list of (timestamp, right) where
    right is CALL (bullish flip) or PUT (bearish flip), filtered to [start,end].
    """
    bars = ib.resample_1min(ib.load_day(symbol, date))
    ts = bars["timestamps"].to_numpy().astype("datetime64[s]")
    o = bars["open"].to_numpy(np.float64)
    h = bars["high"].to_numpy(np.float64)
    l = bars["low"].to_numpy(np.float64)
    c = bars["close"].to_numpy(np.float64)
    st, direction, atr = supertrend(h, l, c, period, multiplier)

    rev = np.zeros(len(c), bool)
    rev[1:] = ((direction[1:] != direction[:-1])
               & (direction[1:] != 0) & (direction[:-1] != 0))

    iso = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    lo = np.datetime64(f"{iso}T{start}", "s")
    hi = np.datetime64(f"{iso}T{end}", "s")
    events = []
    for i in np.flatnonzero(rev):
        if lo <= ts[i] <= hi:
            events.append((ts[i], CALL if direction[i] == 1 else PUT))

    return {
        "ts": ts, "open": o, "high": h, "low": l, "close": c,
        "st": st, "direction": direction, "atr": atr, "events": events,
    }
