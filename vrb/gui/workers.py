"""Background QThread worker + thin GUI re-exports of result builders.

Heavy work (backtests, ML training, Kronos inference) runs off the UI thread;
widgets receive plain-dict payloads (defined Qt-free in backtest.payload) via
Qt signals.
"""

from __future__ import annotations

import traceback

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Re-export the Qt-free payload builders so GUI code can import them from here.
from ..backtest.payload import (day_payload, legs_text, run_backtest_days,
                                trade_payload)
from ..ml.kronos_forecaster import resample_1min
from ..data import ib

__all__ = ["FnWorker", "day_payload", "legs_text", "trade_payload",
           "run_backtest_days", "candles_for_day"]


class FnWorker(QThread):
    """Run fn(progress_cb) in a thread; emit progress strings and the result."""

    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn(lambda msg: self.progress.emit(msg))
            self.done.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


def candles_for_day(date: str, symbol: str = "SPX") -> dict:
    """1-min OHLCV candles for the chart, from the 1-sec IB store."""
    bars = ib.load_day(symbol, date)
    df = resample_1min(bars)
    return {
        "ts": df["timestamps"].to_numpy().astype("datetime64[s]"),
        "open": df["open"].to_numpy(float),
        "high": df["high"].to_numpy(float),
        "low": df["low"].to_numpy(float),
        "close": df["close"].to_numpy(float),
    }


def supertrend_for_day(date: str, symbol: str, period: int, multiplier: float,
                       start: str, end: str, rth_only: bool = True) -> dict:
    """SuperTrend chart data for one day, optionally sliced to the RTH window.

    Returns ts/open/high/low/close/st/direction (display slice) plus the raw
    reversal events (within [start, end]) for marker placement.
    """
    from ..indicators.supertrend import day_signals
    sig = day_signals(date, symbol, period, multiplier, start, end)
    ts = sig["ts"]
    iso = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    if rth_only:
        rth_open = np.datetime64(f"{iso}T08:30:00", "s")
        rth_close = np.datetime64(f"{iso}T15:00:00", "s")
        m = (ts >= rth_open) & (ts <= rth_close)
    else:
        m = np.ones(len(ts), bool)
    return {
        "ts": ts[m], "open": sig["open"][m], "high": sig["high"][m],
        "low": sig["low"][m], "close": sig["close"][m],
        "st": sig["st"][m], "direction": sig["direction"][m],
        "events": sig["events"],
    }
