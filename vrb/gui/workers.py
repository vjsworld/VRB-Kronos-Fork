"""Background QThread worker + thin GUI re-exports of result builders.

Heavy work (backtests, ML training, Kronos inference) runs off the UI thread;
widgets receive plain-dict payloads (defined Qt-free in backtest.payload) via
Qt signals.
"""

from __future__ import annotations

import traceback

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
