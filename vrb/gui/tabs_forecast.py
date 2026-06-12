"""Kronos Forecast tab: sampled foundation-model forecast paths on a chart.

Pick a day and an anchor time; Kronos (GPU) draws N independently sampled
forward paths from the ES 1-min context, overlaid on the history candles,
with the actual future bars (when the anchor is historical) for comparison.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
                             QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..data.calendar import ib_dates
from . import theme
from .charts import SignalChart, to_epoch
from .workers import FnWorker


class ForecastTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.worker: FnWorker | None = None

        root = QVBoxLayout(self)
        bar = QGroupBox("Kronos forecast")
        h = QHBoxLayout(bar)

        h.addWidget(QLabel("Day"))
        self.day_box = QComboBox()
        for d in sorted(ib_dates("ES"))[-120:]:
            self.day_box.addItem(f"{d[:4]}-{d[4:6]}-{d[6:]}", d)
        self.day_box.setCurrentIndex(self.day_box.count() - 1)
        h.addWidget(self.day_box)

        h.addWidget(QLabel("Anchor (hr CT)"))
        self.time_box = QDoubleSpinBox()
        self.time_box.setRange(2.0, 14.5); self.time_box.setValue(10.0); self.time_box.setSingleStep(0.25)
        h.addWidget(self.time_box)

        h.addWidget(QLabel("Horizon (min)"))
        self.horizon_box = QSpinBox(); self.horizon_box.setRange(5, 120); self.horizon_box.setValue(30)
        h.addWidget(self.horizon_box)

        h.addWidget(QLabel("Paths"))
        self.paths_box = QSpinBox(); self.paths_box.setRange(1, 50); self.paths_box.setValue(12)
        h.addWidget(self.paths_box)

        h.addWidget(QLabel("Context"))
        # Only futures carry the overnight session needed for the 400-bar
        # lookback; cash indices (~390 RTH bars) can't satisfy it before ~15:00.
        self.symbol_box = QComboBox(); self.symbol_box.addItems(["ES", "NQ"])
        h.addWidget(self.symbol_box)

        self.run_btn = QPushButton("Forecast"); self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.run_forecast)
        h.addWidget(self.run_btn)
        self.status = QLabel(""); self.status.setObjectName("status")
        h.addWidget(self.status, stretch=1)
        root.addWidget(bar)

        self.chart = SignalChart()
        self.chart.equity_plot.setLabel("left", "Paths vs actual")
        root.addWidget(self.chart, stretch=1)

    def run_forecast(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        date = self.day_box.currentData()
        hr = self.time_box.value()
        anchor = np.datetime64(
            f"{date[:4]}-{date[4:6]}-{date[6:]}T{int(hr):02d}:{int(round((hr % 1) * 60)):02d}:00", "s")
        horizon = int(self.horizon_box.value())
        n_paths = int(self.paths_box.value())
        symbol = self.symbol_box.currentText()
        self.run_btn.setEnabled(False)
        self.status.setText("Loading Kronos..." if self.state.kronos is None
                            else "Sampling forecast paths...")

        def job(progress_cb):
            kf = self.state.get_kronos(pred_len=horizon)
            progress_cb(f"sampling {n_paths} paths x {horizon} min on {symbol}...")
            return kf.forecast_paths(date, anchor, symbol, n_paths)

        self.worker = FnWorker(job)
        self.worker.progress.connect(self.status.setText)
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _failed(self, tb: str) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText(tb.splitlines()[-1])
        print(tb)

    def _finished(self, fc: dict) -> None:
        self.run_btn.setEnabled(True)
        paths = fc["paths"]
        last = fc["last_close"]
        mean_path = paths.mean(axis=0)
        exp_ret = np.log(mean_path[-1] / last)
        self.status.setText(
            f"E[ret] {exp_ret * 100:+.3f}%  |  path spread "
            f"{(paths[:, -1].std() / last) * 100:.3f}%  |  anchor close {last:,.2f}")

        self.chart.set_candles(fc["hist_ts"], fc["hist_o"], fc["hist_h"],
                               fc["hist_l"], fc["hist_c"])
        t_pred = to_epoch(fc["pred_ts"])
        t0 = to_epoch(fc["hist_ts"][-1:])[0]
        for p in paths:
            self.chart.price_plot.plot(
                np.concatenate([[t0], t_pred]), np.concatenate([[last], p]),
                pen=pg.mkPen(theme.FORECAST, width=1, style=Qt.PenStyle.SolidLine))
        self.chart.price_plot.plot(
            np.concatenate([[t0], t_pred]), np.concatenate([[last], mean_path]),
            pen=pg.mkPen(theme.ACCENT, width=3))

        # actual future closes, if the anchor is historical
        mask = (fc["bar_ts"] > fc["hist_ts"][-1]) & (fc["bar_ts"] <= fc["pred_ts"][-1])
        if mask.any():
            self.chart.price_plot.plot(
                np.concatenate([[t0], to_epoch(fc["bar_ts"][mask])]),
                np.concatenate([[last], fc["bar_close"][mask]]),
                pen=pg.mkPen(theme.EXIT, width=2, style=Qt.PenStyle.DashLine))

        # spread fan in the lower plot: per-minute path stdev
        self.chart.equity_plot.clear()
        self.chart.equity_plot.plot(t_pred, paths.std(axis=0),
                                    pen=pg.mkPen(theme.FORECAST, width=2))
        self.chart.price_plot.autoRange()
