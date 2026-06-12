"""ML Lab tab: build features, walk-forward train, backtest the signal OOS.

Trains a gradient-boosting forecaster on point-in-time snapshot features,
reports walk-forward IC / hit-rate metrics and per-feature information
coefficients, then runs the resulting signal through the real-NBBO option
backtester. The resulting trades flow to the shared state so the Performance
Report and Backtest charts can display them.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QProgressBar, QPushButton, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from ..data.calendar import common_dates
from . import theme
from .charts import EquityReportChart
from .workers import FnWorker


class MLLabTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.worker: FnWorker | None = None

        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        left = QWidget(); ll = QVBoxLayout(left)
        cfg = QGroupBox("Training configuration"); form = QVBoxLayout(cfg)

        def row(label, widget):
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(label)); h.addWidget(widget, stretch=1)
            form.addWidget(w)
            return widget

        self.days_box = row("Days (recent)", QSpinBox())
        self.days_box.setRange(10, 600); self.days_box.setValue(90)
        self.every_box = row("Sample every (s)", QSpinBox())
        self.every_box.setRange(15, 600); self.every_box.setValue(60); self.every_box.setSingleStep(15)
        self.horizon_box = row("Target horizon (s)", QSpinBox())
        self.horizon_box.setRange(60, 3600); self.horizon_box.setValue(900); self.horizon_box.setSingleStep(60)
        self.hold_box = row("Signal hold (s)", QSpinBox())
        self.hold_box.setRange(60, 3600); self.hold_box.setValue(900); self.hold_box.setSingleStep(60)
        self.target_box = row("Target", QComboBox())
        self.target_box.addItems(["fwd_ret", "fwd_straddle_ret"])

        self.train_btn = QPushButton("Train + Backtest"); self.train_btn.setObjectName("primary")
        self.train_btn.clicked.connect(self.train)
        form.addWidget(self.train_btn)
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.hide()
        form.addWidget(self.progress)
        self.status = QLabel(""); self.status.setObjectName("status"); self.status.setWordWrap(True)
        form.addWidget(self.status)
        ll.addWidget(cfg)

        ll.addWidget(QLabel("Walk-forward metrics"))
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ll.addWidget(self.metrics_table, stretch=1)

        ll.addWidget(QLabel("Feature information coefficient (|Spearman|)"))
        self.fi_table = QTableWidget(0, 2)
        self.fi_table.setHorizontalHeaderLabels(["Feature", "|IC|"])
        self.fi_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.fi_table.verticalHeader().setVisible(False)
        self.fi_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ll.addWidget(self.fi_table, stretch=1)

        right = QWidget(); rl = QVBoxLayout(right)
        self.title = QLabel("Out-of-sample signal equity"); self.title.setObjectName("title")
        rl.addWidget(self.title)
        self.charts = EquityReportChart()
        rl.addWidget(self.charts)

        split.addWidget(left); split.addWidget(right)
        split.setSizes([420, 950])

    def train(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        n = int(self.days_box.value())
        dates = common_dates(self.state.root)[-n:]
        if len(dates) < 10:
            self.status.setText("Not enough data days.")
            return
        params = dict(
            every_secs=int(self.every_box.value()),
            horizon_secs=int(self.horizon_box.value()),
            hold_secs=int(self.hold_box.value()),
            target=self.target_box.currentText(),
        )
        self.train_btn.setEnabled(False); self.progress.show()
        self.status.setText(f"Training over {len(dates)} days...")

        def job(progress_cb):
            from ..ml.pipeline import run_ml_pipeline
            return run_ml_pipeline(dates, self.state.root, progress_cb=progress_cb, **params)

        self.worker = FnWorker(job)
        self.worker.progress.connect(self.status.setText)
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _failed(self, tb: str) -> None:
        self.train_btn.setEnabled(True); self.progress.hide()
        self.status.setText(tb.splitlines()[-1]); print(tb)

    def _finished(self, result: dict) -> None:
        self.train_btn.setEnabled(True); self.progress.hide()
        payloads = result["payloads"]
        total = sum(p["pnl"] for p in payloads)
        self.status.setText(
            f"Done. {len(result['train_days'])} train / {len(result['test_days'])} test days. "
            f"OOS total ${total:,.2f}")

        m = result["metrics"]
        order = ["n_train", "n_test", "ic_overall", "ic_day_mean", "ic_day_t",
                 "hit_rate_all", "hit_rate_conviction"]
        extra = [("threshold |pred|", result["threshold"]),
                 ("OOS total P&L $", total),
                 ("OOS day win rate", float(np.mean([p["pnl"] > 0 for p in payloads])) if payloads else 0.0)]
        self.metrics_table.setRowCount(len(order) + len(extra))
        for r, k in enumerate(order):
            v = m[k]
            self.metrics_table.setItem(r, 0, QTableWidgetItem(k))
            self.metrics_table.setItem(r, 1, QTableWidgetItem(
                f"{v:.4f}" if isinstance(v, float) else str(v)))
        for i, (k, v) in enumerate(extra):
            r = len(order) + i
            self.metrics_table.setItem(r, 0, QTableWidgetItem(k))
            self.metrics_table.setItem(r, 1, QTableWidgetItem(f"{v:,.4f}"))

        fi = result["feature_ic"]
        self.fi_table.setRowCount(len(fi))
        max_ic = max((v for _, v in fi), default=1.0) or 1.0
        for r, (name, ic) in enumerate(fi):
            self.fi_table.setItem(r, 0, QTableWidgetItem(name))
            it = QTableWidgetItem(f"{ic:.4f}")
            it.setForeground(QColor(theme.ACCENT if ic >= 0.5 * max_ic else theme.FG_DIM))
            self.fi_table.setItem(r, 1, it)

        daily = np.array([p["pnl"] for p in payloads], np.float64)
        self.charts.set_results(daily)
        self.title.setText(f"Out-of-sample signal equity — {len(payloads)} test days, "
                           f"${total:,.2f}")
        # publish to shared state so Report + Backtest tabs can show these trades
        self.state.set_results(f"ML signal ({self.target_box.currentText()})", payloads)
