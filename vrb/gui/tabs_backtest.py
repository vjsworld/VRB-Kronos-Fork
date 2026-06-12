"""Backtest tab: strategy config -> run -> day list -> signal chart.

Select a day in the results table and the candlestick chart shows that
session with entry arrows (blue=buy, red=sell), white exit arrows, and the
intraday equity curve underneath.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QPushButton, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from ..backtest.strategies import IronCondor, ShortStraddle
from ..data.calendar import common_dates
from . import theme
from .charts import SignalChart
from .workers import FnWorker, candles_for_day, run_backtest_days


class BacktestTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state          # shared AppState
        self.worker: FnWorker | None = None
        self._build_ui()

    # ----------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        # left: controls + day list
        left = QWidget()
        ll = QVBoxLayout(left)

        cfg = QGroupBox("Strategy")
        form = QVBoxLayout(cfg)
        self.strategy_box = QComboBox()
        self.strategy_box.addItems(["Short Straddle", "Iron Condor"])
        form.addWidget(self.strategy_box)

        def spin(label, lo, hi, val, step=1.0, decimals=2):
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(label))
            box = QDoubleSpinBox()
            box.setRange(lo, hi); box.setValue(val)
            box.setSingleStep(step); box.setDecimals(decimals)
            h.addWidget(box)
            form.addWidget(w)
            return box

        self.days_box = QSpinBox(); self.days_box.setRange(2, 600); self.days_box.setValue(30)
        wrap = QWidget(); h = QHBoxLayout(wrap); h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("Days (most recent)")); h.addWidget(self.days_box)
        form.addWidget(wrap)

        self.entry_box = spin("Entry time (decimal hr CT)", 8.5, 14.9, 9.0, 0.25)
        self.exit_box = spin("Exit time (decimal hr CT)", 9.0, 15.0, 14.75, 0.25)
        self.stop_box = spin("Stop (x credit)", 1.1, 10.0, 2.0, 0.1)
        self.target_box = spin("Profit target (fraction)", 0.05, 0.95, 0.5, 0.05)
        self.delta_box = spin("Condor short delta", 0.03, 0.45, 0.16, 0.01)
        self.wing_box = spin("Condor wing (pts)", 5.0, 200.0, 25.0, 5.0, 0)

        self.run_btn = QPushButton("Run Backtest"); self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.run_backtest)
        form.addWidget(self.run_btn)
        self.status = QLabel(""); self.status.setObjectName("status"); self.status.setWordWrap(True)
        form.addWidget(self.status)
        ll.addWidget(cfg)

        ll.addWidget(QLabel("Days — select to chart"))
        self.day_table = QTableWidget(0, 4)
        self.day_table.setHorizontalHeaderLabels(["Date", "P&L $", "Trades", "Exits"])
        self.day_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.day_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.day_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.day_table.setAlternatingRowColors(True)
        self.day_table.itemSelectionChanged.connect(self._day_selected)
        ll.addWidget(self.day_table, stretch=1)

        # right: chart + per-day trade list
        right = QWidget()
        rl = QVBoxLayout(right)
        self.chart_title = QLabel("Run a backtest, then select a day")
        self.chart_title.setObjectName("title")
        rl.addWidget(self.chart_title)
        self.chart = SignalChart()
        rl.addWidget(self.chart, stretch=3)
        self.trade_table = QTableWidget(0, 7)
        self.trade_table.setHorizontalHeaderLabels(
            ["Entry", "Exit", "Structure", "Side", "Fill", "Exit reason", "P&L $"])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trade_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rl.addWidget(self.trade_table, stretch=1)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([330, 1100])

    # ------------------------------------------------------------- actions
    def _hhmmss(self, decimal_hr: float) -> str:
        h = int(decimal_hr)
        m = int(round((decimal_hr - h) * 60))
        return f"{h:02d}:{m:02d}:00"

    def strategy_factory(self):
        entry, exit_ = self._hhmmss(self.entry_box.value()), self._hhmmss(self.exit_box.value())
        stop, target = self.stop_box.value(), self.target_box.value()
        if self.strategy_box.currentText() == "Short Straddle":
            return lambda: ShortStraddle(entry, exit_, stop, target)
        delta, wing = self.delta_box.value(), self.wing_box.value()
        return lambda: IronCondor(entry, exit_, delta, wing, stop, target)

    def run_backtest(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        n = int(self.days_box.value())
        dates = common_dates(self.state.root)[-n:]
        if not dates:
            self.status.setText("No common data days found.")
            return
        factory = self.strategy_factory()
        label = self.strategy_box.currentText()
        self.run_btn.setEnabled(False)
        self.status.setText(f"Running {label} over {len(dates)} days...")

        def job(progress_cb):
            return run_backtest_days(dates, factory, self.state.root, progress_cb)

        self.worker = FnWorker(job)
        self.worker.progress.connect(self.status.setText)
        self.worker.done.connect(lambda payloads: self._finished(label, payloads))
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _failed(self, tb: str) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText(tb.splitlines()[-1])
        print(tb)

    def _finished(self, label: str, payloads: list[dict]) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText(f"Done: {len(payloads)} days.")
        self.state.set_results(label, payloads)
        self.populate_days(payloads)

    # -------------------------------------------------------------- results
    def populate_days(self, payloads: list[dict]) -> None:
        self.day_table.setRowCount(len(payloads))
        for r, p in enumerate(payloads):
            d = p["date"]
            items = [
                QTableWidgetItem(f"{d[:4]}-{d[4:6]}-{d[6:]}"),
                QTableWidgetItem(f"{p['pnl']:,.2f}"),
                QTableWidgetItem(str(p["n_trades"])),
                QTableWidgetItem(p["reasons"]),
            ]
            items[1].setForeground(QColor(theme.WIN if p["pnl"] >= 0 else theme.LOSS))
            for c, it in enumerate(items):
                if c in (1, 2):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.day_table.setItem(r, c, it)
        if payloads:
            self.day_table.selectRow(len(payloads) - 1)

    def _day_selected(self) -> None:
        rows = self.day_table.selectionModel().selectedRows()
        if not rows or not self.state.payloads:
            return
        p = self.state.payloads[rows[0].row()]
        self.show_day(p)

    def show_day(self, p: dict) -> None:
        date = p["date"]
        self.chart_title.setText(
            f"{self.state.strategy_label} — {date[:4]}-{date[4:6]}-{date[6:]} — "
            f"P&L ${p['pnl']:,.2f} — settle {p['spot_close']:,.2f}")
        try:
            candles = candles_for_day(date, self.state.cash_symbol)
        except FileNotFoundError:
            self.chart_title.setText(f"{date}: no underlying data")
            return
        self.chart.set_candles(candles["ts"], candles["open"], candles["high"],
                               candles["low"], candles["close"])
        self.chart.set_equity(p["equity_ts"][::6], p["equity"][::6])  # 30s marks
        self.chart.add_trade_markers(p["trades"])

        self.trade_table.setRowCount(len(p["trades"]))
        for r, t in enumerate(p["trades"]):
            vals = [
                str(t["entry_ts"])[11:], str(t["exit_ts"])[11:], t["legs"],
                "BUY" if t["direction"] == "buy" else "SELL",
                f"{abs(t['entry_value']):.2f}", t["reason"], f"{t['pnl']:,.2f}",
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 6:
                    it.setForeground(QColor(theme.WIN if t["pnl"] >= 0 else theme.LOSS))
                self.trade_table.setItem(r, c, it)
