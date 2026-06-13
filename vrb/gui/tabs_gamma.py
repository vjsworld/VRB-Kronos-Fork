"""Last Hour Gamma Explosion tab.

Configure the SuperTrend signal (on a 1-min ES/NQ chart) and the 0DTE option
trade parameters, run the backtest, then pick a day to see the futures
candles with the SuperTrend line overlaid and the signals/exits marked:
blue arrows = long signals (buy call), red = short signals (buy put), white =
exits.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from ..backtest.strategies import LastHourGammaExplosion
from ..data.calendar import common_dates
from . import theme
from .charts import SignalChart
from .workers import FnWorker, run_backtest_days, supertrend_for_day


class GammaExplosionTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.worker: FnWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        left = QWidget(); ll = QVBoxLayout(left)
        cfg = QGroupBox("Last Hour Gamma Explosion"); form = QVBoxLayout(cfg)

        def add(label, widget):
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(label)); h.addWidget(widget, stretch=1)
            form.addWidget(w)
            return widget

        self.symbol_box = add("Signal chart", QComboBox()); self.symbol_box.addItems(["ES", "NQ"])
        self.days_box = add("Days (recent)", QSpinBox())
        self.days_box.setRange(2, 600); self.days_box.setValue(30)

        form.addWidget(self._sub("SuperTrend"))
        self.atr_period_box = add("ATR period", QSpinBox())
        self.atr_period_box.setRange(2, 100); self.atr_period_box.setValue(10)
        self.atr_mult_box = add("ATR multiplier", QDoubleSpinBox())
        self.atr_mult_box.setRange(0.5, 15.0); self.atr_mult_box.setValue(3.0); self.atr_mult_box.setSingleStep(0.5)

        form.addWidget(self._sub("Trade"))
        self.delta_box = add("Target delta", QDoubleSpinBox())
        self.delta_box.setRange(0.02, 0.5); self.delta_box.setValue(0.20); self.delta_box.setSingleStep(0.01)
        self.target_box = add("Profit target (x cost)", QDoubleSpinBox())
        self.target_box.setRange(1.5, 50.0); self.target_box.setValue(5.0); self.target_box.setSingleStep(0.5)
        self.qty_box = add("Contracts", QSpinBox()); self.qty_box.setRange(1, 100); self.qty_box.setValue(1)
        self.mintte_box = add("Min time-to-expiry (s)", QSpinBox())
        self.mintte_box.setRange(0, 3600); self.mintte_box.setValue(120); self.mintte_box.setSingleStep(30)
        self.reverse_box = QCheckBox("Reverse on opposite signal (stop-and-reverse)")
        self.reverse_box.setChecked(True)
        form.addWidget(self.reverse_box)
        self.invert_box = QCheckBox("Reverse Entry Sig (buy put on long, call on short)")
        self.invert_box.setChecked(False)
        form.addWidget(self.invert_box)

        form.addWidget(self._sub("Signal window (CT)"))
        self.entry_box = add("Start time (hr)", QDoubleSpinBox())
        self.entry_box.setRange(8.5, 14.99); self.entry_box.setValue(14.0); self.entry_box.setSingleStep(0.25)
        self.exit_box = add("End time (hr)", QDoubleSpinBox())
        self.exit_box.setRange(8.5, 15.0); self.exit_box.setValue(15.0); self.exit_box.setSingleStep(0.25)

        self.notes_edit = add("Experiment notes", QLineEdit())
        self.notes_edit.setPlaceholderText("optional label, recorded to journal")
        self.run_btn = QPushButton("Run Backtest"); self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.run_backtest)
        form.addWidget(self.run_btn)
        self.status = QLabel(""); self.status.setObjectName("status"); self.status.setWordWrap(True)
        form.addWidget(self.status)
        ll.addWidget(cfg)

        ll.addWidget(QLabel("Days — select to chart"))
        self.day_table = QTableWidget(0, 3)
        self.day_table.setHorizontalHeaderLabels(["Date", "P&L $", "Trades"])
        self.day_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.day_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.day_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.day_table.setAlternatingRowColors(True)
        self.day_table.itemSelectionChanged.connect(self._day_selected)
        ll.addWidget(self.day_table, stretch=1)

        right = QWidget(); rl = QVBoxLayout(right)
        self.chart_title = QLabel("Run a backtest, then select a day")
        self.chart_title.setObjectName("title")
        rl.addWidget(self.chart_title)
        self.chart = SignalChart()
        self.chart.price_plot.setLabel("left", "ES (1-min) + SuperTrend")
        rl.addWidget(self.chart, stretch=3)
        self.trade_table = QTableWidget(0, 6)
        self.trade_table.setHorizontalHeaderLabels(
            ["Entry", "Exit", "Option", "Signal", "Cost", "P&L $"])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trade_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rl.addWidget(self.trade_table, stretch=1)

        split.addWidget(left); split.addWidget(right)
        split.setSizes([360, 1080])

    def _sub(self, text: str) -> QLabel:
        lbl = QLabel(text); lbl.setStyleSheet(f"color:{theme.ACCENT}; font-weight:700; margin-top:4px;")
        return lbl

    # ------------------------------------------------------------- helpers
    def _hhmmss(self, hr: float) -> str:
        h = int(hr); m = int(round((hr - h) * 60))
        if m == 60:
            h, m = h + 1, 0
        return f"{h:02d}:{m:02d}:00"

    def _params(self) -> dict:
        return dict(
            symbol=self.symbol_box.currentText(),
            entry_time=self._hhmmss(self.entry_box.value()),
            exit_time=self._hhmmss(self.exit_box.value()),
            atr_period=int(self.atr_period_box.value()),
            atr_mult=float(self.atr_mult_box.value()),
            target_mult=float(self.target_box.value()),
            target_delta=float(self.delta_box.value()),
            qty=int(self.qty_box.value()),
            min_tte=int(self.mintte_box.value()),
            reverse=self.reverse_box.isChecked(),
            invert=self.invert_box.isChecked(),
        )

    # -------------------------------------------------------------- actions
    def run_backtest(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        p = self._params()
        self._active = p
        dates = common_dates(self.state.root)[-int(self.days_box.value()):]
        if not dates:
            self.status.setText("No common data days.")
            return

        kwargs = dict(
            entry_time=p["entry_time"], exit_time=p["exit_time"],
            atr_period=p["atr_period"], atr_mult=p["atr_mult"],
            target_mult=p["target_mult"], target_delta=p["target_delta"],
            qty=p["qty"], signal_symbol=p["symbol"], min_tte_secs=p["min_tte"],
            reverse_on_opposite=p["reverse"], invert_signals=p["invert"])

        self.run_btn.setEnabled(False)
        self.status.setText(f"Running over {len(dates)} days (parallel)...")

        def job(progress_cb):
            from ..backtest.parallel import run_days_parallel
            return run_days_parallel(dates, LastHourGammaExplosion, kwargs,
                                     self.state.root, progress_cb=progress_cb)

        self.worker = FnWorker(job)
        self.worker.progress.connect(self.status.setText)
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _failed(self, tb: str) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText(tb.splitlines()[-1]); print(tb)

    def _finished(self, payloads: list[dict]) -> None:
        self.run_btn.setEnabled(True)
        wins = sum(1 for p in payloads for t in p["trades"] if t["reason"] == "target")
        n = sum(p["n_trades"] for p in payloads)
        total = sum(p["pnl"] for p in payloads)
        spread = sum(p.get("cost_spread", 0.0) for p in payloads)
        self.status.setText(f"Done: {len(payloads)} days, {n} trades, {wins} hit target. "
                            f"Net ${total:,.0f}  (spread cost ${spread:,.0f})")
        self.state.set_results("Last Hour Gamma Explosion", payloads)
        self.populate_days(payloads)
        # record this experiment to the permanent research journal
        try:
            from ..research import registry
            registry.record("LastHourGammaExplosion", getattr(self, "_active", {}),
                            self.state.root, [p["date"] for p in payloads], payloads,
                            notes=self.notes_edit.text().strip())
        except Exception as e:  # journaling must never break the run
            print("journal record failed:", e)

    def populate_days(self, payloads: list[dict]) -> None:
        self.day_table.setRowCount(len(payloads))
        for r, p in enumerate(payloads):
            d = p["date"]
            items = [QTableWidgetItem(f"{d[:4]}-{d[4:6]}-{d[6:]}"),
                     QTableWidgetItem(f"{p['pnl']:,.2f}"),
                     QTableWidgetItem(str(p["n_trades"]))]
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
        self.show_day(self.state.payloads[rows[0].row()])

    def show_day(self, p: dict) -> None:
        date = p["date"]
        a = getattr(self, "_active", self._params())
        self.chart_title.setText(
            f"Last Hour Gamma Explosion — {date[:4]}-{date[4:6]}-{date[6:]} — "
            f"P&L ${p['pnl']:,.2f} — {p['n_trades']} trades")
        try:
            st = supertrend_for_day(date, a["symbol"], a["atr_period"], a["atr_mult"],
                                    a["entry_time"], a["exit_time"])
        except FileNotFoundError:
            self.chart_title.setText(f"{date}: no {a['symbol']} data")
            return
        self.chart.set_candles(st["ts"], st["open"], st["high"], st["low"], st["close"])
        self.chart.add_supertrend(st["ts"], st["st"], st["direction"])

        # SPX cash line (left axis) + held-option premium (right axis) with the
        # trade markers anchored to the OPTION premium curve (the instrument we
        # actually trade), not the ES underlying that produced the signal
        self._overlay_underlying_and_premium(date, p)

        self.chart.set_equity(p["equity_ts"][::6], p["equity"][::6])
        self.chart.set_equity_underlay(st["ts"], st["close"], color=theme.FG, label="ES")

        self.trade_table.setRowCount(len(p["trades"]))
        for r, t in enumerate(p["trades"]):
            signal = "LONG" if t["direction"] == "buy" else "SHORT"
            vals = [str(t["entry_ts"])[11:], str(t["exit_ts"])[11:], t["legs"],
                    signal, f"{abs(t['entry_value']):.2f}", f"{t['pnl']:,.2f}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 3:
                    it.setForeground(QColor(theme.BUY if signal == "LONG" else theme.SELL))
                elif c == 5:
                    it.setForeground(QColor(theme.WIN if t["pnl"] >= 0 else theme.LOSS))
                self.trade_table.setItem(r, c, it)

    def _overlay_underlying_and_premium(self, date: str, p: dict) -> None:
        """Add SPX cash as a left-axis line and the held-option premium on the
        right axis (call premium while long a call, put premium while long a put)."""
        from ..options.chain import CALL, ChainDay
        try:
            day = ChainDay.load(date, self.state.root)
        except (FileNotFoundError, ValueError):
            return
        # SPX cash line on the left axis (what the options actually settle to)
        self.chart.add_overlay_line(day.ts, day.spot, theme.FG, width=1.5, name="SPX")

        # premium of whichever option is held, switching call<->put with position
        call_prem = np.full(len(day.ts), np.nan)
        put_prem = np.full(len(day.ts), np.nan)
        for t in p["trades"]:
            for leg in (t.get("legs_detail") or []):
                k = day.k_index(leg["strike"])
                right = leg["right"]
                i0 = int(np.searchsorted(day.ts, np.datetime64(t["entry_ts"], "s")))
                i1 = int(np.searchsorted(day.ts, np.datetime64(t["exit_ts"], "s")))
                i0 = max(0, min(i0, len(day.ts) - 1))
                i1 = max(i0, min(i1, len(day.ts) - 1))
                seg = day.mid[i0:i1 + 1, k, right]
                target = call_prem if right == CALL else put_prem
                target[i0:i1 + 1] = seg
        self.chart.set_premium(day.ts, call_prem, put_prem)
        # markers anchored to the option premium, not the ES candles
        self.chart.add_premium_markers(p["trades"], day.ts, call_prem, put_prem)
