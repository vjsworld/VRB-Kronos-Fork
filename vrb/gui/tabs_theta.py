"""Theta Harvesting tab — sell defined-risk 0DTE premium in the decay window.

The seller's side of the gamma play: instead of paying theta on long options,
we collect it by selling an iron condor (defined risk) or a short strangle
(undefined risk) at ~target delta, managed with a profit target (fraction of
credit) and a stop (multiple of credit), else held to 15:00 settlement.

Same layout/feel as the Gamma tab: ES candles + SPX line on the left axis, the
sold structure's decaying buyback cost on the right axis, the short strikes
drawn as a 'tent', a day list, a trade table, and journal recording.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from ..backtest.strategies import (IronCondor, ShortStrangle,
                                    SuperTrendCreditSpread)
from ..data.calendar import common_dates
from ..data.theta import CALL
from . import theme
from .charts import SignalChart
from .workers import FnWorker, supertrend_for_day

PREMIUM_COLOR = "#ffca28"  # gold: the credit we're harvesting


class ThetaHarvestTab(QWidget):
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
        cfg = QGroupBox("Theta Harvesting (sell premium)"); form = QVBoxLayout(cfg)

        def add(label, widget):
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(label)); h.addWidget(widget, stretch=1)
            form.addWidget(w)
            return widget

        self.structure_box = add("Structure", QComboBox())
        self.structure_box.addItems(["Iron Condor", "Short Strangle",
                                     "SuperTrend Credit Spread"])
        self.structure_box.currentTextChanged.connect(self._toggle_structure)
        self.symbol_box = add("Signal / context chart", QComboBox()); self.symbol_box.addItems(["ES", "NQ"])
        self.days_box = add("Days (recent)", QSpinBox())
        self.days_box.setRange(2, 600); self.days_box.setValue(60)

        self.st_header = self._sub("SuperTrend signal")
        form.addWidget(self.st_header)
        self.atr_period_box = add("ATR period", QSpinBox())
        self.atr_period_box.setRange(2, 100); self.atr_period_box.setValue(10)
        self.atr_mult_box = add("ATR multiplier", QDoubleSpinBox())
        self.atr_mult_box.setRange(0.5, 15.0); self.atr_mult_box.setValue(3.0); self.atr_mult_box.setSingleStep(0.5)
        self.reverse_box = QCheckBox("Reverse on opposite signal")
        self.reverse_box.setChecked(True); form.addWidget(self.reverse_box)

        form.addWidget(self._sub("Structure"))
        self.delta_box = add("Short delta", QDoubleSpinBox())
        self.delta_box.setRange(0.03, 0.45); self.delta_box.setValue(0.16); self.delta_box.setSingleStep(0.01)
        self.wing_box = add("Wing width (pts)", QDoubleSpinBox())
        self.wing_box.setRange(5.0, 200.0); self.wing_box.setValue(30.0); self.wing_box.setSingleStep(5.0)
        self.qty_box = add("Contracts", QSpinBox()); self.qty_box.setRange(1, 100); self.qty_box.setValue(1)

        form.addWidget(self._sub("Management"))
        self.profit_box = add("Profit target (frac of credit)", QDoubleSpinBox())
        self.profit_box.setRange(0.05, 0.95); self.profit_box.setValue(0.5); self.profit_box.setSingleStep(0.05)
        self.stop_box = add("Stop (x credit)", QDoubleSpinBox())
        self.stop_box.setRange(1.1, 10.0); self.stop_box.setValue(2.0); self.stop_box.setSingleStep(0.5)

        form.addWidget(self._sub("Entry window (CT)"))
        self.entry_box = add("Entry time (hr)", QDoubleSpinBox())
        self.entry_box.setRange(8.5, 14.5); self.entry_box.setValue(10.0); self.entry_box.setSingleStep(0.25)
        self.exit_box = add("Exit time (hr)", QDoubleSpinBox())
        self.exit_box.setRange(9.0, 15.0); self.exit_box.setValue(15.0); self.exit_box.setSingleStep(0.25)

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
        self.chart.price_plot.setLabel("left", "ES (1-min) + SPX + short strikes")
        rl.addWidget(self.chart, stretch=3)
        self.trade_table = QTableWidget(0, 6)
        self.trade_table.setHorizontalHeaderLabels(
            ["Entry", "Exit", "Structure", "Credit", "Exit reason", "P&L $"])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trade_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rl.addWidget(self.trade_table, stretch=1)

        split.addWidget(left); split.addWidget(right)
        split.setSizes([360, 1080])
        self._toggle_structure(self.structure_box.currentText())

    def _sub(self, text: str) -> QLabel:
        lbl = QLabel(text); lbl.setStyleSheet(f"color:{theme.ACCENT}; font-weight:700; margin-top:4px;")
        return lbl

    def _toggle_structure(self, structure: str) -> None:
        is_credit = structure == "SuperTrend Credit Spread"
        # SuperTrend signal controls only apply to the credit-spread structure
        self.st_header.setVisible(is_credit)
        self.atr_period_box.parent().setVisible(is_credit)
        self.atr_mult_box.parent().setVisible(is_credit)
        self.reverse_box.setVisible(is_credit)
        # wings apply to condor and credit spread, not the naked strangle
        self.wing_box.setEnabled(structure != "Short Strangle")

    # ------------------------------------------------------------- helpers
    def _hhmmss(self, hr: float) -> str:
        h = int(hr); m = int(round((hr - h) * 60))
        if m == 60:
            h, m = h + 1, 0
        return f"{h:02d}:{m:02d}:00"

    def _spec(self):
        entry, exit_ = self._hhmmss(self.entry_box.value()), self._hhmmss(self.exit_box.value())
        delta, qty = self.delta_box.value(), int(self.qty_box.value())
        stop, profit = self.stop_box.value(), self.profit_box.value()
        structure = self.structure_box.currentText()
        if structure == "Iron Condor":
            return IronCondor, dict(entry_time=entry, exit_time=exit_, target_delta=delta,
                                    wing_pts=self.wing_box.value(), stop_mult=stop,
                                    profit_frac=profit, qty=qty)
        if structure == "SuperTrend Credit Spread":
            return SuperTrendCreditSpread, dict(
                entry_time=entry, exit_time=exit_, atr_period=int(self.atr_period_box.value()),
                atr_mult=float(self.atr_mult_box.value()), short_delta=delta,
                wing_pts=self.wing_box.value(), stop_mult=stop, profit_frac=profit,
                qty=qty, signal_symbol=self.symbol_box.currentText(),
                reverse_on_opposite=self.reverse_box.isChecked())
        return ShortStrangle, dict(entry_time=entry, exit_time=exit_, target_delta=delta,
                                   stop_mult=stop, profit_frac=profit, qty=qty)

    # -------------------------------------------------------------- actions
    def run_backtest(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        cls, kwargs = self._spec()
        self._active = (cls, kwargs)
        dates = common_dates(self.state.root)[-int(self.days_box.value()):]
        if not dates:
            self.status.setText("No common data days.")
            return
        label = self.structure_box.currentText()
        self.run_btn.setEnabled(False)
        self.status.setText(f"Running {label} over {len(dates)} days (parallel)...")

        def job(progress_cb):
            from ..backtest.parallel import run_days_parallel
            return run_days_parallel(dates, cls, kwargs, self.state.root, progress_cb=progress_cb)

        self.worker = FnWorker(job)
        self.worker.progress.connect(self.status.setText)
        self.worker.done.connect(lambda pls: self._finished(cls.__name__, kwargs, pls))
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _failed(self, tb: str) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText(tb.splitlines()[-1]); print(tb)

    def _finished(self, cls_name: str, kwargs: dict, payloads: list[dict]) -> None:
        self.run_btn.setEnabled(True)
        n = sum(p["n_trades"] for p in payloads)
        total = sum(p["pnl"] for p in payloads)
        gross = sum(p.get("gross_mid_pnl", 0.0) for p in payloads)
        self.status.setText(f"Done: {len(payloads)} days, {n} trades. "
                            f"Net ${total:,.0f}  (gross@mid ${gross:,.0f})")
        self.state.set_results(f"Theta: {self.structure_box.currentText()}", payloads)
        self.populate_days(payloads)
        try:
            from ..research import registry
            registry.record(cls_name, kwargs, self.state.root,
                            [p["date"] for p in payloads], payloads,
                            notes=self.notes_edit.text().strip())
        except Exception as e:
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
        from ..options.chain import ChainDay
        date = p["date"]
        sym = self.symbol_box.currentText()
        self.chart_title.setText(
            f"{self.structure_box.currentText()} — {date[:4]}-{date[4:6]}-{date[6:]} — "
            f"P&L ${p['pnl']:,.2f} — {p['n_trades']} trades")
        try:
            st = supertrend_for_day(date, sym, int(self.atr_period_box.value()),
                                    float(self.atr_mult_box.value()),
                                    self._hhmmss(self.entry_box.value()),
                                    self._hhmmss(self.exit_box.value()))
            day = ChainDay.load(date, self.state.root)
        except (FileNotFoundError, ValueError):
            self.chart_title.setText(f"{date}: missing data")
            return
        self.chart.set_candles(st["ts"], st["open"], st["high"], st["low"], st["close"])
        self.chart.add_supertrend(st["ts"], st["st"], st["direction"])
        self.chart.add_overlay_line(day.ts, day.spot, theme.FG, width=1.5, name="SPX")

        # structure buyback cost (decaying credit) on the right axis + the tent
        pos_prem = np.full(len(day.ts), np.nan)
        markers, segments = [], []
        for t in p["trades"]:
            legs = t.get("legs_detail") or []
            if not legs:
                continue
            i0 = int(np.clip(np.searchsorted(day.ts, np.datetime64(t["entry_ts"], "s")), 0, len(day.ts) - 1))
            i1 = int(np.clip(np.searchsorted(day.ts, np.datetime64(t["exit_ts"], "s")), 0, len(day.ts) - 1))
            buyback = np.zeros(i1 - i0 + 1)
            for leg in legs:
                k = day.k_index(leg["strike"])
                seg = day.mid[i0:i1 + 1, k, leg["right"]]
                buyback += -leg["qty"] * np.nan_to_num(seg, nan=0.0)
                if leg["qty"] < 0:  # short strikes form the tent
                    segments.append((t["entry_ts"], t["exit_ts"], leg["strike"], theme.FG_DIM))
            pos_prem[i0:i1 + 1] = buyback
            markers.append({"x": t["entry_ts"], "y": float(buyback[0]), "kind": "entry",
                            "color": PREMIUM_COLOR,
                            "text": f"SELL @ {abs(t['entry_value']):.2f}"})
            pnl = t["pnl"]
            markers.append({"x": t["exit_ts"], "y": float(buyback[-1]), "kind": "exit",
                            "color": theme.EXIT,
                            "text": f"{t['reason'].upper()}  {pnl:+,.0f}",
                            "label_color": theme.WIN if pnl >= 0 else theme.LOSS})

        self.chart.add_strike_segments(segments)
        self.chart.set_premium(day.ts, pos_prem, None, color_a=PREMIUM_COLOR,
                               label="Structure buyback cost (pts)")
        self.chart.add_premium_markers_xy(markers)
        self.chart.set_equity(p["equity_ts"][::6], p["equity"][::6])

        self.trade_table.setRowCount(len(p["trades"]))
        for r, t in enumerate(p["trades"]):
            vals = [str(t["entry_ts"])[11:], str(t["exit_ts"])[11:], t["legs"],
                    f"{abs(t['entry_value']):.2f}", t["reason"], f"{t['pnl']:,.2f}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 5:
                    it.setForeground(QColor(theme.WIN if t["pnl"] >= 0 else theme.LOSS))
                self.trade_table.setItem(r, c, it)
