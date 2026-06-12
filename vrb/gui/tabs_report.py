"""Performance Report tab: TradeStation-style summary + equity graphs."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QSplitter,
                             QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from . import theme
from .charts import EquityReportChart
from .report import compute_report, monthly_table


class ReportTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        left = QWidget(); ll = QVBoxLayout(left)
        self.title = QLabel("Strategy Performance Report")
        self.title.setObjectName("title")
        ll.addWidget(self.title)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Metric", "All Trades", "Buys", "Sells"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ll.addWidget(self.table)

        right = QWidget(); rl = QVBoxLayout(right)
        self.charts = EquityReportChart()
        rl.addWidget(self.charts, stretch=3)
        rl.addWidget(QLabel("Periodical Returns (monthly)"))
        self.monthly = QTableWidget(0, 3)
        self.monthly.setHorizontalHeaderLabels(["Month", "Net Profit $", "Cumulative $"])
        self.monthly.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.monthly.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rl.addWidget(self.monthly, stretch=1)

        split.addWidget(left); split.addWidget(right)
        split.setSizes([560, 800])

        self.placeholder = QLabel("Run a backtest (Backtest tab or ML Lab) to populate the report.")
        self.placeholder.setObjectName("status")
        ll.addWidget(self.placeholder)

    def refresh(self) -> None:
        payloads = self.state.payloads
        if not payloads:
            return
        self.placeholder.hide()
        self.title.setText(f"Strategy Performance Report — {self.state.strategy_label} "
                           f"({len(payloads)} days)")
        model = compute_report(payloads)
        rows = model["rows"]
        self.table.setRowCount(len(rows))
        bold = QFont(); bold.setBold(True)
        for r, (kind, label, va, vb, vs) in enumerate(rows):
            if kind == "section":
                it = QTableWidgetItem(label)
                it.setFont(bold)
                it.setForeground(QColor(theme.ACCENT))
                self.table.setItem(r, 0, it)
                self.table.setSpan(r, 0, 1, 4)
                continue
            self.table.setItem(r, 0, QTableWidgetItem("    " + label))
            for c, v in ((1, va), (2, vb), (3, vs)):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if v.startswith("($") or v.startswith("-"):
                    it.setForeground(QColor(theme.LOSS))
                elif v.startswith("$") and v not in ("$0.00",):
                    it.setForeground(QColor(theme.WIN))
                self.table.setItem(r, c, it)

        self.charts.set_results(model["daily"])
        months = monthly_table(payloads)
        self.monthly.setRowCount(len(months))
        for r, (m, pnl, cum) in enumerate(months):
            self.monthly.setItem(r, 0, QTableWidgetItem(m))
            for c, v in ((1, pnl), (2, cum)):
                it = QTableWidgetItem(f"{v:,.2f}")
                it.setForeground(QColor(theme.WIN if v >= 0 else theme.LOSS))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.monthly.setItem(r, c, it)
