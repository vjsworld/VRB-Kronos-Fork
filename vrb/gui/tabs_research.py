"""Research Journal tab: the permanent, sortable record of every backtest.

Each row is one recorded run with its strategy, key parameters, and the stats
that matter for research — net P&L, the spread/commission cost drag, Sharpe,
win rates, profit factor, drawdown. Click a header to sort; this is where we
compare experiments and decide what to try next.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QPushButton,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..research import registry
from . import theme

COLUMNS = [
    ("Logged", "logged_at"), ("Strategy", "strategy"), ("Root", "root"),
    ("Start", "start"), ("End", "end"), ("Days", "n_days"), ("Trades", "n_trades"),
    ("Net $", "total_pnl"), ("Gross@mid $", "gross_mid_pnl"),
    ("Spread $", "cost_spread"), ("Comm $", "cost_commission"),
    ("Sharpe", "sharpe_daily_ann"), ("Day win%", "day_win_rate"),
    ("Trade win%", "trade_win_rate"), ("PF", "profit_factor"),
    ("MaxDD $", "max_drawdown"), ("Params", "_params"), ("Notes", "notes"),
]


class _NumItem(QTableWidgetItem):
    """Table item that sorts numerically by a stored value."""
    def __init__(self, text: str, value: float):
        super().__init__(text)
        self._v = value
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other):
        return self._v < getattr(other, "_v", 0)


class ResearchTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        title = QLabel("Research Journal — every backtest, permanently recorded")
        title.setObjectName("title")
        bar.addWidget(title)
        bar.addStretch(1)
        self.summary = QLabel(""); self.summary.setObjectName("status")
        bar.addWidget(self.summary)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        bar.addWidget(refresh)
        root.addLayout(bar)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in COLUMNS])
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(len(COLUMNS) - 2, QHeaderView.ResizeMode.Interactive)
        root.addWidget(self.table)

        self.refresh()

    @staticmethod
    def _params_str(params: dict) -> str:
        keep = ("atr_period", "atr_mult", "target_mult", "target_delta",
                "entry_time", "exit_time", "reverse_on_opposite", "invert_signals",
                "signal_symbol", "stop_mult", "profit_frac", "wing_pts")
        bits = []
        for k in keep:
            if k in params:
                v = params[k]
                short = k.replace("_", "")[:6]
                bits.append(f"{short}={v}")
        return " ".join(bits) or str(params)

    def refresh(self) -> None:
        runs = registry.load_all()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(runs))
        pct = {"day_win_rate", "trade_win_rate"}
        money = {"total_pnl", "gross_mid_pnl", "cost_spread", "cost_commission", "max_drawdown"}
        for r, run in enumerate(runs):
            stats = run.get("stats", {})
            for c, (_label, key) in enumerate(COLUMNS):
                if key == "_params":
                    it = QTableWidgetItem(self._params_str(run.get("params", {})))
                elif key in ("logged_at", "strategy", "root", "start", "end", "notes"):
                    it = QTableWidgetItem(str(run.get(key, "")))
                else:
                    v = stats.get(key, 0)
                    fv = float(v) if isinstance(v, (int, float)) else 0.0
                    if key in pct:
                        txt = f"{fv * 100:.0f}%"
                    elif key in money:
                        txt = f"{fv:,.0f}"
                    elif key == "profit_factor":
                        txt = "inf" if fv == float("inf") else f"{fv:.2f}"
                    elif key in ("n_days", "n_trades"):
                        txt = str(int(fv))
                    else:
                        txt = f"{fv:.2f}"
                    it = _NumItem(txt, fv)
                    if key in money or key == "total_pnl":
                        it.setForeground(QColor(theme.WIN if fv >= 0 else theme.LOSS))
                    if key == "total_pnl":
                        it.setForeground(QColor(theme.WIN if fv >= 0 else theme.LOSS))
                self.table.setItem(r, c, it)
        self.table.setSortingEnabled(True)

        best = max((run["stats"].get("total_pnl", 0) for run in runs if run.get("stats")),
                   default=0.0)
        self.summary.setText(f"{len(runs)} runs recorded — best net ${best:,.0f}")
