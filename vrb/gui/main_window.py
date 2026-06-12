"""VRB Trading Workbench main window: multi-tab shell."""

from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMainWindow,
                             QTabWidget, QVBoxLayout, QWidget)

from ..config import OPTION_ROOTS
from . import theme
from .state import AppState
from .tabs_backtest import BacktestTab
from .tabs_forecast import ForecastTab
from .tabs_gamma import GammaExplosionTab
from .tabs_mllab import MLLabTab
from .tabs_report import ReportTab
from .tabs_research import ResearchTab


class MainWindow(QMainWindow):
    def __init__(self, root: str = "SPXW"):
        super().__init__()
        self.setWindowTitle("VRB Trading Workbench — 0DTE Backtester & Kronos Forecaster")
        self.resize(1500, 950)
        self.state = AppState(root)

        header = QWidget()
        hl = QHBoxLayout(header); hl.setContentsMargins(10, 6, 10, 6)
        title = QLabel("VRB Trading Workbench"); title.setObjectName("title")
        hl.addWidget(title)
        hl.addStretch(1)
        hl.addWidget(QLabel("Option root"))
        self.root_box = QComboBox(); self.root_box.addItems(list(OPTION_ROOTS))
        self.root_box.setCurrentText(root)
        self.root_box.currentTextChanged.connect(self._root_changed)
        hl.addWidget(self.root_box)

        self.tabs = QTabWidget()
        self.backtest_tab = BacktestTab(self.state)
        self.gamma_tab = GammaExplosionTab(self.state)
        self.report_tab = ReportTab(self.state)
        self.forecast_tab = ForecastTab(self.state)
        self.mllab_tab = MLLabTab(self.state)
        self.research_tab = ResearchTab(self.state)
        self.tabs.addTab(self.backtest_tab, "Backtest")
        self.tabs.addTab(self.gamma_tab, "Gamma Explosion")
        self.tabs.addTab(self.report_tab, "Performance Report")
        self.tabs.addTab(self.mllab_tab, "ML Lab")
        self.tabs.addTab(self.forecast_tab, "Kronos Forecast")
        self.tabs.addTab(self.research_tab, "Research Journal")
        self.tabs.currentChanged.connect(self._tab_changed)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(header)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")
        self.state.results_changed.connect(self._results_changed)

    def _root_changed(self, root: str) -> None:
        self.state.root = root
        self.statusBar().showMessage(f"Active root: {root} (re-run to refresh)")

    def _tab_changed(self, idx: int) -> None:
        # refresh the journal whenever it's shown so new runs appear
        if self.tabs.widget(idx) is self.research_tab:
            self.research_tab.refresh()

    def _results_changed(self) -> None:
        self.report_tab.refresh()
        self.statusBar().showMessage(
            f"{self.state.strategy_label}: {len(self.state.payloads)} days loaded")

    def closeEvent(self, event) -> None:
        """Join any in-flight worker thread before teardown so Qt never
        destroys a running QThread (which would abort the process)."""
        for tab in (self.backtest_tab, self.gamma_tab, self.report_tab,
                    self.forecast_tab, self.mllab_tab):
            w = getattr(tab, "worker", None)
            if w is not None and w.isRunning():
                w.requestInterruption()
                w.quit()
                w.wait(5000)
        super().closeEvent(event)
