"""Shared application state across tabs."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from ..config import OPTION_ROOTS


class AppState(QObject):
    """Holds the active option root and the most recent backtest result set.

    results_changed fires whenever a new set of day payloads is published so
    the Report tab (and any other listener) can refresh.
    """

    results_changed = pyqtSignal()

    def __init__(self, root: str = "SPXW"):
        super().__init__()
        self.root = root
        self.strategy_label = ""
        self.payloads: list[dict] = []
        self.kronos = None  # lazily constructed KronosForecaster

    @property
    def cash_symbol(self) -> str:
        return OPTION_ROOTS[self.root][0]

    def set_results(self, label: str, payloads: list[dict]) -> None:
        self.strategy_label = label
        self.payloads = payloads
        self.results_changed.emit()

    def get_kronos(self, pred_len: int = 15):
        """Lazily build/refresh the Kronos forecaster (loads model on first use)."""
        if self.kronos is None or self.kronos.pred_len != pred_len:
            from ..ml.kronos_forecaster import KronosForecaster
            self.kronos = KronosForecaster(pred_len=pred_len)
        return self.kronos
