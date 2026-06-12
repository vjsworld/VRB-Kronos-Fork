"""Render every tab offscreen to PNG with real data driven through it.

Run:  QT_QPA_PLATFORM=offscreen python -m vrb.tests.screenshot_gui
Writes vrb_out/shot_*.png and asserts each tab produced a non-blank image.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from vrb.backtest.payload import run_backtest_days
from vrb.backtest.strategies import IronCondor, ShortStraddle
from vrb.data.calendar import common_dates
from vrb.gui import theme
from vrb.gui.main_window import MainWindow

OUT = Path(__file__).resolve().parent.parent.parent / "vrb_out"
OUT.mkdir(exist_ok=True)


def grab(widget, name: str) -> None:
    app = QApplication.instance()
    for _ in range(6):
        app.processEvents()
    pix = widget.grab()
    path = OUT / f"shot_{name}.png"
    pix.save(str(path))
    img = pix.toImage()
    # crude non-blank check: sample pixels, ensure >1 distinct color
    colors = {img.pixel(x, y) for x in range(0, img.width(), 47)
              for y in range(0, img.height(), 47)}
    print(f"  {name}: {path.name}  {img.width()}x{img.height()}  {len(colors)} distinct colors")
    assert len(colors) > 3, f"{name} looks blank ({len(colors)} colors)"


def main() -> None:
    theme.apply_pg_theme()
    app = QApplication([])
    app.setFont(QFont("Segoe UI", 9))
    app.setStyleSheet(theme.STYLESHEET)
    win = MainWindow("SPXW")
    win.resize(1500, 950)
    win.show()
    for _ in range(6):
        app.processEvents()

    # Drive a real backtest synchronously (no thread) and publish results
    dates = common_dates("SPXW")[-8:]
    print(f"backtest over {dates}")
    payloads = run_backtest_days(dates, lambda: ShortStraddle(), "SPXW", print)
    win.state.set_results("Short Straddle", payloads)
    win.backtest_tab.populate_days(payloads)
    # select a day that actually traded to exercise the signal chart
    traded = next((i for i, p in enumerate(payloads) if p["n_trades"] > 0), len(payloads) - 1)
    win.backtest_tab.day_table.selectRow(traded)
    for _ in range(6):
        app.processEvents()

    win.tabs.setCurrentIndex(0); grab(win, "tab1_backtest")
    win.tabs.setCurrentIndex(1); win.report_tab.refresh(); grab(win, "tab2_report")
    win.tabs.setCurrentIndex(2); grab(win, "tab3_mllab")
    win.tabs.setCurrentIndex(3); grab(win, "tab4_forecast")

    # also render an iron condor day to confirm multi-leg arrows
    payloads2 = run_backtest_days(dates, lambda: IronCondor(), "SPXW", print)
    win.state.set_results("Iron Condor", payloads2)
    win.backtest_tab.populate_days(payloads2)
    win.backtest_tab.day_table.selectRow(
        next((i for i, p in enumerate(payloads2) if p["n_trades"] > 0), 0))
    win.tabs.setCurrentIndex(0)
    grab(win, "tab1_backtest_condor")

    print("OK: all tabs rendered")


if __name__ == "__main__":
    main()
