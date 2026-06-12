"""Dark theme: colors, Qt stylesheet, pyqtgraph defaults."""

from __future__ import annotations

import pyqtgraph as pg

# ---------------------------------------------------------------- palette
BG = "#0f1419"
BG_PANEL = "#161b22"
BG_ALT = "#1c2128"
FG = "#c9d1d9"
FG_DIM = "#8b949e"
ACCENT = "#58a6ff"
GRID = "#21262d"

UP = "#26a69a"        # bullish candle
DOWN = "#ef5350"      # bearish candle
BUY = "#2979ff"       # blue arrows = buys
SELL = "#ff1744"      # red arrows = sells
EXIT = "#ffffff"      # white arrows = exits
EQUITY = "#26a69a"
DRAWDOWN = "#ef5350"
FORECAST = "#00e5ff"
WIN = "#26a69a"
LOSS = "#ef5350"


def apply_pg_theme() -> None:
    pg.setConfigOptions(
        background=BG, foreground=FG, antialias=True,
        useOpenGL=False,  # CPU raster is more reliable across Windows drivers
    )


STYLESHEET = f"""
QMainWindow, QWidget {{ background: {BG}; color: {FG}; font-size: 12px; }}
QTabWidget::pane {{ border: 1px solid {GRID}; background: {BG}; }}
QTabBar::tab {{
    background: {BG_PANEL}; color: {FG_DIM}; padding: 8px 18px;
    border: 1px solid {GRID}; border-bottom: none; font-weight: 600;
}}
QTabBar::tab:selected {{ background: {BG}; color: {ACCENT}; }}
QGroupBox {{
    border: 1px solid {GRID}; border-radius: 4px; margin-top: 12px;
    padding-top: 8px; font-weight: 600; color: {FG_DIM};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QPushButton {{
    background: {BG_ALT}; border: 1px solid {GRID}; border-radius: 4px;
    padding: 6px 14px; color: {FG}; font-weight: 600;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:disabled {{ color: {GRID}; }}
QPushButton#primary {{ background: #1f4068; border-color: {ACCENT}; }}
QComboBox, QSpinBox, QDoubleSpinBox, QTimeEdit, QLineEdit {{
    background: {BG_ALT}; border: 1px solid {GRID}; border-radius: 3px;
    padding: 4px 6px; color: {FG};
}}
QTableWidget {{
    background: {BG_PANEL}; alternate-background-color: {BG_ALT};
    gridline-color: {GRID}; border: 1px solid {GRID};
    selection-background-color: #1f4068; selection-color: {FG};
}}
QHeaderView::section {{
    background: {BG_ALT}; color: {FG_DIM}; border: none;
    border-right: 1px solid {GRID}; border-bottom: 1px solid {GRID};
    padding: 5px; font-weight: 700;
}}
QLabel#title {{ font-size: 14px; font-weight: 700; color: {ACCENT}; }}
QLabel#status {{ color: {FG_DIM}; }}
QProgressBar {{
    background: {BG_ALT}; border: 1px solid {GRID}; border-radius: 3px;
    text-align: center; color: {FG};
}}
QProgressBar::chunk {{ background: #1f4068; }}
QSplitter::handle {{ background: {GRID}; }}
QScrollBar:vertical {{ background: {BG_PANEL}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {GRID}; border-radius: 4px; }}
"""
