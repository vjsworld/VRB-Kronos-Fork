"""Entry point: python -m vrb.gui"""

from __future__ import annotations

import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from . import theme
from .main_window import MainWindow


def main() -> int:
    theme.apply_pg_theme()
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    app.setStyleSheet(theme.STYLESHEET)
    root = sys.argv[1] if len(sys.argv) > 1 else "SPXW"
    win = MainWindow(root)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
