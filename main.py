#!/usr/bin/env python3
"""VRB Trading Workbench launcher.

Run from the repo root:
    python main.py            # SPXW (default)
    python main.py NDX        # start on a different option root

This is a thin wrapper around `python -m vrb.gui`.
"""

import sys

# Lazy import inside the guard so multiprocessing workers (which re-import this
# module on Windows spawn) don't pull in PyQt6 — keeps the parallel backtest
# workers lightweight.
if __name__ == "__main__":
    from vrb.gui.__main__ import main
    sys.exit(main())

