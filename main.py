#!/usr/bin/env python3
"""VRB Trading Workbench launcher.

Run from the repo root:
    python main.py            # SPXW (default)
    python main.py NDX        # start on a different option root

This is a thin wrapper around `python -m vrb.gui`.
"""

import sys

from vrb.gui.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
