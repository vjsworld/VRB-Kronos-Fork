"""Logging + timing helpers used across the package.

Every test, backtest, and data load logs through here so performance is always
visible. Logs go to the console (stderr) and to a rotating file at
vrb_out/vrb.log. Use:

    from vrb.util.timing import get_logger, Timer
    log = get_logger(__name__)
    with Timer("load chain", log) as t:
        ...
    # t.ms holds the elapsed milliseconds afterwards

Set VRB_LOG_LEVEL=DEBUG for per-day cache detail; default INFO.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
LOG_PATH = Path(__file__).resolve().parent.parent.parent / "vrb_out" / "vrb.log"


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = getattr(logging, os.environ.get("VRB_LOG_LEVEL", "INFO").upper(), logging.INFO)
    root = logging.getLogger("vrb")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s | %(message)s",
                            datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Only the main process owns the rotating log file; worker processes log to
    # their own stderr to avoid multi-process file/rotation races.
    if multiprocessing.current_process().name == "MainProcess":
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            fileh = RotatingFileHandler(LOG_PATH, maxBytes=4_000_000, backupCount=3,
                                        encoding="utf-8", delay=True)
            fileh.setLevel(logging.DEBUG)  # file always keeps full detail
            fileh.setFormatter(fmt)
            root.addHandler(fileh)
        except OSError:
            pass  # read-only fs / locked file: console logging still works
    _CONFIGURED = True


def get_logger(name: str = "vrb") -> logging.Logger:
    _configure()
    if not name.startswith("vrb"):
        name = f"vrb.{name.split('.')[-1]}"
    return logging.getLogger(name)


class Timer:
    """Context manager that logs and records elapsed wall-clock time.

    The elapsed time is available as `.seconds` and `.ms` after the block.
    Pass log=None to time silently (just record), or a logger to also log.
    """

    def __init__(self, label: str, log: logging.Logger | None = None,
                 level: int = logging.INFO):
        self.label = label
        self.log = log
        self.level = level
        self.seconds = 0.0

    @property
    def ms(self) -> float:
        return self.seconds * 1000.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        if self.log is not None:
            self.log.log(self.level, f"{self.label}: {self.ms:.0f}ms")
