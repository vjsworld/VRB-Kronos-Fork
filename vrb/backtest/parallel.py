"""Parallel multi-day backtest runner.

Each trading day is an independent backtest, so they fan out across CPU cores
with a process pool. The first run over a large date range also warms the NPZ
cache in parallel (cold parquet pivots are the dominant cost); subsequent runs
are limited only by NPZ load + engine time.

Workers receive a picklable (strategy_cls, kwargs) spec and rebuild the
strategy fresh per day. Keep this module Qt-free and cheap to import — spawned
workers import it directly.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

from ..options.chain import ChainDay
from ..util.timing import get_logger
from .engine import Backtest
from .payload import day_payload

log = get_logger(__name__)


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _run_one(args: tuple):
    date, strategy_cls, kwargs, root, engine_kwargs = args
    try:
        day = ChainDay.load(date, root)
    except (FileNotFoundError, ValueError) as e:
        return (date, None, type(e).__name__)
    t0 = time.perf_counter()
    res = Backtest(day, **(engine_kwargs or {})).run(strategy_cls(**kwargs))
    payload = day_payload(day, res)
    payload["_elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return (date, payload, None)


def run_days_parallel(
    dates: list[str],
    strategy_cls,
    strategy_kwargs: dict,
    root: str = "SPXW",
    n_workers: int | None = None,
    engine_kwargs: dict | None = None,
    progress_cb=lambda msg: None,
    chunksize: int = 4,
) -> list[dict]:
    """Backtest each date in its own process; return day payloads in date order.

    Falls back to serial execution for tiny date ranges (process spin-up isn't
    worth it under ~4 days).
    """
    if not dates:
        return []
    n_workers = n_workers or default_workers()
    args = [(d, strategy_cls, strategy_kwargs, root, engine_kwargs) for d in dates]
    payloads: list[dict] = []
    skipped = 0
    t0 = time.perf_counter()

    if len(dates) <= 3 or n_workers == 1:
        log.info("backtest %d days serially (%s)", len(dates), strategy_cls.__name__)
        results = map(_run_one, args)
        payloads, skipped = _collect(results, len(dates), progress_cb)
    else:
        log.info("backtest %d days on %d workers (%s)", len(dates), n_workers,
                 strategy_cls.__name__)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = ex.map(_run_one, args, chunksize=chunksize)
            payloads, skipped = _collect(results, len(dates), progress_cb)

    elapsed = time.perf_counter() - t0
    if payloads:
        avg_engine = sum(p.get("_elapsed_ms", 0) for p in payloads) / len(payloads)
    else:
        avg_engine = 0.0
    log.info("done: %d ran, %d skipped in %.1fs (%.0fms/day wall, %.0fms/day engine)",
             len(payloads), skipped, elapsed,
             elapsed / len(dates) * 1000, avg_engine)
    progress_cb(f"Done: {len(payloads)} days, {skipped} skipped in {elapsed:.1f}s "
                f"({elapsed / len(dates) * 1000:.0f}ms/day)")
    return payloads


def _collect(results, total: int, progress_cb):
    payloads, skipped = [], 0
    for i, (date, payload, err) in enumerate(results, 1):
        if payload is None:
            skipped += 1
            progress_cb(f"[{i}/{total}] {date} skipped ({err})")
        else:
            payloads.append(payload)
            if i % 10 == 0 or i == total:
                progress_cb(f"[{i}/{total}] {date}  ${payload['pnl']:,.2f}")
    return payloads, skipped
