"""NPZ cache layer.

Parquet is the system of record (updated daily by the downloaders); NPZ files
are derived caches that load 10-50x faster because they skip parsing/pivoting.
A cache entry is keyed to its source parquet's (mtime_ns, size) so a re-download
of the same day automatically invalidates the cache.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import CACHE_ROOT
from ..util.timing import get_logger

log = get_logger(__name__)


def _source_fingerprint(source: Path) -> np.ndarray:
    st = source.stat()
    return np.array([st.st_mtime_ns, st.st_size], dtype=np.int64)


def load_or_build(
    cache_rel: str,
    source: Path,
    builder: Callable[[], dict[str, np.ndarray]],
    compressed: bool = True,
) -> dict[str, np.ndarray]:
    """Return builder() output, transparently cached as an NPZ file.

    cache_rel: cache file path relative to CACHE_ROOT, e.g. "SPXW/20260605.npz"
    source:    the parquet file the cache is derived from
    builder:   produces a dict of numpy arrays when the cache is cold/stale
    """
    cache_path = CACHE_ROOT / cache_rel
    fingerprint = _source_fingerprint(source)

    if cache_path.exists():
        try:
            t0 = time.perf_counter()
            with np.load(cache_path, allow_pickle=False) as z:
                if "_fingerprint" in z and np.array_equal(z["_fingerprint"], fingerprint):
                    out = {k: z[k] for k in z.files if k != "_fingerprint"}
                    log.debug("cache HIT  %s (%.0fms)", cache_rel, (time.perf_counter() - t0) * 1000)
                    return out
        except Exception:
            pass  # corrupt/partial cache (BadZipFile, zlib.error, ...): rebuild below

    t0 = time.perf_counter()
    arrays = builder()
    # cold build is the slow path (parquet parse/pivot) — log at INFO so a slow
    # run visibly attributes its time to cache misses rather than looking hung
    log.info("cache MISS %s — built in %.0fms", cache_rel, (time.perf_counter() - t0) * 1000)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compressed else np.savez
    # unique tmp per process: concurrent builders must not share a tmp file
    tmp = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}-{uuid.uuid4().hex}.tmp.npz")
    save(tmp, _fingerprint=fingerprint, **arrays)
    try:
        tmp.replace(cache_path)
    except PermissionError:
        # Windows: destination open in another process; their copy is valid
        tmp.unlink(missing_ok=True)
    return arrays
