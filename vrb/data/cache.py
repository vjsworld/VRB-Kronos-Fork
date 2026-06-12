"""NPZ cache layer.

Parquet is the system of record (updated daily by the downloaders); NPZ files
are derived caches that load 10-50x faster because they skip parsing/pivoting.
A cache entry is keyed to its source parquet's (mtime_ns, size) so a re-download
of the same day automatically invalidates the cache.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import CACHE_ROOT


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
            with np.load(cache_path, allow_pickle=False) as z:
                if "_fingerprint" in z and np.array_equal(z["_fingerprint"], fingerprint):
                    return {k: z[k] for k in z.files if k != "_fingerprint"}
        except Exception:
            pass  # corrupt/partial cache (BadZipFile, zlib.error, ...): rebuild below

    arrays = builder()
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
