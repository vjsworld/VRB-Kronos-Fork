"""Trading-day discovery from what's actually on disk.

A date counts as tradeable for a given study only if every required dataset
has a file for it; holidays and still-downloading days fall out naturally.
"""

from __future__ import annotations

from pathlib import Path

from ..config import IB_LAYOUT, IB_ROOT, OPTION_ROOTS, THETA_ROOT


def _dates_in(folder: Path) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.glob("*.parquet") if p.stem.isdigit() and len(p.stem) == 8}


def ib_dates(symbol: str) -> set[str]:
    sec_type, interval, _ = IB_LAYOUT[symbol]
    return _dates_in(IB_ROOT / symbol / sec_type / interval)


def option_dates(root: str) -> set[str]:
    return _dates_in(THETA_ROOT / root / "quotes")


def common_dates(option_root: str = "SPXW", extra_symbols: tuple[str, ...] = ("VIX",)) -> list[str]:
    """Sorted YYYYMMDD dates where option quotes AND all underlyings exist."""
    cash, fut, _ = OPTION_ROOTS[option_root]
    days = option_dates(option_root) & ib_dates(cash) & ib_dates(fut)
    for sym in extra_symbols:
        days &= ib_dates(sym)
    return sorted(days)
