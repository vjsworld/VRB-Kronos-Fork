"""Pre-build NPZ caches so later runs are instant.

Usage:
    python -m vrb.scripts.build_cache --last 60          # most recent 60 days
    python -m vrb.scripts.build_cache --all              # everything on disk
"""

from __future__ import annotations

import argparse
import time

from ..config import OPTION_ROOTS
from ..data import ib, theta
from ..data.calendar import common_dates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="SPXW")
    ap.add_argument("--last", type=int, default=0, help="only the N most recent days")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    cash, fut, _ = OPTION_ROOTS[args.root]
    dates = common_dates(args.root)
    if args.last and not args.all:
        dates = dates[-args.last:]
    print(f"caching {len(dates)} days for {args.root} (+{cash}, {fut}, VIX)")

    t0 = time.time()
    for i, d in enumerate(dates, 1):
        try:
            theta.load_chain_day(args.root, d)
            for sym in (cash, fut, "VIX"):
                ib.load_day(sym, d)
            print(f"  [{i}/{len(dates)}] {d} ok ({time.time() - t0:.0f}s elapsed)")
        except (FileNotFoundError, ValueError) as e:
            print(f"  [{i}/{len(dates)}] {d} skipped: {e.__class__.__name__}")


if __name__ == "__main__":
    main()
