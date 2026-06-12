"""Run a strategy backtest over a date range.

Usage:
    python -m vrb.scripts.run_backtest --strategy straddle --last 30
    python -m vrb.scripts.run_backtest --strategy condor --last 60 --delta 0.10 --wing 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..backtest import stats
from ..backtest.strategies import IronCondor, ShortStraddle
from ..data.calendar import common_dates

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "vrb_out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["straddle", "condor"], default="straddle")
    ap.add_argument("--root", default="SPXW")
    ap.add_argument("--last", type=int, default=30)
    ap.add_argument("--entry", default="09:00:00")
    ap.add_argument("--exit", default="14:45:00")
    ap.add_argument("--stop", type=float, default=2.0)
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.16, help="condor short delta")
    ap.add_argument("--wing", type=float, default=25.0, help="condor wing width, pts")
    args = ap.parse_args()

    dates = common_dates(args.root)[-args.last:]
    print(f"{args.strategy} over {len(dates)} days ({dates[0]}..{dates[-1]})")

    if args.strategy == "straddle":
        factory = lambda: ShortStraddle(args.entry, args.exit, args.stop, args.target)
    else:
        factory = lambda: IronCondor(args.entry, args.exit, args.delta,
                                     args.wing, args.stop, args.target)

    results = stats.run_days(dates, factory, root=args.root)
    stats.print_summary(args.strategy, stats.summarize(results))

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"equity_{args.strategy}.csv"
    with open(out, "w") as f:
        f.write("date,pnl,cumulative\n")
        cum = 0.0
        for r in results:
            cum += r.pnl
            f.write(f"{r.date},{r.pnl:.2f},{cum:.2f}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
