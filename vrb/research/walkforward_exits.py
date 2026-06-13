"""Walk-forward validation of the strangle EXIT grid (profit-take x stop).

Same protocol as walkforward.py but the selection space is the 16 exit cells
(entry fixed 10:00, 16-delta naked, fill 0.75): pick the best cell on each
rolling 120-day train window, trade it on the next untouched 40-day window.
Also scores the economically-motivated fixed cell (hold + 3x stop) across the
same OOS days for comparison — with the caveat that it was first seen in-sample.

Run: python -m vrb.research.walkforward_exits
"""

from __future__ import annotations

import numpy as np

from ..backtest.parallel import run_days_parallel
from ..backtest.strategies import ShortStrangle
from ..data.calendar import common_dates
from ..util.timing import Timer, get_logger
from .walkforward import TEST_DAYS, TRAIN_DAYS, stats, walk_forward

log = get_logger("research.wfo_exits")
FILL = 0.75
PFS = [0.25, 0.5, 0.75, None]
STOPS = [1.5, 2.0, 3.0, 100.0]
WINGS = [0.0, 100.0]   # 0 = naked (undefined risk), 100 = far protective wings


def main() -> None:
    dates = common_dates("SPXW")
    per_cfg: dict[str, dict] = {}
    n = len(PFS) * len(STOPS) * len(WINGS)
    with Timer(f"precompute {n} exit cells x {len(dates)} days", log):
        for pf in PFS:
            for stop in STOPS:
                for wing in WINGS:
                    kw = dict(entry_time="10:00:00", exit_time="15:00:00",
                              target_delta=0.16, profit_frac=pf, stop_mult=stop,
                              wing_pts=wing)
                    pls = run_days_parallel(dates, ShortStrangle, kw, "SPXW",
                                            engine_kwargs={"improvement": FILL},
                                            progress_cb=lambda m: None)
                    lbl = f"pf{pf}_stop{stop:g}_w{int(wing)}"
                    per_cfg[lbl] = {p["date"]: p["pnl"] for p in pls}
                    log.info("  %s net $%.0f", lbl, sum(per_cfg[lbl].values()))

    print(f"\n=== WALK-FORWARD: strangle exit cells (train {TRAIN_DAYS}d -> test {TEST_DAYS}d, fill {FILL}) ===\n")
    for kind in ("pnl", "sharpe"):
        od, op, folds = walk_forward(per_cfg, dates, kind)
        s = stats(op)
        pos = sum(1 for f in folds if f[5] > 0)
        print(f"WFO select-by-{kind}: OOS net=${s['net']:,.0f}  sharpe={s['sharpe']:.2f}  "
              f"daywin={s['daywin']*100:.0f}%  maxDD=${s['maxdd']:,.0f}  folds+ {pos}/{len(folds)}")
        for tr0, tr1, te0, te1, lbl, pnl in folds:
            print(f"    {te0}-{te1}: chose {lbl:18s} OOS ${pnl:>8,.0f}")
        print()

    # naked vs winged: does tail control keep the OOS edge with smaller DD?
    od, _, _ = walk_forward(per_cfg, dates, "sharpe")
    print("Fixed-cell reference on the sharpe-WFO OOS days (naked vs winged):")
    for lbl in ("pfNone_stop3_w0", "pfNone_stop3_w100", "pf0.25_stop100_w0", "pf0.25_stop100_w100"):
        v = np.array([per_cfg[lbl].get(d, 0.0) for d in od], float)
        s = stats(v)
        print(f"  {lbl:22s} net=${s['net']:>8,.0f}  sharpe={s['sharpe']:.2f}  maxDD=${s['maxdd']:>8,.0f}")
    print("(in-sample-identified cells; reference only)")


if __name__ == "__main__":
    main()
