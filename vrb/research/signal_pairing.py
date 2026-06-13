"""Pair the Gamma Explosion signal with the 10x explosion database.

Question 1 (lead-lag): does a SuperTrend flip LEAD premium explosions? For
every flip we compute the same forward label as the precursor study — did any
eligible cheap OTM option of the flip's side achieve `ratio` within the
horizon — and compare the flip-conditioned rate against the unconditional
base rate sampled across all minutes. Lift > 1 means the flip is genuinely
positioned BEFORE explosions (not after, where the gamma buyer died).

Question 2 is built on top in strategies.GammaExplosionGated: only take flips
when the precursor-model score is in the top decile (model trained on the
first 70% of days; backtests run on the rest — honest OOS).

Run: python -m vrb.research.signal_pairing
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from ..data.calendar import common_dates
from ..indicators.supertrend import day_signals
from ..options.chain import CALL, PUT, ChainDay
from ..util.timing import Timer, get_logger
from .expansion import day_context
from .precursors import _label

log = get_logger("research.pairing")
GRID = 5
HORIZONS_MIN = (15, 30, 60)
RATIO = 10.0
MIN_MID, MAX_MID = 0.15, 3.0


def day_rows(date: str, root="SPXW", atr_period=10, atr_mult=3.0,
             base_every_min=10) -> dict:
    """Per-day: flip-conditioned labels + base-rate labels for both sides."""
    try:
        day = ChainDay.load(date, root)
        sig = day_signals(date, "ES", atr_period, atr_mult, "08:35:00", "14:00:00")
    except (FileNotFoundError, ValueError):
        return {"flips": [], "base": []}
    ctx = day_context(day)
    t0, t1 = day.t_index("08:35:00"), day.t_index("14:00:00")

    def labels(t, side):
        return [_label(day, ctx, t, side, h * 60 // GRID, MIN_MID, MAX_MID, RATIO)
                for h in HORIZONS_MIN]

    flips = []
    for ts, right in sig["events"]:
        t = int(np.searchsorted(day.ts, ts))
        if t0 <= t < t1 and np.isfinite(ctx["spot"][t]):
            flips.append({"side": right, "labels": labels(t, right)})

    base = []
    for t in range(t0, t1, base_every_min * 60 // GRID):
        if not np.isfinite(ctx["spot"][t]):
            continue
        for side in (CALL, PUT):
            base.append({"side": side, "labels": labels(t, side)})
    return {"flips": flips, "base": base}


def main() -> None:
    dates = common_dates("SPXW")
    fn = partial(day_rows)
    flips, base = [], []
    with Timer(f"pairing study {len(dates)} days", log):
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
            for r in ex.map(fn, dates, chunksize=4):
                flips.extend(r["flips"]); base.extend(r["base"])

    print(f"\n=== SuperTrend flip -> 10x explosion lead-lag ===")
    print(f"flips: {len(flips)} | base samples: {len(base)}\n")
    for side, name in ((CALL, "BULLISH flip -> CALL explosion"),
                       (PUT, "BEARISH flip -> PUT explosion")):
        f = np.array([x["labels"] for x in flips if x["side"] == side], float)
        b = np.array([x["labels"] for x in base if x["side"] == side], float)
        print(name + f"  (n_flips={len(f)})")
        for i, h in enumerate(HORIZONS_MIN):
            fr = f[:, i].mean() if len(f) else np.nan
            br = b[:, i].mean() if len(b) else np.nan
            lift = fr / br if br and np.isfinite(br) and br > 0 else np.nan
            print(f"  within {h:3d}m: flip-rate {fr*100:5.2f}%  base {br*100:5.2f}%  LIFT {lift:.2f}x")
        print()


if __name__ == "__main__":
    main()
