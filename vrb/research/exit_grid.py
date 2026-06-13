"""Exit-mechanics grid: profit targets x stop losses x scaling-in, both sides.

Answers two questions the journal couldn't: (1) does ANY profit-target/stop
combination rescue the premium buyer, (2) which exit pair is best for the
premium seller. Buyer grid runs at MID fills (most favorable — if it fails
there, it fails everywhere). All runs are journaled.

Run: python -m vrb.research.exit_grid
"""

from __future__ import annotations

import numpy as np

from ..backtest.parallel import run_days_parallel
from ..backtest.strategies import CompressionBuyer, ShortStrangle
from ..data.calendar import common_dates
from ..research import registry
from ..util.timing import Timer, get_logger

log = get_logger("research.exitgrid")


def run_one(cls, label, fill, dates, **kw):
    pls = run_days_parallel(dates, cls, kw, "SPXW",
                            engine_kwargs={"improvement": fill},
                            progress_cb=lambda m: None)
    rec = registry.record(cls.__name__, {**kw, "fill_quality": fill}, "SPXW",
                          [p["date"] for p in pls], pls, notes=label)
    s = rec["stats"]
    print(f"{label:36s} net=${s['total_pnl']:>9,.0f}  gross@mid=${s['gross_mid_pnl']:>8,.0f}  "
          f"trades={s['n_trades']:>4d}  twin%={s['trade_win_rate']*100:>3.0f}  "
          f"PF={s['profit_factor']:.2f}  maxDD=${s['max_drawdown']:>8,.0f}")
    return s


def main() -> None:
    dates = common_dates("SPXW")

    print("=== BUYER exit grid (CompressionBuyer puts, mid fills, hold<=120m) ===")
    print("    target x stop x scaling — gross@mid is the verdict column")
    with Timer("buyer grid", log):
        for target in (3.0, 5.0, 10.0, 20.0):
            for stop in (0.0, 0.5):
                for adds in (0, 2):
                    label = f"buy t{target:g}x stop{stop:g} adds{adds}"
                    run_one(CompressionBuyer, label, 1.0, dates,
                            side="put", target_delta=0.06, target_mult=target,
                            stop_frac=stop, scale_adds=adds, scale_trigger=0.5,
                            min_iv_rv=1.05, max_rv=0.30, hold_min=120)

    print("\n=== SELLER exit grid (ShortStrangle 16d 10:00, fill 0.75) ===")
    with Timer("seller grid", log):
        for pf in (0.25, 0.5, 0.75, None):
            for stop in (1.5, 2.0, 3.0, 100.0):
                label = f"sell pf{pf} stop{stop:g}"
                run_one(ShortStrangle, label, 0.75, dates,
                        entry_time="10:00:00", exit_time="15:00:00",
                        target_delta=0.16, profit_frac=pf, stop_mult=stop)


if __name__ == "__main__":
    main()
