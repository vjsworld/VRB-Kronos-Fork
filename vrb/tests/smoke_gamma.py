"""Smoke test: SuperTrend signals + Last Hour Gamma Explosion backtest.

Run: python -m vrb.tests.smoke_gamma
(Has a __main__ guard because the parallel runner spawns worker processes.)
"""
import numpy as np

from vrb.backtest.engine import Backtest
from vrb.backtest.parallel import run_days_parallel
from vrb.backtest.payload import day_payload
from vrb.backtest.strategies import LastHourGammaExplosion
from vrb.data.calendar import common_dates
from vrb.indicators.supertrend import day_signals
from vrb.options.chain import CALL, ChainDay
from vrb.util.timing import Timer, get_logger

log = get_logger("test.gamma")


def main() -> None:
    # 1) SuperTrend on ES for a recent day
    date = common_dates("SPXW")[-1]
    sig = day_signals(date, "ES", period=10, multiplier=3.0,
                      start="14:00:00", end="15:00:00")
    print(f"{date}: {len(sig['ts'])} ES 1-min bars, "
          f"{int((np.diff(sig['direction']) != 0).sum())} total flips, "
          f"{len(sig['events'])} signals in last hour")
    for ts, right in sig["events"]:
        print(f"  {ts}  {'CALL (long)' if right == CALL else 'PUT (short)'}")

    # 2) Backtest over recent days (parallel, timed)
    print("\nLast Hour Gamma Explosion, 20delta, 5x target, 14:00-15:00:")
    with Timer("parallel backtest 10 days", log):
        par = run_days_parallel(common_dates("SPXW")[-10:], LastHourGammaExplosion,
                                dict(target_mult=5.0), "SPXW", progress_cb=lambda m: None)
    print(f"  parallel total ${sum(p['pnl'] for p in par):,.2f} over {len(par)} days")

    # 3) verify payload tags for chart arrows — wide window for more trades
    day = ChainDay.load(date, "SPXW")
    res = Backtest(day).run(LastHourGammaExplosion(entry_time="09:00:00", exit_time="14:59:00"))
    p = day_payload(day, res)
    if p["trades"]:
        t0 = p["trades"][0]
        print(f"sample trade: dir={t0['direction']} txn={t0['transaction']} "
              f"entry='{t0['entry_text']}' legs_detail={t0.get('legs_detail')}")
        assert t0["transaction"] == "BUY"  # we always buy options here
        assert t0["direction"] in ("buy", "sell")
    print("OK")


if __name__ == "__main__":
    main()
