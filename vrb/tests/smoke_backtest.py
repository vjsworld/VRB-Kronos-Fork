"""Smoke test: short straddle + iron condor over a handful of real days."""
from vrb.backtest import stats
from vrb.backtest.strategies import IronCondor, ShortStraddle
from vrb.data.calendar import common_dates

dates = common_dates("SPXW")[-5:]
print("dates:", dates)

print("\nShortStraddle 09:00->14:45, stop 2x, target 50%:")
res = stats.run_days(dates, lambda: ShortStraddle())
stats.print_summary("short straddle", stats.summarize(res))

print("\nIronCondor 16-delta, 25pt wings:")
res = stats.run_days(dates, lambda: IronCondor())
stats.print_summary("iron condor", stats.summarize(res))
