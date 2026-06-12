"""Smoke test: SuperTrend signals + Last Hour Gamma Explosion backtest."""
import numpy as np

from vrb.backtest.engine import Backtest
from vrb.backtest.payload import day_payload
from vrb.backtest.strategies import LastHourGammaExplosion
from vrb.data.calendar import common_dates
from vrb.indicators.supertrend import day_signals
from vrb.options.chain import CALL, ChainDay

# 1) SuperTrend on ES for a recent day
date = common_dates("SPXW")[-1]
sig = day_signals(date, "ES", period=10, multiplier=3.0,
                  start="14:00:00", end="15:00:00")
print(f"{date}: {len(sig['ts'])} ES 1-min bars, "
      f"{int((np.diff(sig['direction']) != 0).sum())} total flips, "
      f"{len(sig['events'])} signals in last hour")
for ts, right in sig["events"]:
    print(f"  {ts}  {'CALL (long)' if right == CALL else 'PUT (short)'}")

# 2) Backtest over recent days
print("\nLast Hour Gamma Explosion, 20delta, 5x target, 14:00-15:00:")
dates = common_dates("SPXW")[-10:]
total = 0.0
for d in dates:
    try:
        day = ChainDay.load(d, "SPXW")
    except (FileNotFoundError, ValueError):
        print(f"  {d}: skipped (no 0DTE data)")
        continue
    eng = Backtest(day)
    res = eng.run(LastHourGammaExplosion())
    p = day_payload(day, res)
    total += p["pnl"]
    if p["trades"]:
        kinds = ",".join(f"{t['legs']}->{t['reason']}" for t in p["trades"])
        print(f"  {d}: ${p['pnl']:8.2f}  {p['n_trades']} trades: {kinds}")
    else:
        print(f"  {d}: ${p['pnl']:8.2f}  no signals")
print(f"total ${total:,.2f}")

# 3) verify payload tags for chart arrows — use a wide window for more trades
day = ChainDay.load(date, "SPXW")
res = Backtest(day).run(LastHourGammaExplosion(entry_time="09:00:00", exit_time="14:59:00"))
p = day_payload(day, res)
if p["trades"]:
    t0 = p["trades"][0]
    print(f"\nsample trade: dir={t0['direction']} txn={t0['transaction']} "
          f"entry='{t0['entry_text']}'")
    assert t0["transaction"] == "BUY"  # we always buy options here
    assert t0["direction"] in ("buy", "sell")
print("OK")
