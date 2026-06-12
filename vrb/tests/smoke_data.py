"""Smoke test: load real data, verify grid, IV, greeks, cache speed."""
import time

import numpy as np

from vrb.options.chain import ChainDay

t0 = time.time()
day = ChainDay.load("20260605")
t1 = time.time()
print(f"cold load (parquet pivot + cache write): {t1 - t0:.1f}s")
print("grid T x K x 2 =", day.bid.shape, "| ts", day.ts[0], "->", day.ts[-1])
print("strikes", day.strikes[0], "->", day.strikes[-1], "n =", len(day.strikes))
print(f"spot open/settlement: {day.spot[0]:.2f} / {day.settlement:.2f}")
print(f"vix at open: {day.vix[0]:.2f}")

t = day.t_index("10:30:00")
k = day.atm_k(t)
print(f"10:30 CT spot={day.spot[t]:.2f} atm_strike={day.strikes[k]} "
      f"call {day.bid[t, k, 0]}/{day.ask[t, k, 0]} put {day.bid[t, k, 1]}/{day.ask[t, k, 1]}")
print(f"atm_iv at 10:30 = {day.atm_iv(t):.4f}  tau = {day.tau[t] * 365 * 24:.2f} hrs left")
g = day.greeks_at(t)
print(f"atm call delta {g['delta'][k, 0]:.3f}, put delta {g['delta'][k, 1]:.3f}, "
      f"gamma {g['gamma'][k, 0]:.5f}")

iv = day.iv_at(t)
ok = np.isfinite(iv).mean()
print(f"iv_at(t): {ok:.1%} of {iv.size} contracts have finite IV")

t0 = time.time()
ChainDay.load("20260605")
t1 = time.time()
print(f"warm load (npz cache): {t1 - t0:.2f}s")
