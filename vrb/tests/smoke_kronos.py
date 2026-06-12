"""Smoke test: Kronos forecast features on one real day (GPU)."""
import time

import numpy as np

from vrb.ml.kronos_forecaster import KronosForecaster
from vrb.options.chain import ChainDay

day = ChainDay.load("20260605")
t0 = time.time()
kf = KronosForecaster()
print(f"model load: {time.time() - t0:.1f}s")

# every 30 min, 09:00 -> 14:30
t_indices = np.array([day.t_index(f"{h:02d}:{m:02d}:00")
                      for h in range(9, 15) for m in (0, 30)])
t0 = time.time()
feats = kf.day_features("20260605", day.ts, t_indices)
print(f"12 forecasts (batched): {time.time() - t0:.1f}s")
for t, (er, pv) in zip(t_indices, feats):
    print(f"  {day.ts[t]}  exp_ret={er: .5f}  path_vol={pv:.3f}")
print("finite:", np.isfinite(feats).all(axis=0))
