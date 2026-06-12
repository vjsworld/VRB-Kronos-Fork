"""Measure whether Kronos has tradeable edge for the theta strategy.

Two questions, answered with data, not hope:
  1. Direction — does Kronos's forecast mean return predict the actual forward
     ES return over the horizon? (rank IC)
  2. Volatility — does Kronos's forecast path vol predict realized forward vol,
     and how do both compare to the option's implied vol? (the vol-risk-premium
     that premium selling harvests)

Run: python -m vrb.research.kronos_study
"""

from __future__ import annotations

import numpy as np

from ..data import ib
from ..data.calendar import common_dates
from ..ml.forecaster import _spearman
from ..ml.kronos_forecaster import KronosForecaster
from ..options.chain import ChainDay
from ..util.timing import Timer, get_logger

log = get_logger("research.kronos")
HORIZON_MIN = 30
SAMPLE_TIMES = [f"{h:02d}:{m:02d}:00" for h in range(10, 15) for m in (0, 30) if not (h == 14 and m == 30)]
ANN = np.sqrt(365 * 24 * 60)  # calendar-minute annualization (matches kronos + IV)


def main(n_days: int = 40) -> None:
    dates = common_dates("SPXW")[-n_days:]
    kf = KronosForecaster(pred_len=HORIZON_MIN)
    rows = []  # (exp_ret, path_vol, fwd_ret, realized_vol, atm_iv)
    with Timer(f"Kronos study over {len(dates)} days", log):
        for d in dates:
            try:
                day = ChainDay.load(d, "SPXW")
                bars = ib.resample_1min(ib.load_day("ES", d))
            except (FileNotFoundError, ValueError):
                continue
            bts = bars["timestamps"].to_numpy().astype("datetime64[s]")
            bclose = bars["close"].to_numpy(float)
            t_idx = np.array([day.t_index(h) for h in SAMPLE_TIMES])
            feats = kf.day_features(d, day.ts, t_idx)  # (n,2): exp_ret, path_vol
            for i, h in enumerate(SAMPLE_TIMES):
                t0 = day.ts[t_idx[i]]
                tH = t0 + np.timedelta64(HORIZON_MIN, "m")
                i0 = int(np.searchsorted(bts, t0, "right")) - 1
                iH = int(np.searchsorted(bts, tH, "right")) - 1
                if i0 < 0 or iH <= i0:
                    continue
                fwd_ret = float(np.log(bclose[iH] / bclose[i0]))
                seg = bclose[i0:iH + 1]
                rvol = float(np.std(np.diff(np.log(seg))) * ANN)
                atm_iv = day.atm_iv(t_idx[i])
                if np.isfinite(feats[i, 0]) and np.isfinite(fwd_ret):
                    rows.append((feats[i, 0], feats[i, 1], fwd_ret, rvol, atm_iv))

    a = np.array(rows, float)
    exp_ret, path_vol, fwd_ret, rvol, iv = a.T
    fin = np.isfinite(a).all(axis=1)
    a = a[fin]; exp_ret, path_vol, fwd_ret, rvol, iv = a.T

    print(f"\n=== Kronos edge study: {len(a)} forecasts over {len(dates)} days, {HORIZON_MIN}-min horizon ===\n")
    print("DIRECTION (can Kronos pick which way?):")
    ic = _spearman(exp_ret, fwd_ret)
    hit = float((np.sign(exp_ret) == np.sign(fwd_ret)).mean())
    conv = np.abs(exp_ret) > np.quantile(np.abs(exp_ret), 0.7)
    hit_conv = float((np.sign(exp_ret[conv]) == np.sign(fwd_ret[conv])).mean())
    print(f"  rank IC(exp_ret, fwd_ret) = {ic:+.4f}   (0 = no skill, >0.03 = usable)")
    print(f"  hit rate all = {hit*100:.1f}%   top-30% conviction = {hit_conv*100:.1f}%")
    print()
    print("VOLATILITY (can Kronos forecast realized vol? is premium rich?):")
    print(f"  corr(path_vol, realized_vol) = {_spearman(path_vol, rvol):+.4f}")
    print(f"  corr(IV, realized_vol)       = {_spearman(iv, rvol):+.4f}   (does the market already know?)")
    print(f"  median IV / realized = {np.median(iv / np.maximum(rvol,1e-6)):.2f}   "
          f"(>1 = options richer than moves -> seller edge)")
    print(f"  median Kronos pathvol / realized = {np.median(path_vol/np.maximum(rvol,1e-6)):.2f}")
    print(f"  median IV={np.median(iv):.3f}  Kronos_pathvol={np.median(path_vol):.3f}  realized={np.median(rvol):.3f}")
    print()
    # Does selling only when IV > Kronos-forecast vol improve realized outcomes?
    rich = iv > path_vol
    print("VOL-PREMIUM FILTER (sell only when IV > Kronos forecast vol):")
    print(f"  {rich.mean()*100:.0f}% of samples flagged 'rich'.  "
          f"mean realized vol when rich={rvol[rich].mean():.3f} vs not-rich={rvol[~rich].mean():.3f}")


if __name__ == "__main__":
    main()
