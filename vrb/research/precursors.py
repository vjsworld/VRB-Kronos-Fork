"""Exhaustive precursor study: predict a forward premium explosion from market
state alone, point-in-time (no contract lookahead).

At each sampled minute we compute market-state features (vol compression, range
position, IV/RV, VIX dynamics, skew, overnight gap) and a forward label: did
ANY eligible cheap OTM option of `side` achieve the ratio within the horizon.
Then: univariate AUC, a multivariate gradient-boosting model scored
out-of-sample by day split (the honest predictive power), permutation feature
importance, and conjunction-flag lift tables. All thresholds are parameters.

Run: python -m vrb.research.precursors --side put --horizon 60 --ratio 10
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.calendar import common_dates
from ..options import bs
from ..options.chain import CALL, PUT, ChainDay
from ..util.timing import Timer, get_logger
from .expansion import day_context

log = get_logger("research.precursors")
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "vrb_out" / "research"
GRID = 5

MKT_FEATURES = [
    "tod_min", "tte_min", "ret_5m", "ret_15m", "ret_30m",
    "rv_5m", "rv_15m", "rv_30m", "rv_ratio_5_30",
    "range_pos", "ret_from_open", "vix", "vix_chg_15m",
    "atm_iv", "iv_rv_15m", "skew_pc", "es_gap_pct", "atm_spread_pct",
]


def market_snapshot(day: ChainDay, ctx: dict, t: int) -> dict:
    s = ctx["spot"][t]
    iv = day.atm_iv(t)
    rv5, rv15, rv30 = ctx["rv_5m"][t], ctx["rv_15m"][t], ctx["rv_30m"][t]
    rng = ctx["run_max"][t] - ctx["run_min"][t]
    k_atm = int(np.argmin(np.abs(day.strikes - s))) if np.isfinite(s) else 0
    ab, bb = float(day.ask[t, k_atm, CALL]), float(day.bid[t, k_atm, CALL])
    m = 0.5 * (ab + bb)
    kp = int(np.argmin(np.abs(day.strikes - s * 0.99))) if np.isfinite(s) else 0
    kc = int(np.argmin(np.abs(day.strikes - s * 1.01))) if np.isfinite(s) else 0
    ivp = bs.implied_vol(day.mid[t, kp, PUT], s, day.strikes[kp], day.tau[t], PUT, r=0.04)
    ivc = bs.implied_vol(day.mid[t, kc, CALL], s, day.strikes[kc], day.tau[t], CALL, r=0.04)
    s0 = ctx["spot"][ctx["open_t"]]
    return {
        "tod_min": (t - ctx["open_t"]) * GRID / 60.0,
        "tte_min": float(day.tau[t]) * 365 * 24 * 60,
        "ret_5m": float(ctx["ret_5m"][t]), "ret_15m": float(ctx["ret_15m"][t]),
        "ret_30m": float(ctx["ret_30m"][t]),
        "rv_5m": float(rv5), "rv_15m": float(rv15), "rv_30m": float(rv30),
        "rv_ratio_5_30": float(rv5 / rv30) if np.isfinite(rv5) and np.isfinite(rv30) and rv30 > 0 else np.nan,
        "range_pos": float((s - ctx["run_min"][t]) / rng) if rng > 0 else 0.5,
        "ret_from_open": float(np.log(s / s0)) if np.isfinite(s0) and s0 > 0 and np.isfinite(s) else np.nan,
        "vix": float(day.vix[t]), "vix_chg_15m": float(day.vix[t] - day.vix[max(0, t - 180)]),
        "atm_iv": float(iv), "iv_rv_15m": float(iv / rv15) if np.isfinite(iv) and np.isfinite(rv15) and rv15 > 0 else np.nan,
        "skew_pc": float(ivp - ivc) if np.isfinite(ivp) and np.isfinite(ivc) else np.nan,
        "es_gap_pct": ctx["es_gap"] * 100 if np.isfinite(ctx["es_gap"]) else np.nan,
        "atm_spread_pct": float((ab - bb) / m) if m > 0 else np.nan,
    }


def _label(day, ctx, t, side, horizon_steps, min_mid, max_mid, ratio) -> int:
    s = ctx["spot"][t]
    end = min(len(day.ts), t + 1 + horizon_steps)
    fwd = day.mid[t + 1:end, :, side]
    if fwd.shape[0] == 0 or not np.isfinite(s):
        return 0
    fmax = np.nanmax(np.where(np.isfinite(fwd), fwd, np.nan), axis=0)
    m0 = day.mid[t, :, side]
    otm = (day.strikes < s) if side == PUT else (day.strikes > s)
    elig = otm & np.isfinite(m0) & (m0 >= min_mid) & (m0 <= max_mid) & (day.bid[t, :, side] > 0)
    if not elig.any():
        return 0
    rr = np.where(elig & (m0 > 0), fmax / np.where(m0 > 0, m0, 1), 0.0)
    return int(np.nanmax(rr) >= ratio)


def study_day(date, root, side, horizon_steps, every_steps, min_mid, max_mid,
              ratio, start="08:35:00", end="14:30:00") -> list[dict]:
    try:
        day = ChainDay.load(date, root)
    except (FileNotFoundError, ValueError):
        return []
    ctx = day_context(day)
    t0, t1 = day.t_index(start), day.t_index(end)
    rows = []
    for t in range(t0, t1, every_steps):
        if not np.isfinite(ctx["spot"][t]):
            continue
        feat = market_snapshot(day, ctx, t)
        feat["label"] = _label(day, ctx, t, side, horizon_steps, min_mid, max_mid, ratio)
        feat["date"] = date
        rows.append(feat)
    return rows


def _auc(x, y):
    x = np.asarray(x, float); m = np.isfinite(x)
    x, y = x[m], y[m]
    if y.sum() < 5 or (~y.astype(bool)).sum() < 5:
        return np.nan
    from scipy.stats import mannwhitneyu
    u, _ = mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")
    return float(u / ((y == 1).sum() * (y == 0).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["put", "call"], default="put")
    ap.add_argument("--horizon", type=int, default=60, help="minutes")
    ap.add_argument("--every", type=int, default=5, help="sample minutes")
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--min-mid", type=float, default=0.15)
    ap.add_argument("--max-mid", type=float, default=3.0)
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--root", default="SPXW")
    args = ap.parse_args()
    side = PUT if args.side == "put" else CALL
    dates = common_dates(args.root)
    if args.days:
        dates = dates[-args.days:]

    fn = partial(study_day, root=args.root, side=side,
                 horizon_steps=args.horizon * 60 // GRID, every_steps=args.every * 60 // GRID,
                 min_mid=args.min_mid, max_mid=args.max_mid, ratio=args.ratio)
    rows = []
    with Timer(f"precursor sampling {len(dates)} days", log):
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
            for r in ex.map(fn, dates, chunksize=4):
                rows.extend(r)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"precursors_{args.side}_{args.ratio:g}x_{args.horizon}m.csv", index=False)
    y = df["label"].to_numpy(int)
    base = y.mean()
    print(f"\n=== Precursor study: {args.side}s, {args.ratio:g}x within {args.horizon}m ===")
    print(f"samples {len(df)} | positive rate (base) {base*100:.2f}% | days {df.date.nunique()}\n")

    print("UNIVARIATE AUC (forward-expansion predictiveness):")
    aucs = sorted(((f, _auc(df[f], y)) for f in MKT_FEATURES),
                  key=lambda kv: -abs((kv[1] or 0.5) - 0.5))
    for f, a in aucs:
        print(f"  {f:16s} AUC {a:.3f}" if np.isfinite(a) else f"  {f:16s} n/a")

    # multivariate GBM, day-split OOS
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    days = sorted(df.date.unique())
    cut = days[int(len(days) * 0.7)]
    tr, te = df.date <= cut, df.date > cut
    X = df[MKT_FEATURES].to_numpy(float)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, random_state=7)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])[:, 1]
    print(f"\nMULTIVARIATE GBM (day-split OOS): test AUC {_auc(proba, y[te]):.3f}  "
          f"(train {tr.sum()}, test {te.sum()})")
    # decile lift on OOS
    q = pd.qcut(proba, 10, labels=False, duplicates="drop")
    yt = y[te]
    print("  OOS decile lift (model score -> event rate):")
    for d in sorted(np.unique(q)):
        mask = q == d
        print(f"    decile {d}: rate {yt[mask].mean()*100:5.2f}%  (lift {yt[mask].mean()/base:.2f}x, n={mask.sum()})")
    imp = permutation_importance(clf, X[te], yt, n_repeats=5, random_state=7,
                                 scoring="roc_auc")
    print("  permutation importance (OOS):")
    for i in np.argsort(imp.importances_mean)[::-1][:8]:
        print(f"    {MKT_FEATURES[i]:16s} {imp.importances_mean[i]:+.4f}")

    # conjunction flag lift
    print("\nCONJUNCTION FLAGS (event rate vs base):")
    rv_lo = df.rv_15m < df.rv_15m.quantile(0.33)
    ivrv_hi = df.iv_rv_15m > 1.10
    rng_hi = df.range_pos > 0.6
    vix_dn = df.vix_chg_15m < 0
    flags = {"low RV": rv_lo, "IV/RV>1.1": ivrv_hi, "range top": rng_hi, "VIX falling": vix_dn}
    for name, fl in flags.items():
        print(f"  {name:16s} rate {y[fl].mean()*100:5.2f}%  lift {y[fl].mean()/base:.2f}x  (n={fl.sum()})")
    allf = rv_lo & ivrv_hi & rng_hi & vix_dn
    print(f"  ALL FOUR         rate {y[allf].mean()*100:5.2f}%  lift {y[allf].mean()/base:.2f}x  (n={allf.sum()})")
    print(f"\ncsv -> {OUT_DIR / f'precursors_{args.side}_{args.ratio:g}x_{args.horizon}m.csv'}")


if __name__ == "__main__":
    main()
