"""Train the ML forecaster and backtest its signal on out-of-sample days.

Pipeline:
  1. build snapshot features over N days (optionally + Kronos forecast features)
  2. walk-forward split by day, fit gradient boosting on forward returns
  3. report IC / hit-rate metrics
  4. turn test-set predictions into {-1,0,+1} signals and run them through the
     option backtester (long ATM call/put), so the model is judged on net P&L
     against real NBBO quotes, not just statistics.

Usage:
    python -m vrb.scripts.train_forecaster --last 90
    python -m vrb.scripts.train_forecaster --last 90 --with-kronos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..backtest import stats
from ..backtest.engine import Backtest
from ..backtest.strategies import SignalDirectional
from ..data.calendar import common_dates
from ..ml import features, forecaster
from ..options.chain import ChainDay

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "vrb_out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="SPXW")
    ap.add_argument("--last", type=int, default=90)
    ap.add_argument("--every", type=int, default=60, help="sample every N secs")
    ap.add_argument("--horizon", type=int, default=900, help="target horizon secs")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--with-kronos", action="store_true")
    ap.add_argument("--hold", type=int, default=900, help="signal hold secs")
    args = ap.parse_args()

    dates = common_dates(args.root)[-args.last:]
    print(f"building features over {len(dates)} days ({dates[0]}..{dates[-1]}), "
          f"every {args.every}s, horizon {args.horizon}s")
    ds = features.build_dataset(dates, args.root, args.every, args.horizon)
    print(f"dataset: X {ds['X'].shape}, targets {ds['target_names']}")

    kf = None
    if args.with_kronos:
        from ..ml.kronos_forecaster import KRONOS_FEATURE_NAMES, KronosForecaster
        print("adding Kronos forecast features (GPU)...")
        kf = KronosForecaster(pred_len=max(1, args.horizon // 60))
        extra = np.full((len(ds["X"]), 2), np.nan)
        for d in sorted(set(ds["days"])):
            rows = np.where(ds["days"] == d)[0]
            day = ChainDay.load(d, args.root)
            extra[rows] = kf.day_features(d, day.ts, ds["t_idx"][rows])
            print(f"  {d}: kronos features for {len(rows)} rows")
        keep = np.isfinite(extra).all(axis=1)
        ds["X"] = np.hstack([ds["X"], extra])[keep]
        ds["Y"], ds["days"], ds["t_idx"] = ds["Y"][keep], ds["days"][keep], ds["t_idx"][keep]
        ds["feature_names"] = list(ds["feature_names"]) + KRONOS_FEATURE_NAMES
        print(f"dataset with kronos: X {ds['X'].shape}")

    wf = forecaster.walk_forward(ds, target="fwd_ret", train_frac=args.train_frac)
    print("\n=== walk-forward metrics (fwd_ret) ===")
    for k, v in wf.metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  train days {len(wf.train_days)}  test days {len(wf.test_days)}")

    # signal threshold from train-set prediction spread
    tr_mask = np.isin(ds["days"], wf.train_days)
    train_pred = wf.model.predict(ds["X"][tr_mask])
    threshold = float(np.quantile(np.abs(train_pred), 0.85))
    print(f"\nsignal threshold |pred| > {threshold:.6f} (85th pct of train)")

    print("\nbacktesting signal on test days (long ATM call/put, "
          f"hold {args.hold}s, real NBBO fills):")
    results = []
    for d in wf.test_days:
        day = ChainDay.load(d, args.root)
        # regenerate features for signal use, filtered on X only — the dataset
        # rows were also filtered on finite TARGETS, which is future knowledge
        # a live trader wouldn't have at decision time
        day_X, _, day_t = features.day_features(day, args.every, args.horizon)
        if kf is not None:
            day_X = np.hstack([day_X, kf.day_features(d, day.ts, day_t)])
        sig = forecaster.signal_for_day(
            wf.model, day_X, day_t, len(day.ts), threshold)
        engine = Backtest(day)
        res = engine.run(SignalDirectional(sig, hold_secs=args.hold))
        results.append(res)
        n = len(res.trades)
        print(f"  {d}: pnl ${res.pnl:9.2f}  ({n} trades)")
    stats.print_summary("ML signal (out-of-sample)", stats.summarize(results))

    OUT_DIR.mkdir(exist_ok=True)
    model_path = OUT_DIR / ("forecaster_kronos.joblib" if args.with_kronos else "forecaster.joblib")
    forecaster.save(wf.model, model_path)
    print(f"\nsaved model -> {model_path}")


if __name__ == "__main__":
    main()
