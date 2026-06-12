"""End-to-end ML pipeline shared by the GUI and CLI.

build features -> walk-forward train -> threshold -> out-of-sample signal
backtest with real NBBO fills. Returns everything a UI needs to display.
"""

from __future__ import annotations

import numpy as np

from ..backtest.engine import Backtest
from ..backtest.strategies import SignalDirectional
from ..options.chain import ChainDay
from . import features, forecaster


def _feature_ic(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[tuple[str, float]]:
    """Per-feature |Spearman| vs target on the given rows (train diagnostics)."""
    out = []
    for j, name in enumerate(names):
        ic = forecaster._spearman(X[:, j], y)
        out.append((name, float(abs(ic)) if np.isfinite(ic) else 0.0))
    return sorted(out, key=lambda kv: -kv[1])


def run_ml_pipeline(
    dates: list[str],
    root: str = "SPXW",
    every_secs: int = 60,
    horizon_secs: int = 900,
    train_frac: float = 0.7,
    threshold_q: float = 0.85,
    hold_secs: int = 900,
    target: str = "fwd_ret",
    progress_cb=lambda msg: None,
) -> dict:
    progress_cb(f"building features over {len(dates)} days...")
    ds = features.build_dataset(dates, root, every_secs, horizon_secs, verbose=False)
    progress_cb(f"dataset {ds['X'].shape}; training walk-forward...")
    wf = forecaster.walk_forward(ds, target=target, train_frac=train_frac)

    tr_mask = np.isin(ds["days"], wf.train_days)
    y_col = ds["target_names"].index(target)
    train_pred = wf.model.predict(ds["X"][tr_mask])
    threshold = float(np.quantile(np.abs(train_pred), threshold_q))
    fi = _feature_ic(ds["X"][tr_mask], ds["Y"][tr_mask, y_col], list(ds["feature_names"]))

    from ..backtest.payload import day_payload  # Qt-free, safe for headless CLI

    payloads = []
    for i, d in enumerate(wf.test_days, 1):
        day = ChainDay.load(d, root)
        day_X, _, day_t = features.day_features(day, every_secs, horizon_secs)
        sig = forecaster.signal_for_day(wf.model, day_X, day_t, len(day.ts), threshold)
        engine = Backtest(day)
        result = engine.run(SignalDirectional(sig, hold_secs=hold_secs))
        payloads.append(day_payload(day, result))
        progress_cb(f"[{i}/{len(wf.test_days)}] {d}  pnl ${result.pnl:,.2f}")

    return {
        "metrics": wf.metrics,
        "train_days": wf.train_days,
        "test_days": wf.test_days,
        "threshold": threshold,
        "feature_ic": fi,
        "payloads": payloads,
        "model": wf.model,
    }
