"""Walk-forward gradient-boosting forecaster over snapshot features.

Splits are strictly by trade date (no shuffling) so the test set is always
out-of-sample in time. Reports rank IC and directional hit rate, and can
convert predictions into a per-snapshot {-1, 0, +1} signal for the
SignalDirectional option strategy — closing the loop from model to P&L
against real bid/ask quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan


@dataclass
class WalkForwardResult:
    model: HistGradientBoostingRegressor
    train_days: list[str]
    test_days: list[str]
    pred: np.ndarray          # test-set predictions
    truth: np.ndarray
    test_mask: np.ndarray     # rows of the dataset in the test split
    metrics: dict[str, float]


def walk_forward(
    dataset: dict,
    target: str = "fwd_ret",
    train_frac: float = 0.7,
    **model_kwargs,
) -> WalkForwardResult:
    X, Y, days = dataset["X"], dataset["Y"], dataset["days"]
    y = Y[:, dataset["target_names"].index(target)]

    unique_days = sorted(set(days))
    n_train = max(1, int(len(unique_days) * train_frac))
    train_days, test_days = unique_days[:n_train], unique_days[n_train:]
    tr = np.isin(days, train_days)
    te = np.isin(days, test_days)

    # early_stopping is OFF deliberately: sklearn's internal validation split
    # is shuffled, and samples 60s apart share 900s-forward targets, so the
    # val score would be contaminated by temporal overlap with train rows.
    params = dict(max_iter=200, learning_rate=0.05, max_depth=4,
                  l2_regularization=1.0, early_stopping=False, random_state=7)
    params.update(model_kwargs)
    model = HistGradientBoostingRegressor(**params)
    model.fit(X[tr], y[tr])

    pred = model.predict(X[te])
    truth = y[te]

    per_day_ic = [_spearman(pred[days[te] == d], truth[days[te] == d]) for d in test_days]
    per_day_ic = [ic for ic in per_day_ic if np.isfinite(ic)]
    nonzero = np.abs(pred) > np.quantile(np.abs(pred), 0.7)  # top-30% conviction
    metrics = {
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "ic_overall": _spearman(pred, truth),
        "ic_day_mean": float(np.mean(per_day_ic)) if per_day_ic else np.nan,
        "ic_day_t": (float(np.mean(per_day_ic) / (np.std(per_day_ic) / np.sqrt(len(per_day_ic))))
                     if len(per_day_ic) > 2 and np.std(per_day_ic) > 0 else np.nan),
        "hit_rate_all": float((np.sign(pred) == np.sign(truth)).mean()),
        "hit_rate_conviction": float((np.sign(pred[nonzero]) == np.sign(truth[nonzero])).mean())
        if nonzero.any() else np.nan,
    }
    return WalkForwardResult(model, train_days, test_days, pred, truth, te, metrics)


def signal_for_day(
    model,
    day_X: np.ndarray,
    t_idx: np.ndarray,
    n_grid: int,
    threshold: float,
) -> np.ndarray:
    """Expand sampled predictions into a per-grid-point {-1,0,+1} signal."""
    sig = np.zeros(n_grid, np.int8)
    keep = np.isfinite(day_X).all(axis=1)
    if keep.sum() == 0:
        return sig
    pred = model.predict(day_X[keep])
    pos = t_idx[keep]
    sig[pos[pred > threshold]] = 1
    sig[pos[pred < -threshold]] = -1
    return sig


def save(model, path) -> None:
    joblib.dump(model, path)


def load(path):
    return joblib.load(path)
