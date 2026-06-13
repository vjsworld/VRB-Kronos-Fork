"""Walk-forward validation for the theta strangle.

The honest test: split the history into rolling train/test windows. On each
train window pick the best parameter cell; trade ONLY that cell on the next,
untouched test window; concatenate the test segments. The result is an
out-of-sample equity curve with no look-ahead — the difference between a
backtest and an edge.

Efficiency: every grid config is backtested over all days ONCE (the per-day
P&L is cached in a matrix), then the walk-forward is pure slicing. So we can
explore window sizes and selection objectives for free.

Run: python -m vrb.research.walkforward
"""

from __future__ import annotations

import numpy as np

from ..backtest.parallel import run_days_parallel
from ..backtest.strategies import ShortStrangle
from ..data.calendar import common_dates
from ..util.timing import Timer, get_logger

log = get_logger("research.wfo")

ENTRIES = ["10:00:00", "11:00:00", "12:00:00", "13:00:00"]
DELTAS = [0.10, 0.16, 0.22, 0.30]
WINGS = [0.0, 150.0]
FILL = 0.75
TRAIN_DAYS = 120
TEST_DAYS = 40


def build_grid() -> list[dict]:
    grid = []
    for e in ENTRIES:
        for d in DELTAS:
            for w in WINGS:
                grid.append({"label": f"e{e[:5]}_d{d:.2f}_w{int(w)}",
                             "kwargs": dict(entry_time=e, exit_time="15:00:00",
                                            target_delta=d, profit_frac=0.5,
                                            stop_mult=2.0, wing_pts=w)})
    return grid


def precompute(grid: list[dict], dates: list[str], fill: float) -> dict[str, dict]:
    """config label -> {date: net pnl} over every day."""
    out = {}
    with Timer(f"precompute {len(grid)} configs x {len(dates)} days", log):
        for i, cfg in enumerate(grid, 1):
            pls = run_days_parallel(dates, ShortStrangle, cfg["kwargs"], "SPXW",
                                    engine_kwargs={"improvement": fill},
                                    progress_cb=lambda m: None)
            out[cfg["label"]] = {p["date"]: p["pnl"] for p in pls}
            log.info("  [%d/%d] %s done", i, len(grid), cfg["label"])
    return out


def _objective(pnls: np.ndarray, kind: str) -> float:
    if len(pnls) == 0:
        return -1e18
    if kind == "sharpe":
        return pnls.mean() / pnls.std() if pnls.std() > 0 else (pnls.mean() * 1e6)
    return float(pnls.sum())  # "pnl"


def walk_forward(per_cfg: dict[str, dict], dates: list[str], kind: str,
                 train_days=TRAIN_DAYS, test_days=TEST_DAYS):
    """Return (oos_dates, oos_pnls, fold_log)."""
    oos_dates, oos_pnls, folds = [], [], []
    i = 0
    while i + train_days + test_days <= len(dates):
        train = dates[i:i + train_days]
        test = dates[i + train_days:i + train_days + test_days]
        best_lbl, best_obj = None, -1e18
        for lbl, dpnl in per_cfg.items():
            tr = np.array([dpnl.get(d, 0.0) for d in train], float)
            obj = _objective(tr, kind)
            if obj > best_obj:
                best_obj, best_lbl = obj, lbl
        test_pnls = [per_cfg[best_lbl].get(d, 0.0) for d in test]
        oos_dates.extend(test); oos_pnls.extend(test_pnls)
        folds.append((train[0], train[-1], test[0], test[-1], best_lbl, float(np.sum(test_pnls))))
        i += test_days
    return oos_dates, np.array(oos_pnls, float), folds


def stats(pnls: np.ndarray) -> dict:
    if len(pnls) == 0:
        return {}
    cum = np.concatenate([[0.0], np.cumsum(pnls)])
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return {"net": float(pnls.sum()), "days": len(pnls),
            "sharpe": float(pnls.mean() / pnls.std() * np.sqrt(252)) if pnls.std() > 0 else 0.0,
            "daywin": float((pnls > 0).mean()), "maxdd": dd,
            "avg_day": float(pnls.mean())}


def main() -> None:
    dates = common_dates("SPXW")
    grid = build_grid()
    per_cfg = precompute(grid, dates, FILL)

    print(f"\n=== WALK-FORWARD VALIDATION ===")
    print(f"grid {len(grid)} configs | train {TRAIN_DAYS}d -> test {TEST_DAYS}d | fill {FILL}\n")

    # in-sample best (the seductive number): single config best over ALL days
    is_best = max(per_cfg.items(),
                  key=lambda kv: sum(v for v in kv[1].values()))
    is_pnls = np.array(list(is_best[1].values()), float)
    print(f"IN-SAMPLE best single config : {is_best[0]:18s} {stats(is_pnls)}")

    # fixed sensible baseline, no optimization
    base = "e10:00_d0.16_w0"
    if base in per_cfg:
        b = np.array(list(per_cfg[base].values()), float)
        print(f"FIXED 16d naked (no opt)     : {base:18s} {stats(b)}")

    for kind in ("pnl", "sharpe"):
        od, op, folds = walk_forward(per_cfg, dates, kind)
        s = stats(op)
        print(f"\nWALK-FORWARD (select by {kind}): OOS net=${s['net']:,.0f}  sharpe={s['sharpe']:.2f}  "
              f"daywin={s['daywin']*100:.0f}%  maxDD=${s['maxdd']:,.0f}  over {s['days']} OOS days, {len(folds)} folds")
        pos = sum(1 for f in folds if f[5] > 0)
        print(f"  folds positive: {pos}/{len(folds)}   chosen cells over time:")
        for tr0, tr1, te0, te1, lbl, pnl in folds:
            print(f"    train {tr0}-{tr1} -> test {te0}-{te1}: {lbl:18s} OOS ${pnl:>8,.0f}")
        if kind == "pnl":
            _plot(od, op, folds)


def _plot(oos_dates, oos_pnls, folds) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
    except Exception:
        return
    cum = np.cumsum(oos_pnls)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(cum)), cum, color="#26a69a", lw=2)
    ax.axhline(0, color="#888", ls=":")
    ax.fill_between(range(len(cum)), cum, 0, where=(cum < 0), color="#ef5350", alpha=0.25)
    ax.set_title(f"Walk-forward OOS equity — strangle (select by train P&L) — net ${cum[-1]:,.0f}")
    ax.set_xlabel("out-of-sample trading day"); ax.set_ylabel("cumulative $")
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent.parent / "vrb_out" / "walkforward_oos.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="white")
    print(f"\n  OOS equity curve -> {out}")


if __name__ == "__main__":
    main()
