"""Multi-day backtest runner and performance statistics."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..options.chain import ChainDay
from .engine import Backtest, DayResult


def run_days(
    dates: list[str],
    strategy_factory: Callable[[], object],
    root: str = "SPXW",
    verbose: bool = True,
    **engine_kwargs,
) -> list[DayResult]:
    """Run a fresh strategy instance over each trade date, skipping bad days."""
    results = []
    for d in dates:
        try:
            day = ChainDay.load(d, root)
        except (FileNotFoundError, ValueError) as e:
            if verbose:
                print(f"  {d}: skipped ({e.__class__.__name__})")
            continue
        engine = Backtest(day, **engine_kwargs)
        res = engine.run(strategy_factory())
        results.append(res)
        if verbose:
            n = len(res.trades)
            reasons = ",".join(tr.exit_reason for tr in res.trades) or "no trades"
            print(f"  {d}: pnl ${res.pnl:9.2f}  ({n} trades: {reasons})")
    return results


def summarize(results: list[DayResult]) -> dict[str, float]:
    pnls = np.array([r.pnl for r in results], np.float64)
    trades = [tr for r in results for tr in r.trades]
    if len(pnls) == 0:
        return {"days": 0}
    cum = np.cumsum(pnls)
    # include the zero starting equity so a losing streak from day one counts
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    stats = {
        "days": len(pnls),
        "total_pnl": float(pnls.sum()),
        "avg_day_pnl": float(pnls.mean()),
        "day_win_rate": float((pnls > 0).mean()),
        "sharpe_daily_ann": float(pnls.mean() / pnls.std() * np.sqrt(252)) if pnls.std() > 0 else 0.0,
        "max_drawdown": float((cum - peak).min()),
        "n_trades": len(trades),
        "best_day": float(pnls.max()),
        "worst_day": float(pnls.min()),
    }
    return stats


def print_summary(title: str, stats: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    if stats.get("days", 0) == 0:
        print("no results")
        return
    print(f"days {stats['days']}  trades {stats['n_trades']}  "
          f"total ${stats['total_pnl']:.2f}  avg/day ${stats['avg_day_pnl']:.2f}")
    print(f"day win rate {stats['day_win_rate']:.1%}  sharpe(ann) {stats['sharpe_daily_ann']:.2f}  "
          f"maxDD ${stats['max_drawdown']:.2f}")
    print(f"best ${stats['best_day']:.2f}  worst ${stats['worst_day']:.2f}")
