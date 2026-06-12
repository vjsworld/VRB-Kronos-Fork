"""Backtest experiment registry.

Every backtest run is recorded as one JSON line in vrb_out/research/backtests.jsonl
so experiments are permanent, comparable, and never silently repeated. A record
holds the strategy, its full parameter set, the date range, and a flat stats
block (net P&L, the gross/spread/commission decomposition, Sharpe, win rates,
profit factor, drawdown, expectancy). This record-keeping is deliberate: in
systematic research the journal of what was tried — and what failed — is the
asset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..util.timing import get_logger

log = get_logger(__name__)

REGISTRY_DIR = Path(__file__).resolve().parent.parent.parent / "vrb_out" / "research"
REGISTRY_PATH = REGISTRY_DIR / "backtests.jsonl"


def summarize_payloads(payloads: list[dict]) -> dict:
    """Flat performance stats computed from day payloads (see backtest.payload)."""
    if not payloads:
        return {"n_days": 0, "n_trades": 0}
    daily = np.array([p["pnl"] for p in payloads], np.float64)
    trades = [t for p in payloads for t in p["trades"]]
    tpnl = np.array([t["pnl"] for t in trades], np.float64) if trades else np.array([])
    wins = tpnl > 0

    gross_mid = float(sum(p.get("gross_mid_pnl", 0.0) for p in payloads))
    spread = float(sum(p.get("cost_spread", 0.0) for p in payloads))
    commission = float(sum(p.get("cost_commission", 0.0) for p in payloads))
    total = float(daily.sum())

    cum = np.concatenate([[0.0], np.cumsum(daily)])
    max_dd = float((cum - np.maximum.accumulate(cum)).min())
    gp = float(tpnl[wins].sum()) if wins.any() else 0.0
    gl = float(tpnl[~wins].sum()) if (~wins).any() and tpnl.size else 0.0

    return {
        "n_days": len(payloads),
        "n_trades": int(tpnl.size),
        "total_pnl": total,
        "gross_mid_pnl": gross_mid,
        "cost_spread": spread,
        "cost_commission": commission,
        "cost_total": spread + commission,
        "avg_day_pnl": float(daily.mean()),
        "avg_trade_pnl": float(tpnl.mean()) if tpnl.size else 0.0,
        "day_win_rate": float((daily > 0).mean()),
        "trade_win_rate": float(wins.mean()) if tpnl.size else 0.0,
        "profit_factor": (gp / abs(gl)) if gl < 0 else float("inf") if gp > 0 else 0.0,
        "sharpe_daily_ann": (float(daily.mean() / daily.std() * np.sqrt(252))
                             if len(daily) > 1 and daily.std() > 0 else 0.0),
        "max_drawdown": max_dd,
        "best_day": float(daily.max()),
        "worst_day": float(daily.min()),
    }


def record(strategy: str, params: dict, root: str, dates: list[str],
           payloads: list[dict], notes: str = "", run_ts: str | None = None) -> dict:
    """Append a backtest run to the registry and return the record."""
    stats = summarize_payloads(payloads)
    rec = {
        "run_id": run_ts or time.strftime("%Y%m%d_%H%M%S"),
        "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy,
        "root": root,
        "start": dates[0] if dates else "",
        "end": dates[-1] if dates else "",
        "params": {k: _jsonable(v) for k, v in params.items()},
        "stats": stats,
        "notes": notes,
    }
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    log.info("recorded %s %s: net $%.0f (spread $%.0f, comm $%.0f) over %d days",
             rec["run_id"], strategy, stats.get("total_pnl", 0),
             stats.get("cost_spread", 0), stats.get("cost_commission", 0),
             stats.get("n_days", 0))
    return rec


def load_all() -> list[dict]:
    """All recorded runs, newest first."""
    if not REGISTRY_PATH.exists():
        return []
    out = []
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return list(reversed(out))


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v
