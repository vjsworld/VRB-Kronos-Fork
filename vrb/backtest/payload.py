"""Plain-dict result payloads consumed by the GUI, reports, and exports.

No Qt imports here — this is the bridge between the engine and any frontend.
"""

from __future__ import annotations

import numpy as np

from ..options.chain import CALL, ChainDay
from .engine import Backtest, DayResult, Trade


def legs_text(day: ChainDay, trade: Trade) -> str:
    parts = []
    for leg in trade.legs:
        side = "+" if leg.qty > 0 else "-"
        right = "C" if leg.right == CALL else "P"
        parts.append(f"{side}{abs(leg.qty)} {day.strikes[leg.k]:.0f}{right}")
    return " ".join(parts)


def trade_payload(day: ChainDay, trade: Trade) -> dict:
    exit_value = float(np.sum(
        np.array([l.qty for l in trade.legs], np.float64) * trade.exit_prices))
    return {
        "label": trade.label,
        "legs": legs_text(day, trade),
        "entry_ts": day.ts[trade.entry_t],
        "exit_ts": day.ts[trade.exit_t],
        "entry_value": trade.entry_value,
        "exit_value": exit_value,
        "pnl": trade.pnl(day.multiplier),
        "reason": trade.exit_reason,
        "contracts": int(sum(abs(l.qty) for l in trade.legs)),
        "direction": "buy" if trade.entry_value > 0 else "sell",
        "entry_text": "",
        "exit_text": "",
    }


def day_payload(day: ChainDay, result: DayResult) -> dict:
    trades = []
    for tr in result.trades:
        t = trade_payload(day, tr)
        side = "BUY" if t["direction"] == "buy" else "SELL"
        t["entry_text"] = f"{side} {t['legs']} @ {abs(t['entry_value']):.2f}"
        t["exit_text"] = f"{t['reason'].upper()} @ {abs(t['exit_value']):.2f}"
        trades.append(t)
    return {
        "date": result.date,
        "pnl": result.pnl,
        "n_trades": len(result.trades),
        "reasons": ", ".join(tr.exit_reason for tr in result.trades),
        "trades": trades,
        "equity_ts": day.ts,
        "equity": result.equity,
        "spot_close": float(day.settlement),
    }


def run_backtest_days(dates: list[str], strategy_factory, root: str,
                      progress_cb, engine_kwargs: dict | None = None) -> list[dict]:
    payloads = []
    for i, d in enumerate(dates, 1):
        try:
            day = ChainDay.load(d, root)
        except (FileNotFoundError, ValueError):
            progress_cb(f"[{i}/{len(dates)}] {d} skipped (no usable data)")
            continue
        engine = Backtest(day, **(engine_kwargs or {}))
        result = engine.run(strategy_factory())
        payloads.append(day_payload(day, result))
        progress_cb(f"[{i}/{len(dates)}] {d}  pnl ${result.pnl:,.2f}")
    return payloads
