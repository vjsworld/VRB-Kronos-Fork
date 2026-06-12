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
    legs_detail = [{"strike": float(day.strikes[l.k]), "right": int(l.right),
                    "qty": int(l.qty)} for l in trade.legs]
    debit = trade.entry_value > 0
    # transaction = what we actually did (debit=BUY, credit=SELL);
    # direction = market view for the arrow color (may differ, e.g. long put).
    transaction = "BUY" if debit else "SELL"
    direction = trade.signal_direction or ("buy" if debit else "sell")
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
        "direction": direction,
        "transaction": transaction,
        "legs_detail": legs_detail,
        "entry_text": "",
        "exit_text": "",
    }


def day_payload(day: ChainDay, result: DayResult) -> dict:
    trades = []
    for tr in result.trades:
        t = trade_payload(day, tr)
        t["entry_text"] = f"{t['transaction']} {t['legs']} @ {abs(t['entry_value']):.2f}"
        t["exit_text"] = f"{t['reason'].upper()} @ {abs(t['exit_value']):.2f}"
        trades.append(t)
    costs = _cost_decomposition(day, result)
    return {
        "date": result.date,
        "pnl": result.pnl,
        "n_trades": len(result.trades),
        "reasons": ", ".join(tr.exit_reason for tr in result.trades),
        "trades": trades,
        "equity_ts": day.ts,
        "equity": result.equity,
        "spot_close": float(day.settlement),
        # P&L decomposition (dollars): net = gross_mid - spread - commission
        "gross_mid_pnl": costs["gross_mid"],
        "cost_spread": costs["spread"],
        "cost_commission": costs["commission"],
    }


def _cost_decomposition(day: ChainDay, result: DayResult) -> dict:
    """Split realized P&L into the edge at mid vs the bid/ask spread paid vs
    commission. gross_mid - spread - commission == net pnl. Settlement legs
    have no spread (cash-settled at intrinsic)."""
    mult = day.multiplier
    gross_mid = spread = commission = 0.0
    for tr in result.trades:
        commission += tr.entry_costs + tr.exit_costs
        settled = tr.exit_reason == "settlement"
        for i, leg in enumerate(tr.legs):
            entry_mid = float(day.mid[tr.entry_t, leg.k, leg.right])
            if not np.isfinite(entry_mid):
                entry_mid = float(tr.entry_prices[i])
            if settled:
                exit_mid = float(tr.exit_prices[i])  # intrinsic, no market spread
            else:
                exit_mid = float(day.mid[tr.exit_t, leg.k, leg.right])
                if not np.isfinite(exit_mid):
                    exit_mid = float(tr.exit_prices[i])
            leg_mid = leg.qty * (exit_mid - entry_mid) * mult
            leg_fill = leg.qty * (tr.exit_prices[i] - tr.entry_prices[i]) * mult
            gross_mid += leg_mid
            spread += leg_mid - leg_fill  # edge given up to the spread (>=0 typ.)
    return {"gross_mid": gross_mid, "spread": spread, "commission": commission}


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
