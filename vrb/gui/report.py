"""TradeStation-style Strategy Performance Report.

Metrics follow the TradeStation Performance Summary layout: All Trades plus
the Buys/Sells split (our option-structure analogue of Long/Short), organized
into the classic sections — P&L, trade analysis, streaks, time analysis, and
drawdown/return-on-account.
"""

from __future__ import annotations

import numpy as np

SECONDS_PER_DAY_SESSION = 6.5 * 3600  # 08:30-15:00 CT


def _fmt_money(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    return f"(${abs(v):,.2f})" if v < 0 else f"${v:,.2f}"


def _fmt_pct(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.2f}%"


def _fmt_num(v: float, nd: int = 2) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:,.{nd}f}"


def _streak(wins: np.ndarray, target: bool) -> int:
    best = run = 0
    for w in wins:
        run = run + 1 if w == target else 0
        best = max(best, run)
    return best


def _minutes(td: np.ndarray) -> float:
    return float(np.mean(td)) / 60.0 if len(td) else np.nan


def _trade_stats(trades: list[dict]) -> dict:
    pnl = np.array([t["pnl"] for t in trades], np.float64)
    dur = np.array([(t["exit_ts"] - t["entry_ts"]).astype("timedelta64[s]").astype(np.int64)
                    for t in trades], np.float64)
    wins = pnl > 0
    losses = pnl < 0
    gp = float(pnl[wins].sum()) if wins.any() else 0.0
    gl = float(pnl[losses].sum()) if losses.any() else 0.0
    return {
        "n": len(pnl),
        "net": float(pnl.sum()),
        "gross_profit": gp,
        "gross_loss": gl,
        "profit_factor": gp / abs(gl) if gl < 0 else np.inf if gp > 0 else np.nan,
        "pct_profitable": 100.0 * wins.mean() if len(pnl) else np.nan,
        "n_win": int(wins.sum()),
        "n_loss": int(losses.sum()),
        "n_even": int((pnl == 0).sum()),
        "avg_trade": float(pnl.mean()) if len(pnl) else np.nan,
        "avg_win": float(pnl[wins].mean()) if wins.any() else np.nan,
        "avg_loss": float(pnl[losses].mean()) if losses.any() else np.nan,
        "win_loss_ratio": (float(pnl[wins].mean()) / abs(float(pnl[losses].mean()))
                           if wins.any() and losses.any() else np.nan),
        "largest_win": float(pnl.max()) if len(pnl) else np.nan,
        "largest_loss": float(pnl.min()) if len(pnl) else np.nan,
        "max_consec_win": _streak(wins, True),
        "max_consec_loss": _streak(losses, True) if len(pnl) else 0,
        "avg_min_total": _minutes(dur),
        "avg_min_win": _minutes(dur[wins]),
        "avg_min_loss": _minutes(dur[losses]),
        "contracts": int(sum(t["contracts"] for t in trades)),
        "in_market_secs": float(dur.sum()),
        "pnl_series": pnl,
    }


def compute_report(payloads: list[dict]) -> dict:
    """payloads: day_payload dicts from workers.py -> report model."""
    all_trades = [t for p in payloads for t in p["trades"]]
    buys = [t for t in all_trades if t["direction"] == "buy"]
    sells = [t for t in all_trades if t["direction"] == "sell"]
    a, b, s = (_trade_stats(x) for x in (all_trades, buys, sells))

    daily = np.array([p["pnl"] for p in payloads], np.float64)
    cum = np.cumsum(daily)

    # intraday peak-to-valley drawdown across the whole period, 5-sec marks
    offsets = np.concatenate([[0.0], cum[:-1]])
    intraday = np.concatenate(
        [off + p["equity"] for off, p in zip(offsets, payloads)]) if payloads else np.array([0.0])
    run_peak = np.maximum.accumulate(np.concatenate([[0.0], intraday]))[1:]
    dd_intraday = float((intraday - run_peak).min()) if len(intraday) else 0.0

    closed_cum = np.concatenate([[0.0], np.cumsum(a["pnl_series"])])
    dd_closed = float((closed_cum - np.maximum.accumulate(closed_cum)).min())

    n_days = len(payloads)
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
              if n_days > 1 and daily.std() > 0 else np.nan)
    acct = abs(dd_intraday)
    months = max(n_days / 21.0, 1e-9)
    pct_in_market = 100.0 * a["in_market_secs"] / (n_days * SECONDS_PER_DAY_SESSION) if n_days else np.nan

    rows: list[tuple] = []  # (kind, label, all, buys, sells)

    def sec(name: str) -> None:
        rows.append(("section", name, "", "", ""))

    def row(label: str, fmt, key: str | None = None, vals: tuple | None = None) -> None:
        v = vals if vals is not None else (a[key], b[key], s[key])
        rows.append(("row", label, fmt(v[0]), fmt(v[1]), fmt(v[2])))

    sec("TradeStation Performance Summary")
    row("Total Net Profit", _fmt_money, "net")
    row("Gross Profit", _fmt_money, "gross_profit")
    row("Gross Loss", _fmt_money, "gross_loss")
    row("Profit Factor", _fmt_num, "profit_factor")

    sec("Trade Analysis")
    row("Total Number of Trades", lambda v: f"{int(v)}", "n")
    row("Percent Profitable", _fmt_pct, "pct_profitable")
    row("Winning Trades", lambda v: f"{int(v)}", "n_win")
    row("Losing Trades", lambda v: f"{int(v)}", "n_loss")
    row("Even Trades", lambda v: f"{int(v)}", "n_even")
    row("Avg. Trade Net Profit", _fmt_money, "avg_trade")
    row("Avg. Winning Trade", _fmt_money, "avg_win")
    row("Avg. Losing Trade", _fmt_money, "avg_loss")
    row("Ratio Avg. Win:Avg. Loss", _fmt_num, "win_loss_ratio")
    row("Largest Winning Trade", _fmt_money, "largest_win")
    row("Largest Losing Trade", _fmt_money, "largest_loss")
    row("Max. Consecutive Winning Trades", lambda v: f"{int(v)}", "max_consec_win")
    row("Max. Consecutive Losing Trades", lambda v: f"{int(v)}", "max_consec_loss")
    row("Total Contracts Traded", lambda v: f"{int(v)}", "contracts")

    sec("Time Analysis")
    row("Avg. Time in Total Trades (min)", lambda v: _fmt_num(v, 1), "avg_min_total")
    row("Avg. Time in Winning Trades (min)", lambda v: _fmt_num(v, 1), "avg_min_win")
    row("Avg. Time in Losing Trades (min)", lambda v: _fmt_num(v, 1), "avg_min_loss")
    rows.append(("row", "Trading Period (days)", f"{n_days}", "", ""))
    rows.append(("row", "Percent of Time in the Market", _fmt_pct(pct_in_market), "", ""))

    sec("Drawdown & Returns")
    rows.append(("row", "Max. Drawdown (Intraday Peak to Valley)", _fmt_money(dd_intraday), "", ""))
    rows.append(("row", "Max. Drawdown (Trade Close to Trade Close)", _fmt_money(dd_closed), "", ""))
    rows.append(("row", "Account Size Required", _fmt_money(acct), "", ""))
    rows.append(("row", "Return on Account",
                 _fmt_pct(100.0 * a["net"] / acct if acct > 0 else np.nan), "", ""))
    rows.append(("row", "Avg. Monthly Return", _fmt_money(a["net"] / months), "", ""))
    rows.append(("row", "Sharpe Ratio (daily, annualized)", _fmt_num(sharpe), "", ""))
    rows.append(("row", "Day Win Rate",
                 _fmt_pct(100.0 * float((daily > 0).mean()) if n_days else np.nan), "", ""))

    return {"rows": rows, "daily": daily, "dates": [p["date"] for p in payloads]}


def monthly_table(payloads: list[dict]) -> list[tuple[str, float, float]]:
    """[(YYYY-MM, month_pnl, cum_pnl)]"""
    out: dict[str, float] = {}
    for p in payloads:
        key = f"{p['date'][:4]}-{p['date'][4:6]}"
        out[key] = out.get(key, 0.0) + p["pnl"]
    cum = 0.0
    table = []
    for k in sorted(out):
        cum += out[k]
        table.append((k, out[k], cum))
    return table
