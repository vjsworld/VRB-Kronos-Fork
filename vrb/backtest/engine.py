"""Event-driven single-day 0DTE backtest engine.

Walks the 5-sec snapshot grid of a ChainDay. Fills are taken from the actual
NBBO: buys lift the ask, sells hit the bid (optionally improved toward mid by
FILL_SPREAD_IMPROVEMENT). Positions still open at 15:00 CT are cash-settled at
intrinsic value against the SPX settlement print, like real SPXW 0DTE.

P&L is in dollars (option points x contract multiplier), net of per-contract
commission+fees on every fill (settlement expiry is free, as in real life).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import (COMMISSION_PER_CONTRACT, EXCHANGE_FEES_PER_CONTRACT,
                      FILL_SPREAD_IMPROVEMENT)
from ..options.chain import ChainDay


@dataclass
class Leg:
    k: int        # strike index into chain.strikes
    right: int    # 0=CALL 1=PUT
    qty: int      # >0 long, <0 short (contracts)


@dataclass(eq=False)  # identity semantics: open_trades.remove()/`in` must not
class Trade:          # compare ndarray fields (raises on equal-valued trades)
    label: str
    legs: list[Leg]
    entry_t: int
    entry_prices: np.ndarray            # per-leg fill price (option points)
    exit_t: int | None = None
    exit_prices: np.ndarray | None = None
    exit_reason: str = ""
    entry_costs: float = 0.0            # dollars
    exit_costs: float = 0.0
    # optional market-view tag for chart arrows: "buy" (bullish/blue) or
    # "sell" (bearish/red). Strategies whose debit/credit sign doesn't encode
    # the view (e.g. buying a put is bearish) set this explicitly.
    signal_direction: str = ""

    def pnl(self, multiplier: int) -> float:
        """Realized P&L in dollars (only valid after exit)."""
        qty = np.array([l.qty for l in self.legs], np.float64)
        gross = float(np.sum(qty * (self.exit_prices - self.entry_prices))) * multiplier
        return gross - self.entry_costs - self.exit_costs

    @property
    def entry_value(self) -> float:
        """Signed entry value in option points (negative = net credit)."""
        qty = np.array([l.qty for l in self.legs], np.float64)
        return float(np.sum(qty * self.entry_prices))


class Backtest:
    def __init__(
        self,
        day: ChainDay,
        commission: float = COMMISSION_PER_CONTRACT,
        fees: float = EXCHANGE_FEES_PER_CONTRACT,
        improvement: float = FILL_SPREAD_IMPROVEMENT,
    ):
        self.day = day
        self.cost_per_contract = commission + fees
        self.improvement = improvement
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.equity = np.zeros(len(day.ts), np.float64)  # cumulative P&L marks

    # ------------------------------------------------------------------ fills
    def _fill(self, t: int, legs: list[Leg], closing: bool = False) -> np.ndarray | None:
        """Per-leg fill prices at snapshot t, or None if any leg is unquoted.

        closing=True flips each leg's side (sell what you're long, etc.).
        """
        prices = np.empty(len(legs), np.float64)
        for i, leg in enumerate(legs):
            bid = float(self.day.bid[t, leg.k, leg.right])
            ask = float(self.day.ask[t, leg.k, leg.right])
            if not np.isfinite(bid) or not np.isfinite(ask) or ask <= 0:
                return None
            buying = (leg.qty > 0) != closing
            half_spread = 0.5 * (ask - bid)
            if buying:
                prices[i] = ask - self.improvement * half_spread
            else:
                prices[i] = bid + self.improvement * half_spread
        return prices

    def _costs(self, legs: list[Leg]) -> float:
        return sum(abs(l.qty) for l in legs) * self.cost_per_contract

    # ------------------------------------------------------------------- API
    def open(self, t: int, legs: list[Leg], label: str = "") -> Trade | None:
        prices = self._fill(t, legs, closing=False)
        if prices is None or not np.isfinite(self.day.spot[t]):
            return None
        trade = Trade(label=label, legs=legs, entry_t=t, entry_prices=prices,
                      entry_costs=self._costs(legs))
        self.open_trades.append(trade)
        return trade

    def close(self, t: int, trade: Trade, reason: str = "exit") -> bool:
        prices = self._fill(t, trade.legs, closing=True)
        if prices is None:
            return False
        trade.exit_t = t
        trade.exit_prices = prices
        trade.exit_reason = reason
        trade.exit_costs = self._costs(trade.legs)
        self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        return True

    def mark(self, t: int, trade: Trade) -> float:
        """Liquidation-at-mid value of a trade at t, in option points.

        A NaN mid means the snapshot row is missing, not that the contract is
        worthless — fall back to the last finite mid (one always exists for an
        open trade, since legs only fill at quoted snapshots).
        """
        ks = np.array([l.k for l in trade.legs])
        rs = np.array([l.right for l in trade.legs])
        qty = np.array([l.qty for l in trade.legs], np.float64)
        mids = self.day.mid[t, ks, rs]
        for i in np.flatnonzero(~np.isfinite(mids)):
            col = self.day.mid[:t + 1, ks[i], rs[i]]
            fin = np.flatnonzero(np.isfinite(col))
            mids[i] = col[fin[-1]] if fin.size else 0.0
        return float(np.sum(qty * mids))

    # ------------------------------------------------------------------- run
    def run(self, strategy) -> "DayResult":
        day = self.day
        strategy.on_day_start(self)
        realized = 0.0
        for t in range(len(day.ts)):
            strategy.on_snapshot(self, t)
            newly_closed = [tr for tr in self.closed_trades if tr.exit_t == t]
            for tr in newly_closed:
                realized += tr.pnl(day.multiplier)
            unrealized = sum(
                (self.mark(t, tr) - tr.entry_value) * day.multiplier - tr.entry_costs
                for tr in self.open_trades
            )
            self.equity[t] = realized + unrealized

        # Expiry settlement for anything still open
        t_last = len(day.ts) - 1
        for trade in list(self.open_trades):
            ks = np.array([l.k for l in trade.legs])
            rs = np.array([l.right for l in trade.legs])
            trade.exit_t = t_last
            trade.exit_prices = day.intrinsic(ks, rs)
            trade.exit_reason = "settlement"
            trade.exit_costs = 0.0
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            realized += trade.pnl(day.multiplier)
        self.equity[t_last] = realized
        return DayResult(date=day.date, trades=self.closed_trades, equity=self.equity)


@dataclass
class DayResult:
    date: str
    trades: list[Trade]
    equity: np.ndarray

    @property
    def pnl(self) -> float:
        return float(self.equity[-1])
