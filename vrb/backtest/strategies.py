"""0DTE strategies. Subclass Strategy and implement on_snapshot().

The engine calls on_snapshot for every 5-sec grid point; strategies read the
chain through engine.day and act through engine.open/close. Times are CT.
"""

from __future__ import annotations

import numpy as np

from ..options.chain import CALL, PUT
from .engine import Backtest, Leg, Trade


class Strategy:
    def on_day_start(self, engine: Backtest) -> None:
        pass

    def on_snapshot(self, engine: Backtest, t: int) -> None:
        raise NotImplementedError


class TimedExitMixin:
    """Shared stop-loss / profit-target / time-exit management for premium sellers.

    Exits when the cost to buy back the structure rises to stop_mult x credit,
    falls to (1 - profit_frac) x credit, or the clock hits exit time.
    """

    stop_mult: float = 2.0
    profit_frac: float | None = 0.5

    def manage(self, engine: Backtest, t: int, trade: Trade, t_exit: int) -> None:
        credit = -trade.entry_value          # premium received (>0 for sellers)
        value = -engine.mark(t, trade)       # what the structure is still worth
        if t >= t_exit:
            engine.close(t, trade, "time")
        elif credit <= 0:
            pass
        elif value >= self.stop_mult * credit:
            engine.close(t, trade, "stop")
        elif self.profit_frac is not None and value <= (1 - self.profit_frac) * credit:
            engine.close(t, trade, "target")


class ShortStraddle(TimedExitMixin, Strategy):
    """Sell the ATM straddle at entry_time, manage to stop/target/time."""

    def __init__(self, entry_time="09:00:00", exit_time="14:45:00",
                 stop_mult=2.0, profit_frac=0.5, qty=1):
        self.entry_time, self.exit_time = entry_time, exit_time
        self.stop_mult, self.profit_frac, self.qty = stop_mult, profit_frac, qty
        self.trade: Trade | None = None

    def on_day_start(self, engine: Backtest) -> None:
        self.t_entry = engine.day.t_index(self.entry_time)
        self.t_exit = engine.day.t_index(self.exit_time)
        self.trade = None
        self.done = False

    def on_snapshot(self, engine: Backtest, t: int) -> None:
        if self.done or t < self.t_entry:
            return
        if self.trade is None:
            k = engine.day.atm_k(t)
            legs = [Leg(k, CALL, -self.qty), Leg(k, PUT, -self.qty)]
            self.trade = engine.open(t, legs, "short_straddle")
            if self.trade is None and t > self.t_entry + 60:  # give up after 5 min
                self.done = True
            return
        if self.trade in engine.open_trades:
            self.manage(engine, t, self.trade, self.t_exit)
        else:
            self.done = True


class IronCondor(TimedExitMixin, Strategy):
    """Sell call+put at ~target_delta, buy wings wing_pts further out."""

    def __init__(self, entry_time="09:00:00", exit_time="14:45:00",
                 target_delta=0.16, wing_pts=25.0, stop_mult=2.0,
                 profit_frac=0.5, qty=1):
        self.entry_time, self.exit_time = entry_time, exit_time
        self.target_delta, self.wing_pts = target_delta, wing_pts
        self.stop_mult, self.profit_frac, self.qty = stop_mult, profit_frac, qty

    def on_day_start(self, engine: Backtest) -> None:
        self.t_entry = engine.day.t_index(self.entry_time)
        self.t_exit = engine.day.t_index(self.exit_time)
        self.trade: Trade | None = None
        self.done = False

    def _pick_legs(self, engine: Backtest, t: int) -> list[Leg] | None:
        day = engine.day
        g = day.greeks_at(t)
        delta = g["delta"]                       # (K, 2)
        call_d, put_d = delta[:, CALL], np.abs(delta[:, PUT])
        valid_c = np.isfinite(call_d) & (day.strikes > day.spot[t])
        valid_p = np.isfinite(put_d) & (day.strikes < day.spot[t])
        if valid_c.sum() == 0 or valid_p.sum() == 0:
            return None
        kc = int(np.nanargmin(np.where(valid_c, np.abs(call_d - self.target_delta), np.inf)))
        kp = int(np.nanargmin(np.where(valid_p, np.abs(put_d - self.target_delta), np.inf)))
        # both wings round OUTWARD (at-or-wider than wing_pts) on uneven grids
        kcw = int(np.searchsorted(day.strikes, day.strikes[kc] + self.wing_pts))
        kpw = int(np.searchsorted(day.strikes, day.strikes[kp] - self.wing_pts,
                                  side="right")) - 1
        if kcw >= len(day.strikes) or kpw < 0 or kpw >= kp or kcw <= kc:
            return None
        q = self.qty
        return [Leg(kc, CALL, -q), Leg(kcw, CALL, q), Leg(kp, PUT, -q), Leg(kpw, PUT, q)]

    def on_snapshot(self, engine: Backtest, t: int) -> None:
        if self.done or t < self.t_entry:
            return
        if self.trade is None:
            legs = self._pick_legs(engine, t)
            self.trade = engine.open(t, legs, "iron_condor") if legs else None
            if self.trade is None and t > self.t_entry + 60:
                self.done = True
            return
        if self.trade in engine.open_trades:
            self.manage(engine, t, self.trade, self.t_exit)
        else:
            self.done = True


class SignalDirectional(Strategy):
    """Trade long ATM calls/puts off an external per-snapshot signal.

    signal: (T,) array in {-1, 0, +1}. Enters on nonzero signal, holds for
    hold_secs, one position at a time, stops trading after last_entry time.
    """

    def __init__(self, signal: np.ndarray, hold_secs=900, qty=1,
                 last_entry="14:30:00"):
        self.signal, self.hold_secs, self.qty = signal, hold_secs, qty
        self.last_entry = last_entry

    def on_day_start(self, engine: Backtest) -> None:
        self.trade: Trade | None = None
        self.exit_at = -1
        self.t_last = engine.day.t_index(self.last_entry)

    def on_snapshot(self, engine: Backtest, t: int) -> None:
        if self.trade is not None and self.trade in engine.open_trades:
            if t >= self.exit_at:
                if engine.close(t, self.trade, "time"):
                    self.trade = None
            return
        s = self.signal[t]
        if s == 0 or t > self.t_last:
            return
        k = engine.day.atm_k(t)
        right = CALL if s > 0 else PUT
        self.trade = engine.open(t, [Leg(k, right, self.qty)], "signal")
        if self.trade is not None:
            steps = max(1, self.hold_secs // 5)
            self.exit_at = min(t + steps, len(engine.day.ts) - 2)
