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


class LastHourGammaExplosion(Strategy):
    """Buy 0DTE options on SuperTrend reversals; ride to a profit multiple or expiry.

    Signal source is a 1-min futures chart (default ES). When SuperTrend flips
    bullish we buy a ~target_delta CALL (long signal, blue); when it flips
    bearish we buy a ~target_delta PUT (short signal, red). Each position is
    held until its bid reaches `target_mult` x our entry cost (profit target)
    or it expires at 15:00 settlement ("dies worthless" if it finishes OTM).
    Signals are only taken inside the [entry_time, exit_time] window — the last
    hour, where 0DTE gamma is largest. Concurrent positions are allowed; each
    is managed independently.
    """

    def __init__(self, entry_time="14:00:00", exit_time="15:00:00",
                 atr_period=10, atr_mult=3.0, target_mult=5.0,
                 target_delta=0.20, qty=1, signal_symbol="ES",
                 min_tte_secs=120):
        self.entry_time, self.exit_time = entry_time, exit_time
        self.atr_period, self.atr_mult = int(atr_period), float(atr_mult)
        self.target_mult, self.target_delta = float(target_mult), float(target_delta)
        self.qty, self.signal_symbol = int(qty), signal_symbol
        # don't act on a reversal with less than this long to expiry — too
        # little time left to buy a meaningful option (delta selection degenerates)
        self.min_tte_secs = int(min_tte_secs)

    def on_day_start(self, engine: Backtest) -> None:
        from ..indicators.supertrend import day_signals
        day = engine.day
        self.by_t: dict[int, list[int]] = {}
        try:
            sig = day_signals(day.date, self.signal_symbol, self.atr_period,
                              self.atr_mult, self.entry_time, self.exit_time)
        except FileNotFoundError:
            return  # no futures bars for this day -> no signals
        for ts, right in sig["events"]:
            gt = int(np.searchsorted(day.ts, ts))
            if 0 <= gt < len(day.ts):
                self.by_t.setdefault(gt, []).append(right)

    def _pick_delta_strike(self, day, t: int, right: int) -> int | None:
        g = day.greeks_at(t)
        delta = g["delta"][:, right]
        ask = day.ask[t, :, right]
        if right == CALL:
            valid = np.isfinite(delta) & (day.strikes > day.spot[t]) & (ask > 0)
            diff = np.abs(delta - self.target_delta)
        else:
            valid = np.isfinite(delta) & (day.strikes < day.spot[t]) & (ask > 0)
            diff = np.abs(np.abs(delta) - self.target_delta)
        if not valid.any():
            return None
        return int(np.argmin(np.where(valid, diff, np.inf)))

    def on_snapshot(self, engine: Backtest, t: int) -> None:
        day = engine.day
        tte = (day.expiry - day.ts[t]) / np.timedelta64(1, "s")
        # open a new long-option position for each signal landing at this snapshot
        for right in self.by_t.get(t, []):
            if tte < self.min_tte_secs:
                continue
            k = self._pick_delta_strike(day, t, right)
            if k is None:
                continue
            label = "gamma_call" if right == CALL else "gamma_put"
            trade = engine.open(t, [Leg(k, right, self.qty)], label)
            if trade is not None:
                trade.signal_direction = "buy" if right == CALL else "sell"

        # manage open positions: take profit at target_mult x entry cost
        for trade in list(engine.open_trades):
            leg = trade.legs[0]
            cost = float(trade.entry_prices[0])
            bid = float(day.bid[t, leg.k, leg.right])
            if cost > 0 and np.isfinite(bid) and bid >= self.target_mult * cost:
                engine.close(t, trade, "target")
        # anything still open rides to 15:00 expiry settlement (worthless if OTM)


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
