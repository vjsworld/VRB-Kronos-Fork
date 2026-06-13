"""0DTE strategies. Subclass Strategy and implement on_snapshot().

The engine calls on_snapshot for every 5-sec grid point; strategies read the
chain through engine.day and act through engine.open/close. Times are CT.
"""

from __future__ import annotations

import numpy as np

from ..options import bs
from ..options.chain import CALL, PUT
from .engine import Backtest, Leg, Trade


def trailing_rv(day, t: int, window_secs: int, grid_secs: int = 5) -> float:
    """Annualized realized vol of the underlying over the trailing window,
    measured from the spot series on the option grid (calendar annualization,
    so it's directly comparable to ATM IV)."""
    n = max(2, window_secs // grid_secs)
    s = day.spot[max(0, t - n):t + 1]
    s = s[np.isfinite(s)]
    if len(s) < 10:
        return np.nan
    r = np.diff(np.log(s))
    sd = float(r.std())
    return sd * np.sqrt(bs.SECONDS_PER_YEAR / grid_secs) if sd > 0 else np.nan


def vol_rich_enough(day, t: int, min_iv_rv: float, rv_window_min: float) -> bool:
    """True if ATM IV is at least min_iv_rv x the trailing realized vol — i.e.
    premium is rich enough to sell. min_iv_rv <= 0 disables the filter."""
    if min_iv_rv <= 0:
        return True
    rv = trailing_rv(day, t, int(rv_window_min) * 60)
    iv = day.atm_iv(t)
    if not (np.isfinite(rv) and rv > 0 and np.isfinite(iv)):
        return True  # can't measure -> don't block the trade
    return (iv / rv) >= min_iv_rv


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
                 profit_frac=0.5, qty=1, min_iv_rv=0.0, rv_window_min=30):
        self.entry_time, self.exit_time = entry_time, exit_time
        self.target_delta, self.wing_pts = target_delta, wing_pts
        self.stop_mult, self.profit_frac, self.qty = stop_mult, profit_frac, qty
        self.min_iv_rv, self.rv_window_min = float(min_iv_rv), float(rv_window_min)

    def on_day_start(self, engine: Backtest) -> None:
        self.t_entry = engine.day.t_index(self.entry_time)
        self.t_exit = engine.day.t_index(self.exit_time)
        self.trade: Trade | None = None
        self.done = False

    def _pick_legs(self, engine: Backtest, t: int) -> list[Leg] | None:
        day = engine.day
        if not vol_rich_enough(day, t, self.min_iv_rv, self.rv_window_min):
            return None  # premium not rich enough today
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


class ShortStrangle(TimedExitMixin, Strategy):
    """Sell a call and a put at ~target_delta. With wing_pts > 0, also buy far
    protective wings that distance OUT — capping the tail at minimal cost to the
    gross (deep-OTM wings are nearly free), which is the tail-controlled
    strangle. wing_pts = 0 leaves it naked (undefined risk).

    Time-based management: profit target as a fraction of credit, stop at a
    multiple of credit, else hold to 15:00 settlement.
    """

    def __init__(self, entry_time="10:00:00", exit_time="15:00:00",
                 target_delta=0.16, stop_mult=2.0, profit_frac=0.5, qty=1,
                 min_iv_rv=0.0, rv_window_min=30, wing_pts=0.0):
        self.entry_time, self.exit_time = entry_time, exit_time
        self.target_delta = target_delta
        self.stop_mult, self.profit_frac, self.qty = stop_mult, profit_frac, qty
        self.min_iv_rv, self.rv_window_min = float(min_iv_rv), float(rv_window_min)
        self.wing_pts = float(wing_pts)

    def on_day_start(self, engine: Backtest) -> None:
        self.t_entry = engine.day.t_index(self.entry_time)
        self.t_exit = engine.day.t_index(self.exit_time)
        self.trade: Trade | None = None
        self.done = False

    def _pick_legs(self, engine: Backtest, t: int) -> list[Leg] | None:
        day = engine.day
        if not vol_rich_enough(day, t, self.min_iv_rv, self.rv_window_min):
            return None
        g = day.greeks_at(t)
        call_d, put_d = g["delta"][:, CALL], np.abs(g["delta"][:, PUT])
        valid_c = np.isfinite(call_d) & (day.strikes > day.spot[t]) & (day.ask[t, :, CALL] > 0)
        valid_p = np.isfinite(put_d) & (day.strikes < day.spot[t]) & (day.ask[t, :, PUT] > 0)
        if valid_c.sum() == 0 or valid_p.sum() == 0:
            return None
        kc = int(np.nanargmin(np.where(valid_c, np.abs(call_d - self.target_delta), np.inf)))
        kp = int(np.nanargmin(np.where(valid_p, np.abs(put_d - self.target_delta), np.inf)))
        q = self.qty
        legs = [Leg(kc, CALL, -q), Leg(kp, PUT, -q)]
        if self.wing_pts > 0:  # buy far protective wings -> defined risk
            kcw = int(np.searchsorted(day.strikes, day.strikes[kc] + self.wing_pts))
            kpw = int(np.searchsorted(day.strikes, day.strikes[kp] - self.wing_pts, side="right")) - 1
            if kcw < len(day.strikes) and kcw > kc and kpw >= 0 and kpw < kp:
                legs += [Leg(kcw, CALL, q), Leg(kpw, PUT, q)]
        return legs

    def on_snapshot(self, engine: Backtest, t: int) -> None:
        if self.done or t < self.t_entry:
            return
        if self.trade is None:
            legs = self._pick_legs(engine, t)
            self.trade = engine.open(t, legs, "short_strangle") if legs else None
            if self.trade is None and t > self.t_entry + 60:
                self.done = True
            return
        if self.trade in engine.open_trades:
            self.manage(engine, t, self.trade, self.t_exit)
        else:
            self.done = True


class CompressionBuyer(Strategy):
    """Buy cheap OTM premium when a compression gate fires (the expansion study's
    precursor hypothesis). Gate: trailing realized vol low AND ATM IV/RV rich,
    inside an early entry window. Buys a ~target_delta OTM option (put-weighted
    by default — 62% of 10x events are puts), then exits at target_mult x entry
    cost (asymmetric: a few big wins pay for many full-premium losses) or after
    hold_min, else expiry settlement. Every threshold is optimizable.
    """

    def __init__(self, entry_time="08:35:00", last_entry="13:30:00", side="put",
                 target_delta=0.06, target_mult=5.0, min_premium=0.20, max_premium=1.50,
                 max_rv=0.30, min_iv_rv=1.05, rv_window_min=15, hold_min=60,
                 max_concurrent=1, qty=1):
        self.entry_time, self.last_entry = entry_time, last_entry
        self.side = PUT if side == "put" else CALL
        self.target_delta, self.target_mult = float(target_delta), float(target_mult)
        self.min_premium, self.max_premium = float(min_premium), float(max_premium)
        self.max_rv, self.min_iv_rv = float(max_rv), float(min_iv_rv)
        self.rv_window_min, self.hold_min = float(rv_window_min), int(hold_min)
        self.max_concurrent, self.qty = int(max_concurrent), int(qty)

    def on_day_start(self, engine: Backtest) -> None:
        self.t_first = engine.day.t_index(self.entry_time)
        self.t_last = engine.day.t_index(self.last_entry)
        self.exit_by: dict[int, int] = {}  # id(trade) -> exit grid index

    def _gate(self, day, t: int) -> bool:
        rv = trailing_rv(day, t, int(self.rv_window_min) * 60)
        iv = day.atm_iv(t)
        if not (np.isfinite(rv) and rv > 0 and np.isfinite(iv)):
            return False
        return rv <= self.max_rv and (iv / rv) >= self.min_iv_rv

    def _pick(self, day, t: int) -> int | None:
        r = self.side
        mid = day.mid[t, :, r]
        otm = (day.strikes < day.spot[t]) if r == PUT else (day.strikes > day.spot[t])
        delta = np.abs(day.greeks_at(t)["delta"][:, r])
        ok = (otm & np.isfinite(mid) & (mid >= self.min_premium) & (mid <= self.max_premium)
              & (day.ask[t, :, r] > 0) & np.isfinite(delta))
        if not ok.any():
            return None
        return int(np.argmin(np.where(ok, np.abs(delta - self.target_delta), np.inf)))

    def on_snapshot(self, engine: Backtest, t: int) -> None:
        day = engine.day
        # manage open positions
        for tr in list(engine.open_trades):
            leg = tr.legs[0]
            bid = float(day.bid[t, leg.k, leg.right])
            cost = float(tr.entry_prices[0])
            if np.isfinite(bid) and cost > 0 and bid >= self.target_mult * cost:
                engine.close(t, tr, "target")
            elif t >= self.exit_by.get(id(tr), 10**9):
                engine.close(t, tr, "time")
        # new entry
        if not (self.t_first <= t <= self.t_last):
            return
        if len(engine.open_trades) >= self.max_concurrent or not self._gate(day, t):
            return
        k = self._pick(day, t)
        if k is None:
            return
        trade = engine.open(t, [Leg(k, self.side, self.qty)],
                            "compress_put" if self.side == PUT else "compress_call")
        if trade is not None:
            trade.signal_direction = "sell" if self.side == PUT else "buy"
            self.exit_by[id(trade)] = t + self.hold_min * 60 // 5


class SuperTrendCreditSpread(TimedExitMixin, Strategy):
    """SuperTrend signal -> directional 0DTE credit spread (defined risk).

    A bullish flip sells a **bull put credit spread** (sell a ~short_delta put,
    buy a put wing_pts lower) — it profits if price holds above the short put.
    A bearish flip sells a **bear call credit spread** (sell a ~short_delta
    call, buy a call wing_pts higher) — it profits if price stays below the
    short call. So we collect theta in the direction the trend says is safe.

    Managed like the other premium sellers: take profit at profit_frac of the
    credit, stop at stop_mult x credit, optionally reverse the spread on an
    opposite signal, else hold to 15:00 settlement. Two legs, so far less
    spread drag than an iron condor.
    """

    def __init__(self, entry_time="10:00:00", exit_time="15:00:00",
                 atr_period=10, atr_mult=3.0, short_delta=0.20, wing_pts=25.0,
                 stop_mult=2.0, profit_frac=0.5, qty=1, signal_symbol="ES",
                 min_tte_secs=120, reverse_on_opposite=True,
                 min_iv_rv=0.0, rv_window_min=30):
        self.entry_time, self.exit_time = entry_time, exit_time
        self.atr_period, self.atr_mult = int(atr_period), float(atr_mult)
        self.short_delta, self.wing_pts = float(short_delta), float(wing_pts)
        self.stop_mult, self.profit_frac, self.qty = stop_mult, profit_frac, int(qty)
        self.signal_symbol = signal_symbol
        self.min_tte_secs = int(min_tte_secs)
        self.reverse_on_opposite = bool(reverse_on_opposite)
        self.min_iv_rv, self.rv_window_min = float(min_iv_rv), float(rv_window_min)

    def on_day_start(self, engine: Backtest) -> None:
        from ..indicators.supertrend import day_signals
        day = engine.day
        self.by_t: dict[int, list[int]] = {}
        self.t_exit = day.t_index(self.exit_time)
        try:
            sig = day_signals(day.date, self.signal_symbol, self.atr_period,
                              self.atr_mult, self.entry_time, self.exit_time)
        except FileNotFoundError:
            return
        for ts, right in sig["events"]:
            gt = int(np.searchsorted(day.ts, ts))
            if 0 <= gt < len(day.ts):
                self.by_t.setdefault(gt, []).append(right)

    def _bull_put_spread(self, day, t: int) -> list[Leg] | None:
        put_d = np.abs(day.greeks_at(t)["delta"][:, PUT])
        valid = (np.isfinite(put_d) & (day.strikes < day.spot[t]) & (day.bid[t, :, PUT] > 0))
        if not valid.any():
            return None
        ks = int(np.argmin(np.where(valid, np.abs(put_d - self.short_delta), np.inf)))
        kl = int(np.searchsorted(day.strikes, day.strikes[ks] - self.wing_pts, side="right")) - 1
        if kl < 0 or kl >= ks:
            return None
        q = self.qty
        return [Leg(ks, PUT, -q), Leg(kl, PUT, q)]   # short put + long lower put

    def _bear_call_spread(self, day, t: int) -> list[Leg] | None:
        call_d = day.greeks_at(t)["delta"][:, CALL]
        valid = (np.isfinite(call_d) & (day.strikes > day.spot[t]) & (day.bid[t, :, CALL] > 0))
        if not valid.any():
            return None
        ks = int(np.argmin(np.where(valid, np.abs(call_d - self.short_delta), np.inf)))
        kl = int(np.searchsorted(day.strikes, day.strikes[ks] + self.wing_pts))
        if kl >= len(day.strikes) or kl <= ks:
            return None
        q = self.qty
        return [Leg(ks, CALL, -q), Leg(kl, CALL, q)]  # short call + long higher call

    def on_snapshot(self, engine: Backtest, t: int) -> None:
        day = engine.day
        tte = (day.expiry - day.ts[t]) / np.timedelta64(1, "s")
        for right in self.by_t.get(t, []):
            if tte < self.min_tte_secs:
                continue
            if not vol_rich_enough(day, t, self.min_iv_rv, self.rv_window_min):
                continue  # premium too thin to sell here; keep current position
            bullish = right == CALL  # CALL signal == bullish SuperTrend flip
            if self.reverse_on_opposite:
                for tr in list(engine.open_trades):
                    tr_bullish = tr.legs[0].right == PUT  # bull put spread is short a PUT
                    if tr_bullish != bullish:
                        engine.close(t, tr, "reverse")
            legs = self._bull_put_spread(day, t) if bullish else self._bear_call_spread(day, t)
            if legs is None:
                continue
            trade = engine.open(t, legs, "bull_put" if bullish else "bear_call")
            if trade is not None:
                trade.signal_direction = "buy" if bullish else "sell"

        for tr in list(engine.open_trades):
            if self.reverse_on_opposite:
                # pure stop-and-reverse: hold each spread until the opposite
                # flip (handled above) or the exit time / 15:00 settlement —
                # no early profit-target or stop exits, so we are always in a
                # position and reverse on every SuperTrend flip.
                if t >= self.t_exit:
                    engine.close(t, tr, "time")
            else:
                self.manage(engine, t, tr, self.t_exit)


class LastHourGammaExplosion(Strategy):
    """Buy 0DTE options on SuperTrend reversals; ride to a profit multiple or expiry.

    Signal source is a 1-min futures chart (default ES). When SuperTrend flips
    bullish we buy a ~target_delta CALL (long signal, blue); when it flips
    bearish we buy a ~target_delta PUT (short signal, red). Each position is
    held until its bid reaches `target_mult` x our entry cost (profit target)
    or it expires at 15:00 settlement ("dies worthless" if it finishes OTM).
    Signals are only taken inside the [entry_time, exit_time] window — the last
    hour, where 0DTE gamma is largest.

    With `reverse_on_opposite` (default), an opposite-side signal closes the
    open position(s) at market before opening the new one — a stop-and-reverse
    that cuts a losing side on the trend flip instead of letting it bleed to
    expiry. With it off, positions are concurrent and each rides to its own
    profit target or expiry.
    """

    def __init__(self, entry_time="14:00:00", exit_time="15:00:00",
                 atr_period=10, atr_mult=3.0, target_mult=5.0,
                 target_delta=0.20, qty=1, signal_symbol="ES",
                 min_tte_secs=120, reverse_on_opposite=True,
                 invert_signals=False):
        self.entry_time, self.exit_time = entry_time, exit_time
        self.atr_period, self.atr_mult = int(atr_period), float(atr_mult)
        self.target_mult, self.target_delta = float(target_mult), float(target_delta)
        self.qty, self.signal_symbol = int(qty), signal_symbol
        # don't act on a reversal with less than this long to expiry — too
        # little time left to buy a meaningful option (delta selection degenerates)
        self.min_tte_secs = int(min_tte_secs)
        self.reverse_on_opposite = bool(reverse_on_opposite)
        # fade the signal: buy a put on a bullish flip and a call on a bearish
        # flip (tests the inverse hypothesis). Arrow colors follow the option
        # actually traded, so a faded bullish signal shows as a red put.
        self.invert_signals = bool(invert_signals)

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
            if self.invert_signals:        # fade: buy the opposite right
                right = PUT if right == CALL else CALL
            # stop-and-reverse: a new signal closes the opposite side at market
            if self.reverse_on_opposite:
                opp = PUT if right == CALL else CALL
                for trade in list(engine.open_trades):
                    if trade.legs[0].right == opp:
                        engine.close(t, trade, "reverse")
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
