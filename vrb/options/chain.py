"""ChainDay: one trade date's 0DTE option chain + aligned underlying state.

Wraps the dense (T, K, 2) quote grid from vrb.data.theta with the SPX/VIX
series asof-joined onto the same 5-sec timestamps, plus IV/greeks helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ..config import EXPIRY_TIME, OPTION_ROOTS, RISK_FREE_RATE
from ..data import ib, theta
from . import bs

CALL, PUT = theta.CALL, theta.PUT


@dataclass
class ChainDay:
    root: str
    date: str                  # YYYYMMDD trade date == expiration date
    ts: np.ndarray             # (T,) datetime64[s]
    strikes: np.ndarray        # (K,) float64
    bid: np.ndarray            # (T,K,2) float32
    ask: np.ndarray            # (T,K,2) float32
    bid_size: np.ndarray       # (T,K,2) int32
    ask_size: np.ndarray       # (T,K,2) int32
    spot: np.ndarray           # (T,) float64 cash index, asof
    vix: np.ndarray            # (T,) float64, asof
    multiplier: int = 100

    @classmethod
    def load(cls, date: str, root: str = "SPXW") -> "ChainDay":
        cash_sym, _fut, mult = OPTION_ROOTS[root]
        grid = theta.load_chain_day(root, date)
        cash = ib.load_day(cash_sym, date)
        vix = ib.load_day("VIX", date)
        return cls(
            root=root,
            date=date,
            ts=grid["ts"],
            strikes=grid["strikes"],
            bid=grid["bid"],
            ask=grid["ask"],
            bid_size=grid["bid_size"],
            ask_size=grid["ask_size"],
            spot=ib.asof(cash["ts"], cash["close"], grid["ts"]),
            vix=ib.asof(vix["ts"], vix["close"], grid["ts"]),
            multiplier=mult,
        )

    # ------------------------------------------------------------------ time
    @cached_property
    def expiry(self) -> np.datetime64:
        iso = f"{self.date[:4]}-{self.date[4:6]}-{self.date[6:]}"
        return np.datetime64(f"{iso}T{EXPIRY_TIME}", "s")

    @cached_property
    def tau(self) -> np.ndarray:
        """(T,) time to expiry in years (>= 0)."""
        secs = (self.expiry - self.ts).astype(np.int64)
        return np.maximum(secs, 0) / bs.SECONDS_PER_YEAR

    @cached_property
    def settlement(self) -> float:
        """Cash settlement print: last spot at/before expiry."""
        return float(self.spot[self.ts <= self.expiry][-1])

    # ---------------------------------------------------------------- quotes
    @cached_property
    def mid(self) -> np.ndarray:
        """(T,K,2) mid price; NaN where there is no two-sided market."""
        b = self.bid.astype(np.float64)
        a = self.ask.astype(np.float64)
        m = 0.5 * (b + a)
        return np.where((a > 0) & (b >= 0), m, np.nan)

    def t_index(self, hhmmss: str) -> int:
        """Grid index of the first snapshot at/after a CT wall-clock time."""
        iso = f"{self.date[:4]}-{self.date[4:6]}-{self.date[6:]}T{hhmmss}"
        return int(np.searchsorted(self.ts, np.datetime64(iso, "s")))

    def k_index(self, strike: float) -> int:
        i = int(np.searchsorted(self.strikes, strike))
        if i >= len(self.strikes) or self.strikes[i] != strike:
            raise KeyError(f"strike {strike} not in chain")
        return i

    def atm_k(self, t: int) -> int:
        """Strike index closest to spot at snapshot t."""
        return int(np.argmin(np.abs(self.strikes - self.spot[t])))

    # ------------------------------------------------------------- analytics
    def iv_at(self, t: int) -> np.ndarray:
        """(K,2) implied vol from mid at snapshot t."""
        return bs.implied_vol(
            self.mid[t], self.spot[t], self.strikes[:, None], self.tau[t],
            np.array([CALL, PUT]), r=RISK_FREE_RATE,
        )

    def greeks_at(self, t: int, iv: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """(K,2) greeks at snapshot t (computes IV if not supplied)."""
        if iv is None:
            iv = self.iv_at(t)
        return bs.greeks(
            self.spot[t], self.strikes[:, None], self.tau[t], iv,
            np.array([CALL, PUT]), r=RISK_FREE_RATE,
        )

    def atm_iv(self, t: int) -> float:
        """Mean of call/put IV at the ATM strike (NaN-safe)."""
        k = self.atm_k(t)
        iv = bs.implied_vol(
            self.mid[t, k], self.spot[t], self.strikes[k], self.tau[t],
            np.array([CALL, PUT]), r=RISK_FREE_RATE,
        )
        return float(np.nanmean(iv))

    def intrinsic(self, strike_idx: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Settlement intrinsic value for given legs."""
        k = self.strikes[strike_idx]
        s = self.settlement
        return np.where(right == CALL, np.maximum(s - k, 0.0), np.maximum(k - s, 0.0))
