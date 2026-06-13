"""Premium-expansion event study: find every 10x option move, mine its preconditions.

The 4D matrix is (day, time, strike, right) — each day's (T,K,2) mid grid from
the NPZ cache. The scanner computes, fully vectorized, the maximum forward
mid ratio for every possible anchor and extracts contract-days that achieved
RATIO_MIN with a tradeable entry (two-sided market, mid >= MIN_ENTRY_MID,
anchored before LATEST_ANCHOR).

For anomaly mining we snapshot point-in-time features at each event anchor and
at premium-matched control anchors that did NOT expand, then compare (lift).
Forward descriptors (minutes to 10x, peak path) are kept separate from
features — they describe events, they must never leak into preconditions.
"""

from __future__ import annotations

import numpy as np

from ..data import ib
from ..options import bs
from ..options.chain import CALL, PUT, ChainDay
from ..util.timing import Timer, get_logger

log = get_logger("research.expansion")

# Defaults — every threshold is an optimizable parameter (CLI-overridable):
# the expansion ratio, the minimum tradeable entry mid, the latest anchor time,
# and the control-matching band are all swept in later experiments.
RATIO_MIN = 10.0
MIN_ENTRY_MID = 0.15
LATEST_ANCHOR = "14:30:00"
GRID_SECS = 5

EVENT_FIELDS = [
    "date", "right", "strike", "anchor_t", "entry_mid", "peak_mid", "ratio",
    "min_to_10x", "min_to_peak", "is_event",
    # point-in-time features at anchor:
    "tod_min", "tte_min", "moneyness_pct", "abs_delta", "spread_pts", "spread_pct",
    "ret_5m", "ret_15m", "ret_30m", "rv_5m", "rv_15m", "rv_30m",
    "range_pos", "ret_from_open", "vix", "vix_chg_15m", "atm_iv", "iv_rv_15m",
    "skew_pc", "es_overnight_gap_pct", "book_imb", "with_move",
]


def _suffix_max(a: np.ndarray) -> np.ndarray:
    """fmax[t] = max(a[t:]) along axis 0, NaN-safe."""
    b = np.where(np.isfinite(a), a, -np.inf)
    return np.maximum.accumulate(b[::-1], axis=0)[::-1]


def scan_day(day: ChainDay, ratio_min: float = RATIO_MIN,
             min_entry_mid: float = MIN_ENTRY_MID,
             latest_anchor: str = LATEST_ANCHOR) -> list[dict]:
    """All valid anchors for one day -> per-contract best expansion record."""
    mid = day.mid  # (T,K,2), NaN where one-sided
    t_max = day.t_index(latest_anchor)
    fmax = _suffix_max(mid)
    fmax_next = np.full_like(mid, -np.inf)
    fmax_next[:-1] = fmax[1:]

    valid = (np.isfinite(mid) & (mid >= min_entry_mid)
             & (day.bid > 0) & (day.ask > 0))
    valid[t_max:] = False
    ratio = np.where(valid & (mid > 0), fmax_next / mid, np.nan)

    out = []
    best_ratio = np.nanmax(np.where(np.isfinite(ratio), ratio, np.nan), axis=0)  # (K,2)
    for k in range(mid.shape[1]):
        for r in (CALL, PUT):
            br = best_ratio[k, r]
            if not np.isfinite(br):
                continue
            col = ratio[:, k, r]
            t0 = int(np.nanargmax(col))
            entry = float(mid[t0, k, r])
            # forward path descriptors
            fut = mid[t0 + 1:, k, r]
            finite_fut = np.where(np.isfinite(fut), fut, -np.inf)
            t_peak = int(np.argmax(finite_fut)) + t0 + 1
            cross = np.flatnonzero(finite_fut >= ratio_min * entry)
            out.append({
                "date": day.date, "right": r, "strike": float(day.strikes[k]),
                "anchor_t": t0, "anchor_dt": str(day.ts[t0]),
                "peak_dt": str(day.ts[t_peak]), "entry_mid": entry,
                "peak_mid": float(mid[t_peak, k, r]), "ratio": float(br),
                "min_to_10x": (float(cross[0] + 1) * GRID_SECS / 60.0) if cross.size else np.nan,
                "min_to_peak": float(t_peak - t0) * GRID_SECS / 60.0,
                "is_event": bool(br >= ratio_min),
            })
    return out


# ------------------------------------------------------------------ features
def day_context(day: ChainDay) -> dict:
    """Precompute per-day series used by anchor snapshots."""
    spot = day.spot
    logp = np.log(np.where(np.isfinite(spot) & (spot > 0), spot, np.nan))
    open_t = day.t_index("08:30:05")

    def trail_ret(steps):
        r = np.full(len(spot), np.nan)
        r[steps:] = logp[steps:] - logp[:-steps]
        return r

    def trail_rv(steps):
        d = np.diff(logp, prepend=np.nan)
        out = np.full(len(spot), np.nan)
        c = np.nancumsum(d ** 2)
        n = np.cumsum(np.isfinite(d))
        out[steps:] = np.sqrt(np.maximum(c[steps:] - c[:-steps], 0)
                              / np.maximum(n[steps:] - n[:-steps], 1))
        return out * np.sqrt(bs.SECONDS_PER_YEAR / GRID_SECS)

    run_max = np.fmax.accumulate(np.where(np.isfinite(spot), spot, -np.inf))
    run_min = np.fmin.accumulate(np.where(np.isfinite(spot), spot, np.inf))

    # ES overnight gap: ES at cash open vs ES at prior cash close (ES file
    # includes the overnight session; first bars are 17:00 prior day)
    es_gap = np.nan
    try:
        es = ib.load_day("ES", day.date)
        iso = f"{day.date[:4]}-{day.date[4:6]}-{day.date[6:]}"
        e_open = np.datetime64(f"{iso}T08:30:00", "s")
        i_open = int(np.searchsorted(es["ts"], e_open, "right")) - 1
        if i_open > 60:
            es_gap = float(np.log(es["close"][i_open] / es["close"][0]))
    except (FileNotFoundError, ValueError):
        pass

    return {"spot": spot, "open_t": open_t,
            "ret_5m": trail_ret(60), "ret_15m": trail_ret(180), "ret_30m": trail_ret(360),
            "rv_5m": trail_rv(60), "rv_15m": trail_rv(180), "rv_30m": trail_rv(360),
            "run_max": run_max, "run_min": run_min, "es_gap": es_gap}


def snapshot(day: ChainDay, ctx: dict, rec: dict) -> dict:
    """Point-in-time features at rec['anchor_t'] for contract (strike,right)."""
    t = rec["anchor_t"]
    k = day.k_index(rec["strike"])
    r = rec["right"]
    s = ctx["spot"][t]
    bid, ask = float(day.bid[t, k, r]), float(day.ask[t, k, r])
    mid = rec["entry_mid"]
    iv = day.atm_iv(t)
    g = bs.implied_vol(day.mid[t, k, r], s, day.strikes[k], day.tau[t], r, r=0.04)
    delta = bs.greeks(s, day.strikes[k], day.tau[t], g, r, r=0.04)["delta"] if np.isfinite(g) else np.nan
    rv15 = ctx["rv_15m"][t]
    rng = ctx["run_max"][t] - ctx["run_min"][t]
    ret15 = ctx["ret_15m"][t]
    # skew: put IV 0.5% below spot minus call IV 0.5% above
    kp = int(np.argmin(np.abs(day.strikes - s * 0.995)))
    kc = int(np.argmin(np.abs(day.strikes - s * 1.005)))
    ivp = bs.implied_vol(day.mid[t, kp, PUT], s, day.strikes[kp], day.tau[t], PUT, r=0.04)
    ivc = bs.implied_vol(day.mid[t, kc, CALL], s, day.strikes[kc], day.tau[t], CALL, r=0.04)

    rec.update({
        "tod_min": (t - ctx["open_t"]) * GRID_SECS / 60.0,
        "tte_min": float(day.tau[t]) * 365 * 24 * 60,
        "moneyness_pct": float((day.strikes[k] - s) / s * 100) if np.isfinite(s) else np.nan,
        "abs_delta": float(abs(delta)) if np.isfinite(delta) else np.nan,
        "spread_pts": ask - bid,
        "spread_pct": (ask - bid) / mid if mid > 0 else np.nan,
        "ret_5m": float(ctx["ret_5m"][t]), "ret_15m": float(ret15),
        "ret_30m": float(ctx["ret_30m"][t]),
        "rv_5m": float(ctx["rv_5m"][t]), "rv_15m": float(rv15),
        "rv_30m": float(ctx["rv_30m"][t]),
        "range_pos": float((s - ctx["run_min"][t]) / rng) if rng > 0 else 0.5,
        "ret_from_open": float(np.log(s / ctx["spot"][ctx["open_t"]]))
            if np.isfinite(ctx["spot"][ctx["open_t"]]) and np.isfinite(s) else np.nan,
        "vix": float(day.vix[t]),
        "vix_chg_15m": float(day.vix[t] - day.vix[max(0, t - 180)]),
        "atm_iv": float(iv),
        "iv_rv_15m": float(iv / rv15) if np.isfinite(iv) and np.isfinite(rv15) and rv15 > 0 else np.nan,
        "skew_pc": float(ivp - ivc) if np.isfinite(ivp) and np.isfinite(ivc) else np.nan,
        "es_overnight_gap_pct": ctx["es_gap"] * 100 if np.isfinite(ctx["es_gap"]) else np.nan,
        "book_imb": float(day.bid_size[t, k, r] - day.ask_size[t, k, r]),
        # is the option's direction aligned with the trailing 15m move?
        "with_move": float(np.sign(ret15) * (1 if r == CALL else -1)) if np.isfinite(ret15) else np.nan,
    })
    return rec


def study_day(date: str, root: str = "SPXW", controls_per_event: int = 3,
              rng_seed: int = 7, ratio_min: float = RATIO_MIN,
              min_entry_mid: float = MIN_ENTRY_MID,
              latest_anchor: str = LATEST_ANCHOR) -> list[dict]:
    """Scan one day; snapshot features for events + premium-matched controls."""
    day = ChainDay.load(date, root)
    recs = scan_day(day, ratio_min, min_entry_mid, latest_anchor)
    events = [x for x in recs if x["is_event"]]
    non = [x for x in recs if not x["is_event"] and np.isfinite(x["ratio"])]
    rng = np.random.default_rng(rng_seed + int(date))
    controls: list[dict] = []
    if events and non:
        prem = np.array([x["entry_mid"] for x in non])
        for ev in events:
            band = (prem >= ev["entry_mid"] * 0.5) & (prem <= ev["entry_mid"] * 2.0)
            pool = np.flatnonzero(band)
            if pool.size == 0:
                pool = np.arange(len(non))
            for j in rng.choice(pool, size=min(controls_per_event, pool.size), replace=False):
                # controls are anchored at their own best-expansion anchor —
                # symmetric with events (both are each contract's "best case")
                controls.append(dict(non[j]))
    ctx = day_context(day)
    out = []
    for rec in events + controls:
        try:
            out.append(snapshot(day, ctx, dict(rec)))
        except (KeyError, IndexError):
            continue
    return out
