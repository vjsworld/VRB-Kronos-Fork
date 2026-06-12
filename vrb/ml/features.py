"""Snapshot feature engineering for 0DTE prediction modeling.

For each sampled grid time t we emit market-state features and forward-looking
targets. Everything is strictly point-in-time: features only use data at or
before t; targets only use data after t.
"""

from __future__ import annotations

import numpy as np

from ..config import RISK_FREE_RATE
from ..options import bs
from ..options.chain import CALL, PUT, ChainDay

FEATURE_NAMES = [
    "mins_since_open", "tau_hrs",
    "ret_1m", "ret_5m", "ret_15m",
    "rvol_5m", "rvol_15m",
    "range_pos", "dist_open",
    "vix", "vix_chg_5m",
    "atm_iv", "iv_chg_5m", "rv_iv_ratio",
    "skew_pc", "straddle_pct",
    "atm_spread", "flow_imb_call", "flow_imb_put",
]
TARGET_NAMES = ["fwd_ret", "fwd_straddle_ret"]

GRID_SECS = 5


def _log_ret(spot: np.ndarray, t: int, lag_steps: int) -> float:
    j = t - lag_steps
    if j < 0 or not (np.isfinite(spot[t]) and np.isfinite(spot[j])) or spot[j] <= 0:
        return np.nan
    return float(np.log(spot[t] / spot[j]))


def _rvol(spot: np.ndarray, t: int, window_steps: int) -> float:
    """Annualized realized vol from 5-sec log returns over the window."""
    j = max(0, t - window_steps)
    s = spot[j:t + 1]
    s = s[np.isfinite(s)]
    if len(s) < 10:
        return np.nan
    r = np.diff(np.log(s))
    return float(r.std() * np.sqrt(bs.SECONDS_PER_YEAR / GRID_SECS))


def _iv_near(day: ChainDay, t: int, money: float, right: int) -> float:
    """IV at the strike nearest spot*money for one right."""
    if not np.isfinite(day.spot[t]):
        return np.nan
    k = int(np.argmin(np.abs(day.strikes - day.spot[t] * money)))
    iv = bs.implied_vol(day.mid[t, k, right], day.spot[t], day.strikes[k],
                        day.tau[t], right, r=RISK_FREE_RATE)
    return float(iv)


def day_features(
    day: ChainDay,
    every_secs: int = 60,
    horizon_secs: int = 900,
    start: str = "09:00:00",
    end: str = "14:30:00",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one day -> (X, Y, t_indices). Rows with NaNs are kept; the
    dataset assembler filters them so per-feature debugging stays possible."""
    step = every_secs // GRID_SECS
    hsteps = horizon_secs // GRID_SECS
    t0, t1 = day.t_index(start), day.t_index(end)
    samples = range(t0, min(t1, len(day.ts) - hsteps - 1), step)

    spot, vix = day.spot, day.vix
    day_open_t = day.t_index("08:30:05")

    rows_x, rows_y, t_idx = [], [], []
    atm_iv_hist: dict[int, float] = {}

    for t in samples:
        s = spot[t]
        if not np.isfinite(s):
            continue
        k = day.atm_k(t)
        atm_iv = day.atm_iv(t)
        atm_iv_hist[t] = atm_iv
        # earliest sampled IV within the trailing 5 minutes
        prev_keys = [tt for tt in atm_iv_hist if 0 < t - tt <= 300 // GRID_SECS]
        iv_5m_ago = atm_iv_hist[min(prev_keys)] if prev_keys else np.nan

        seen = spot[day_open_t:t + 1]
        seen = seen[np.isfinite(seen)]
        hi, lo = (seen.max(), seen.min()) if len(seen) else (np.nan, np.nan)
        rng = hi - lo
        straddle = day.mid[t, k, CALL] + day.mid[t, k, PUT]
        rv5 = _rvol(spot, t, 60)
        vix_now = vix[t]
        vix_5m = vix[max(0, t - 60)]

        x = [
            (t - day_open_t) * GRID_SECS / 60.0,
            day.tau[t] * 365 * 24,
            _log_ret(spot, t, 12), _log_ret(spot, t, 60), _log_ret(spot, t, 180),
            rv5, _rvol(spot, t, 180),
            (s - lo) / rng if rng and np.isfinite(rng) and rng > 0 else 0.5,
            float(np.log(s / seen[0])) if len(seen) else np.nan,
            vix_now, vix_now - vix_5m if np.isfinite(vix_now) and np.isfinite(vix_5m) else np.nan,
            atm_iv,
            atm_iv - iv_5m_ago if np.isfinite(iv_5m_ago) else np.nan,
            rv5 / atm_iv if np.isfinite(rv5) and np.isfinite(atm_iv) and atm_iv > 0 else np.nan,
            _iv_near(day, t, 0.995, PUT) - _iv_near(day, t, 1.005, CALL),
            float(straddle / s) if np.isfinite(straddle) else np.nan,
            float(day.ask[t, k, CALL] - day.bid[t, k, CALL]),
            float(np.nansum(day.bid_size[t, max(0, k - 5):k + 6, CALL])
                  - np.nansum(day.ask_size[t, max(0, k - 5):k + 6, CALL])),
            float(np.nansum(day.bid_size[t, max(0, k - 5):k + 6, PUT])
                  - np.nansum(day.ask_size[t, max(0, k - 5):k + 6, PUT])),
        ]

        tf = t + hsteps
        fwd_ret = _log_ret(spot, tf, hsteps)
        straddle_fwd = day.mid[tf, k, CALL] + day.mid[tf, k, PUT]
        fwd_straddle = (float(straddle_fwd / straddle) - 1.0
                        if np.isfinite(straddle_fwd) and np.isfinite(straddle) and straddle > 0
                        else np.nan)
        rows_x.append(x)
        rows_y.append([fwd_ret, fwd_straddle])
        t_idx.append(t)

    return (np.array(rows_x, np.float64),
            np.array(rows_y, np.float64),
            np.array(t_idx, np.int64))


def build_dataset(
    dates: list[str],
    root: str = "SPXW",
    every_secs: int = 60,
    horizon_secs: int = 900,
    verbose: bool = True,
) -> dict:
    """Stack day_features over many dates -> dict with X, Y, days, t_idx."""
    xs, ys, ds, ts = [], [], [], []
    for d in dates:
        try:
            day = ChainDay.load(d, root)
            X, Y, t_idx = day_features(day, every_secs, horizon_secs)
        except (FileNotFoundError, ValueError) as e:
            if verbose:
                print(f"  {d}: skipped ({e.__class__.__name__})")
            continue
        keep = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        xs.append(X[keep]); ys.append(Y[keep])
        ds.extend([d] * int(keep.sum())); ts.append(t_idx[keep])
        if verbose:
            print(f"  {d}: {int(keep.sum())} samples")
    return {
        "X": np.vstack(xs), "Y": np.vstack(ys),
        "days": np.array(ds), "t_idx": np.concatenate(ts),
        "feature_names": FEATURE_NAMES, "target_names": TARGET_NAMES,
    }
