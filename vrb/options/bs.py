"""Vectorized Black-Scholes pricing, greeks, and implied vol.

All functions broadcast over numpy arrays. `right` follows theta.py's
convention: 0=CALL, 1=PUT. Time `tau` is in years (calendar time); for 0DTE
that is seconds-to-expiry / 31_536_000.
"""

from __future__ import annotations

import numpy as np

SECONDS_PER_YEAR = 365.0 * 24 * 3600
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from math import sqrt
    return 0.5 * (1.0 + _erf(x / sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26, max abs error 1.5e-7 — plenty for IV work
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
           + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-x * x))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def price(S, K, tau, sigma, right, r: float = 0.0):
    """Black-Scholes price. Handles tau<=0 / sigma<=0 as intrinsic value."""
    S, K, tau, sigma, right = np.broadcast_arrays(
        *(np.asarray(a, np.float64) for a in (S, K, tau, sigma)), np.asarray(right))
    intrinsic = np.where(right == 0, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    live = (tau > 0) & (sigma > 0) & (S > 0) & (K > 0)
    S_ = np.where(live, S, 1.0)
    K_ = np.where(live, K, 1.0)
    tau_ = np.where(live, tau, 1.0)
    sig_ = np.where(live, sigma, 1.0)
    sqrt_t = np.sqrt(tau_)
    d1 = (np.log(S_ / K_) + (r + 0.5 * sig_ * sig_) * tau_) / (sig_ * sqrt_t)
    d2 = d1 - sig_ * sqrt_t
    disc = np.exp(-r * tau_)
    call = S_ * _norm_cdf(d1) - K_ * disc * _norm_cdf(d2)
    put = K_ * disc * _norm_cdf(-d2) - S_ * _norm_cdf(-d1)
    val = np.where(right == 0, call, put)
    return np.where(live, val, intrinsic)


def greeks(S, K, tau, sigma, right, r: float = 0.0) -> dict[str, np.ndarray]:
    """delta, gamma, vega (per 1.00 vol), theta (per day)."""
    S, K, tau, sigma, right = np.broadcast_arrays(
        *(np.asarray(a, np.float64) for a in (S, K, tau, sigma)), np.asarray(right))
    live = (tau > 0) & (sigma > 0) & (S > 0) & (K > 0)
    S_ = np.where(live, S, 1.0)
    K_ = np.where(live, K, 1.0)
    tau_ = np.where(live, tau, 1.0)
    sig_ = np.where(live, sigma, 1.0)
    sqrt_t = np.sqrt(tau_)
    d1 = (np.log(S_ / K_) + (r + 0.5 * sig_ * sig_) * tau_) / (sig_ * sqrt_t)
    d2 = d1 - sig_ * sqrt_t
    disc = np.exp(-r * tau_)
    pdf = _norm_pdf(d1)

    delta = np.where(right == 0, _norm_cdf(d1), _norm_cdf(d1) - 1.0)
    gamma = pdf / (S_ * sig_ * sqrt_t)
    vega = S_ * pdf * sqrt_t
    theta_call = -S_ * pdf * sig_ / (2 * sqrt_t) - r * K_ * disc * _norm_cdf(d2)
    theta_put = -S_ * pdf * sig_ / (2 * sqrt_t) + r * K_ * disc * _norm_cdf(-d2)
    theta = np.where(right == 0, theta_call, theta_put) / 365.0

    # Live contract with unknown vol (NaN sigma, e.g. unquoted strike): NaN
    # greeks, so downstream isfinite filters exclude it. Expired/degenerate:
    # delta is the intrinsic indicator, the rest 0.
    undef = np.isnan(sigma) & (tau > 0)
    exp_delta = np.where(right == 0, (S > K).astype(np.float64), -(S < K).astype(np.float64))
    return {
        "delta": np.where(undef, np.nan, np.where(live, delta, exp_delta)),
        "gamma": np.where(undef, np.nan, np.where(live, gamma, 0.0)),
        "vega": np.where(undef, np.nan, np.where(live, vega, 0.0)),
        "theta": np.where(undef, np.nan, np.where(live, theta, 0.0)),
    }


def implied_vol(target, S, K, tau, right, r: float = 0.0,
                lo: float = 1e-4, hi: float = 10.0, iters: int = 60):
    """Implied vol via vectorized bisection (monotone in sigma, always converges).

    Returns NaN where the target price is outside the no-arbitrage range
    (below intrinsic or above the sigma=hi price) or inputs are degenerate.
    """
    target, S, K, tau, right = np.broadcast_arrays(
        *(np.asarray(a, np.float64) for a in (target, S, K, tau)), np.asarray(right))
    # the sigma->0 BS floor is the DISCOUNTED intrinsic (K*exp(-r*tau) vs K):
    # ITM European puts legitimately quote below undiscounted intrinsic, and
    # ITM call targets below the discounted floor are unsolvable.
    disc_K = K * np.exp(-r * np.maximum(tau, 0.0))
    floor = np.where(right == 0, np.maximum(S - disc_K, 0.0), np.maximum(disc_K - S, 0.0))
    valid = (tau > 0) & (S > 0) & (K > 0) & (target >= floor - 1e-12) & np.isfinite(target)
    valid &= target <= price(S, K, tau, np.full_like(S, hi), right, r) + 1e-12

    lo_a = np.full(S.shape, lo)
    hi_a = np.full(S.shape, hi)
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        too_low = price(S, K, tau, mid, right, r) < target
        lo_a = np.where(too_low, mid, lo_a)
        hi_a = np.where(too_low, hi_a, mid)
    iv = 0.5 * (lo_a + hi_a)
    return np.where(valid, iv, np.nan)
