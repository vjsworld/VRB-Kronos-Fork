"""Central configuration: data roots, session times, contract specs.

Everything is naive US/Central time, matching the downloader's storage format.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Data roots (override with env vars if the drives/paths ever move)
# ---------------------------------------------------------------------------
DOWNLOADER_ROOT = Path(
    os.environ.get(
        "VRB_DOWNLOADER_ROOT",
        r"C:\Users\van\source\repos\IBKR Historical Data Downloader",
    )
)
IB_ROOT = Path(os.environ.get("VRB_IB_ROOT", DOWNLOADER_ROOT / "IB"))
THETA_ROOT = Path(os.environ.get("VRB_THETA_ROOT", DOWNLOADER_ROOT / "ThetaData"))

# NPZ cache lives next to this package, outside git (see .gitignore)
CACHE_ROOT = Path(
    os.environ.get("VRB_CACHE_ROOT", Path(__file__).resolve().parent.parent / "vrb_cache")
)

# ---------------------------------------------------------------------------
# IB underlying layout: IB/{SYMBOL}/{SEC_TYPE}/{interval}/{YYYYMMDD}.parquet
# ---------------------------------------------------------------------------
IB_LAYOUT = {
    # symbol: (sec_type, interval, bar_seconds)
    "ES": ("FUT", "1_secs", 1),
    "NQ": ("FUT", "1_secs", 1),
    "SPX": ("IND", "1_secs", 1),
    "NDX": ("IND", "1_secs", 1),
    "VIX": ("IND", "1_min", 60),
}

# ---------------------------------------------------------------------------
# ThetaData layout: ThetaData/{ROOT}/quotes/{YYYYMMDD}.parquet
# ---------------------------------------------------------------------------
OPTION_ROOTS = {
    # option root: (underlying cash symbol, futures symbol, contract multiplier)
    "SPXW": ("SPX", "ES", 100),
    "NDX": ("NDX", "NQ", 100),
}
QUOTE_INTERVAL_SECONDS = 5

# ---------------------------------------------------------------------------
# Session times, naive Central Time
# ---------------------------------------------------------------------------
RTH_START = "08:30:00"   # cash session open
RTH_END = "15:00:00"     # cash session close == SPXW/NDXP 0DTE expiry print
EXPIRY_TIME = "15:00:00"

# Risk-free rate used for IV/greeks. For 0DTE the discounting effect is
# negligible; a flat recent T-bill-ish rate is plenty.
RISK_FREE_RATE = 0.04

# ---------------------------------------------------------------------------
# Backtest cost model defaults
# ---------------------------------------------------------------------------
COMMISSION_PER_CONTRACT = 0.65   # broker commission, each way
EXCHANGE_FEES_PER_CONTRACT = 0.65  # CBOE/ORF/SEC-ish bundle for SPX options
# Fill model: cross the spread, improved by this fraction of half-spread
# (0.0 = fill exactly at bid/ask, 1.0 = fill at mid)
FILL_SPREAD_IMPROVEMENT = 0.0
