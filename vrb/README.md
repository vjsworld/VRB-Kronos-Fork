# VRB — 0DTE Option Backtester, Forecaster & Prediction Modeling

Built on top of the [Kronos](https://github.com/shiyu-coder/Kronos) financial
foundation model fork, driven by two local data feeds that update daily:

| Source | Data | Resolution | Location |
|---|---|---|---|
| IB TWS | ES, NQ futures; SPX, NDX cash OHLCV | 1 sec | `IBKR Historical Data Downloader/IB/` |
| IB TWS | VIX OHLCV | 1 min | `IB/VIX/IND/1_min/` |
| ThetaData | SPXW (NDX soon) 0DTE NBBO quotes, all strikes | 5 sec | `ThetaData/{ROOT}/quotes/` |

All timestamps are **naive US/Central**; same timestamp = same instant in every
file (IB bars are stored bar-close-stamped). See the downloader repo's
`understanding_this_data.md` for the full conventions.

## Architecture

```
vrb/
  config.py               paths, sessions, cost model (env-var overridable)
  data/
    cache.py              parquet -> NPZ cache, fingerprinted by source mtime+size
    ib.py                 underlying bar loaders + asof joins
    theta.py              8M-row quote parquet -> dense (T,K,2) numpy grid
    calendar.py           tradeable dates = files present on disk
  options/
    bs.py                 vectorized Black-Scholes price/greeks/IV (bisection)
    chain.py              ChainDay: quotes + spot + VIX + tau + settlement
  backtest/
    engine.py             5-sec event loop, NBBO fills, multi-leg, cash settlement
    strategies.py         ShortStraddle, IronCondor, SignalDirectional
    stats.py              multi-day runner + summary stats
  ml/
    features.py           19 point-in-time snapshot features, 2 targets
    forecaster.py         walk-forward HistGradientBoosting + IC metrics
    kronos_forecaster.py  Kronos forecast features from ES 1-min context (GPU)
  scripts/
    build_cache.py        pre-warm NPZ caches
    run_backtest.py       strategy backtests from the CLI
    train_forecaster.py   features -> model -> OOS signal backtest, end to end
```

The NPZ cache (`vrb_cache/`, gitignored) makes warm loads ~30x faster than the
parquet pivot (0.08s vs 2.6s per option day). Caches self-invalidate when the
daily downloaders rewrite a parquet file.

## Quick start

```powershell
.\.venv\Scripts\Activate.ps1

# warm the cache for the last 90 days (one-time, ~5 min)
python -m vrb.scripts.build_cache --last 90

# classic premium-selling backtests
python -m vrb.scripts.run_backtest --strategy straddle --last 30
python -m vrb.scripts.run_backtest --strategy condor --last 60 --delta 0.10 --wing 50

# ML: build features, walk-forward train, backtest the signal out-of-sample
python -m vrb.scripts.train_forecaster --last 90
# same but with Kronos foundation-model forecast features (GPU)
python -m vrb.scripts.train_forecaster --last 90 --with-kronos --every 300
```

## Key design decisions

- **Fills**: buys lift the actual NBBO ask, sells hit the bid
  (`FILL_SPREAD_IMPROVEMENT` in config moves fills toward mid if you believe
  you get price improvement). Commission + exchange fees per contract per side.
- **Settlement**: positions held to 15:00 CT cash-settle at intrinsic against
  the last SPX print — like real SPXW.
- **Point-in-time discipline**: features at time t use only data ≤ t; targets
  use only data > t; walk-forward splits are by whole trade date.
- **Kronos features**: ES 1-min bars give a continuous overnight context, so
  the model has a full 400-bar window even at the 08:30 cash open. All sample
  times for a day run through `predict_batch` in one GPU pass.

## Extending

- **NDX**: once the ThetaData NDX download finishes, everything works by
  passing `--root NDX` (`OPTION_ROOTS` already maps NDX -> NQ futures).
- **New strategies**: subclass `vrb.backtest.strategies.Strategy`, implement
  `on_snapshot(engine, t)`, trade via `engine.open/close`.
- **New features**: add to `vrb/ml/features.py` and `FEATURE_NAMES`.
- **Fine-tuning Kronos** on your own ES/SPX data: see `finetune/` in the repo
  root — the qlib pipeline can be adapted to the IB parquet data.
