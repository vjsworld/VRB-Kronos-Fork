"""Kronos foundation-model forecast features for the 0DTE pipeline.

Feeds ES 1-min bars (resampled from the IB 1-sec data — the overnight Globex
session gives a full 400-bar context even right at the cash open) into the
pre-trained Kronos model, and turns the sampled forecast path into two
features per snapshot:

  kronos_exp_ret  - mean forecast log-return over the horizon
  kronos_path_vol - stdev of forecast 1-min returns (annualized)

All sample times for a day are batched through predict_batch in one GPU pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import ib

KRONOS_FEATURE_NAMES = ["kronos_exp_ret", "kronos_path_vol"]


def resample_1min(bars: dict[str, np.ndarray]) -> pd.DataFrame:
    """1-sec bar-close rows -> 1-min OHLCV labeled at minute close (point-in-time)."""
    ts = bars["ts"].astype("datetime64[s]")
    # bar stamped at close: second :00 belongs to the minute that just ended
    minute = (ts - np.timedelta64(1, "s")).astype("datetime64[m]") + np.timedelta64(1, "m")
    df = pd.DataFrame({
        "minute": minute.astype("datetime64[ns]"),
        "open": bars["open"], "high": bars["high"],
        "low": bars["low"], "close": bars["close"], "volume": bars["volume"],
    })
    g = df.groupby("minute", sort=True)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    out["amount"] = out["volume"] * out["close"]
    return out.reset_index().rename(columns={"minute": "timestamps"})


class KronosForecaster:
    def __init__(self, model_name: str = "NeoQuasar/Kronos-small",
                 tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
                 device: str | None = None, lookback: int = 400,
                 pred_len: int = 15, temperature: float = 1.0,
                 top_p: float = 0.9, sample_count: int = 1):
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        model = Kronos.from_pretrained(model_name)
        self.predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        self.lookback, self.pred_len = lookback, pred_len
        self.temperature, self.top_p, self.sample_count = temperature, top_p, sample_count

    def day_features(self, date: str, grid_ts: np.ndarray,
                     t_indices: np.ndarray, fut_symbol: str = "ES") -> np.ndarray:
        """(len(t_indices), 2) Kronos features for the given snapshot indices."""
        bars = resample_1min(ib.load_day(fut_symbol, date))
        bar_ts = bars["timestamps"].to_numpy().astype("datetime64[s]")

        df_list, x_ts_list, y_ts_list, ok = [], [], [], []
        for i, t in enumerate(t_indices):
            snap = grid_ts[t]
            end = int(np.searchsorted(bar_ts, snap, side="right"))
            start = end - self.lookback
            if start < 0:
                ok.append(False)
                continue
            window = bars.iloc[start:end]
            df_list.append(window[["open", "high", "low", "close", "volume", "amount"]])
            x_ts_list.append(pd.Series(window["timestamps"].values))
            future = pd.date_range(
                pd.Timestamp(bar_ts[end - 1]) + pd.Timedelta(minutes=1),
                periods=self.pred_len, freq="min")
            y_ts_list.append(pd.Series(future))
            ok.append(True)

        out = np.full((len(t_indices), 2), np.nan, np.float64)
        if not df_list:
            return out
        preds = self.predictor.predict_batch(
            df_list, x_ts_list, y_ts_list, pred_len=self.pred_len,
            T=self.temperature, top_p=self.top_p,
            sample_count=self.sample_count, verbose=False)

        j = 0
        for i, is_ok in enumerate(ok):
            if not is_ok:
                continue
            pred_close = preds[j]["close"].to_numpy(np.float64)
            last_close = float(df_list[j]["close"].iloc[-1])
            r = np.diff(np.log(np.concatenate([[last_close], pred_close])))
            out[i, 0] = float(np.log(pred_close[-1] / last_close))
            out[i, 1] = float(r.std() * np.sqrt(365 * 24 * 60))
            j += 1
        return out
