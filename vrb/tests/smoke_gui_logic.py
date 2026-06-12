"""Headless smoke of the GUI-feeding logic: payloads, report, ML pipeline."""
from vrb.backtest.payload import run_backtest_days
from vrb.backtest.strategies import ShortStraddle
from vrb.data.calendar import common_dates
from vrb.gui.report import compute_report, monthly_table

dates = common_dates("SPXW")[-6:]
payloads = run_backtest_days(dates, lambda: ShortStraddle(), "SPXW", lambda m: None)
print(f"payloads: {len(payloads)} days, keys={sorted(payloads[0])}")

rep = compute_report(payloads)
print(f"report rows: {len(rep['rows'])}")
for kind, label, a, b, s in rep["rows"]:
    if kind == "section":
        print(f"  [{label}]")
    elif label.strip() in ("Total Net Profit", "Profit Factor", "Percent Profitable",
                            "Max. Drawdown (Intraday Peak to Valley)", "Sharpe Ratio (daily, annualized)"):
        print(f"      {label.strip():42s} all={a:>14} buys={b:>10} sells={s:>10}")
print("monthly:", monthly_table(payloads))

# ML pipeline end-to-end (small, fast)
from vrb.ml.pipeline import run_ml_pipeline
res = run_ml_pipeline(common_dates("SPXW")[-20:], "SPXW", every_secs=120,
                      progress_cb=lambda m: None)
print(f"\nML pipeline: {len(res['test_days'])} test days, "
      f"IC={res['metrics']['ic_overall']:.4f}, "
      f"OOS=${sum(p['pnl'] for p in res['payloads']):,.2f}")
print("top features:", [f"{n}:{v:.3f}" for n, v in res["feature_ic"][:5]])

# --- edge cases from the GUI review ---
import numpy as np
from vrb.gui.report import _fmt_money

assert _fmt_money(np.nan) == "n/a"
assert _fmt_money(np.inf) == "n/a"
assert _fmt_money(-1234.5) == "($1,234.50)"

# empty Buys side: a report over only short trades must not emit '$nan'
rep_short = compute_report(payloads)
buys_cells = [b for kind, _l, _a, b, _s in rep_short["rows"] if kind == "row"]
assert "$nan" not in buys_cells and "nan" not in [c.strip() for c in buys_cells], buys_cells
print("report edge cases OK")

# empty / all-NaN candle arrays must not crash the chart item
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication
from vrb.gui.charts import CandlestickItem, SignalChart
_app = QApplication.instance() or QApplication([])
empty = np.array([], float)
ci = CandlestickItem(empty, empty, empty, empty, empty)
assert ci.boundingRect().isNull() or ci.boundingRect().height() == 0
nan = np.full(5, np.nan)
t = np.arange(5, dtype=float)
ci2 = CandlestickItem(t, nan, nan, nan, nan)
br = ci2.boundingRect()
assert br.isNull(), "all-NaN candles should give an empty rect"
sc = SignalChart()
sc.set_candles(np.array([], "datetime64[s]"), empty, empty, empty, empty)
sc.add_trade_markers([{"entry_ts": np.datetime64("2026-06-05T09:00:00"),
                       "exit_ts": np.datetime64("2026-06-05T09:15:00"),
                       "direction": "sell", "pnl": 100.0}])  # must not raise
print("chart edge cases OK")
print("OK")
