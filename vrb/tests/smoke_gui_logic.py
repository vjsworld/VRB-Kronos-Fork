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
print("OK")
