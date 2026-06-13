"""Run the premium-expansion event study over all days and mine the findings.

Every threshold is a CLI parameter (the user's instruction: "the 10x is an
optimizable parameter, as are most parameters"):

    python -m vrb.research.expansion_study --ratio 10 --min-mid 0.15 \
        --latest-anchor 14:30:00 --controls 3 [--days N]

Outputs:
    vrb_out/research/expansion_events_{ratio}x.csv   full event+control dataset
    vrb_out/research/expansion_findings_{ratio}x.md  findings report
and prints the analysis: day clustering, event anatomy, and the feature LIFT
table (event vs premium-matched control medians + rank AUC) — the anomaly
mining output.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.calendar import common_dates
from ..util.timing import Timer, get_logger
from .expansion import study_day

log = get_logger("research.expansion")
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "vrb_out" / "research"

FEATURES = [
    "tod_min", "tte_min", "moneyness_pct", "abs_delta", "entry_mid",
    "spread_pct", "ret_5m", "ret_15m", "ret_30m", "rv_5m", "rv_15m", "rv_30m",
    "range_pos", "ret_from_open", "vix", "vix_chg_15m", "atm_iv", "iv_rv_15m",
    "skew_pc", "es_overnight_gap_pct", "book_imb", "with_move",
]


def _safe_study(date, **kw):
    try:
        return study_day(date, **kw)
    except (FileNotFoundError, ValueError):
        return []


def rank_auc(ev: np.ndarray, ct: np.ndarray) -> float:
    """P(event sample > control sample) — 0.5 = no separation."""
    ev = ev[np.isfinite(ev)]; ct = ct[np.isfinite(ct)]
    if len(ev) < 5 or len(ct) < 5:
        return np.nan
    from scipy.stats import mannwhitneyu
    u, _ = mannwhitneyu(ev, ct, alternative="two-sided")
    return float(u / (len(ev) * len(ct)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--min-mid", type=float, default=0.15)
    ap.add_argument("--latest-anchor", default="14:30:00")
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--days", type=int, default=0, help="0 = all")
    ap.add_argument("--root", default="SPXW")
    args = ap.parse_args()

    dates = common_dates(args.root)
    if args.days:
        dates = dates[-args.days:]
    fn = partial(_safe_study, root=args.root, controls_per_event=args.controls,
                 ratio_min=args.ratio, min_entry_mid=args.min_mid,
                 latest_anchor=args.latest_anchor)
    rows: list[dict] = []
    with Timer(f"expansion study {len(dates)} days @ {args.ratio}x", log):
        with ProcessPoolExecutor(max_workers=max(1, (__import__('os').cpu_count() or 2) - 1)) as ex:
            for day_rows in ex.map(fn, dates, chunksize=4):
                rows.extend(day_rows)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.ratio:g}x"
    csv_path = OUT_DIR / f"expansion_events_{tag}.csv"
    df.to_csv(csv_path, index=False)

    ev = df[df.is_event].copy()
    ct = df[~df.is_event].copy()
    lines: list[str] = []
    p = lines.append
    p(f"# Premium-expansion study — {tag} (entry mid >= {args.min_mid}, anchor <= {args.latest_anchor})")
    p("")
    p(f"days scanned: {len(dates)} | events: {len(ev)} | controls: {len(ct)}")
    if len(ev) == 0:
        p("No events found."); print("\n".join(lines)); return

    per_day = ev.groupby("date").size()
    p(f"days with >=1 event: {per_day.size}/{len(dates)} "
      f"({per_day.size / len(dates) * 100:.0f}%) | "
      f"median events/event-day: {per_day.median():.0f} | max: {per_day.max()}")
    top = per_day.sort_values(ascending=False).head(10)
    p("")
    p("## Day clustering (top 10 event days)")
    for d, n in top.items():
        p(f"  {d}: {n} events")
    conc = per_day.sort_values(ascending=False)
    half = np.searchsorted(np.cumsum(conc.values) / conc.sum(), 0.5) + 1
    p(f"  -> 50% of all events live on just {half} days "
      f"({half / max(per_day.size, 1) * 100:.0f}% of event days)")

    p("")
    p("## Event anatomy")
    p(f"  calls: {(ev.right == 0).mean() * 100:.0f}%  puts: {(ev.right == 1).mean() * 100:.0f}%")
    p(f"  median entry mid: {ev.entry_mid.median():.2f}  median peak: {ev.peak_mid.median():.2f}  "
      f"median ratio: {ev.ratio.median():.1f}x")
    p(f"  median |moneyness|: {ev.moneyness_pct.abs().median():.2f}%  "
      f"median |delta|: {ev.abs_delta.median():.3f}")
    p(f"  median minutes to {tag}: {ev.min_to_10x.median():.0f}  to peak: {ev.min_to_peak.median():.0f}")
    by_hr = ev.groupby((ev.tod_min // 60).astype(int)).size()
    p("  anchors by hour since open: " + "  ".join(f"h{h}:{n}" for h, n in by_hr.items()))

    p("")
    p("## Feature lift — event vs premium-matched controls (AUC: 0.5 = nothing)")
    p(f"{'feature':22s} {'event med':>10s} {'ctrl med':>10s} {'AUC':>6s}")
    rows_lift = []
    for f in FEATURES:
        if f not in df.columns:
            continue
        a = rank_auc(ev[f].to_numpy(float), ct[f].to_numpy(float))
        rows_lift.append((f, ev[f].median(), ct[f].median(), a))
    rows_lift.sort(key=lambda x: -abs((x[3] or 0.5) - 0.5))
    for f, em, cm, a in rows_lift:
        p(f"{f:22s} {em:>10.3f} {cm:>10.3f} {a:>6.2f}" if np.isfinite(a)
          else f"{f:22s} {em:>10.3f} {cm:>10.3f}    n/a")

    report = "\n".join(lines)
    md_path = OUT_DIR / f"expansion_findings_{tag}.md"
    md_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nevents csv -> {csv_path}\nreport    -> {md_path}")


if __name__ == "__main__":
    main()
