# QUESTION:  how much does reporting F1 at a threshold chosen on the evaluation data inflate it, and does
#            the inflation grow with drift?
# SURPRISE:  yes, and it is a statement about how this field evaluates rather than about any model. A
#            threshold picked to maximise F1 on the test period is an ORACLE: it uses labels a deployment
#            does not have. The resulting number is an upper bound, and it is widely reported as though it
#            were attainable -- including, until recently, everywhere in this project. What is not known is
#            the SIZE of the gap. If it were a point or two, the practice would be harmless. If it grows
#            with drift, then every temporal-evaluation table in the area is inflated by an amount that
#            increases exactly where the papers claim their contribution, and comparisons between models
#            drawn from such tables are not safe.
# ARMS:      three models x two corpora x every evaluation period, using data already produced by
#            research/b5-fairness/, research/b1-threshold-transfer/ and research/calibration-check/. The
#            three models matter here for a reason opposite to the usual one: the claim is only a field
#            finding if the inflation appears for ALL of them. A gap that hit one learner would be a fact
#            about that learner.
# PASS/FAIL: not a gate; it is a measurement that either supports a methodological claim or does not.
#            SUPPORTED if the inflation is small in distribution and grows materially on drifted periods,
#            for every model. NOT SUPPORTED if it is roughly constant across periods, in which case it is
#            a fixed reporting offset, still worth stating but not a drift-evaluation problem.
#
#   python research/threshold-bias/analyse.py
#
# Reads the RESULT lines the earlier experiments already wrote; nothing is retrained. Writes
# per-period.csv for the figure.

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
SOURCES = {
    "APIGraph": HERE.parent / "b5-fairness" / "results.txt",
    "LAMDA": HERE.parent / "b1-threshold-transfer" / "results.txt",
}
CALIB = HERE.parent / "calibration-check" / "results.txt"

# "RESULT thresh year=2016 fptm_val=.. fptm_oracle=.. lgb_val=.. lgb_oracle=.. xgb_val=.. xgb_oracle=.."
ROW = re.compile(
    r"RESULT thresh (?:year|eval)=(\S+)\s+"
    r"fptm_val=([\d.]+)\s+fptm_oracle=([\d.]+)\s+"
    r"lgb_val=([\d.]+)\s+lgb_oracle=([\d.]+)\s+"
    r"xgb_val=([\d.]+)\s+xgb_oracle=([\d.]+)")

MODELS = ("FPTM", "LightGBM", "XGBoost")


def load():
    out = []
    for corpus, path in SOURCES.items():
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = ROW.search(line)
            if not m:
                continue
            period = m.group(1)
            vals = [float(x) for x in m.groups()[1:]]
            for i, model in enumerate(MODELS):
                val, oracle = vals[2 * i], vals[2 * i + 1]
                out.append(dict(corpus=corpus, period=period, model=model,
                                achievable=val, oracle=oracle, inflation=oracle - val))
    return out


def load_rate_matching():
    """The corrected arm, for the 'and here is the fix' half."""
    out = defaultdict(dict)
    if not CALIB.exists():
        return out
    pat = re.compile(r"RESULT corpus=(\S+) model=(\S+) arm=(\S+) mean_f1=([\d.]+) oracle_gap=([\d.-]+)")
    for line in CALIB.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            corpus, model, arm, mean_f1, gap = m.groups()
            out[(corpus, model)][arm] = (float(mean_f1), float(gap))
    return out


def main():
    rows = load()
    if not rows:
        print("no data found -- run the three source experiments first", file=sys.stderr)
        return 1

    # in-distribution periods, against drifted ones
    IID = {"IID", "2013"}          # APIGraph's 2013 is its first test year, closest to the 2012 pool

    print("INFLATION FROM AN ORACLE THRESHOLD -- F1 points, by period")
    print("An oracle threshold is chosen to maximise F1 on the period being scored; the achievable one")
    print("is chosen on held-out data from the TRAINING period and applied unchanged.\n")

    for corpus in SOURCES:
        sub = [r for r in rows if r["corpus"] == corpus]
        if not sub:
            continue
        periods = sorted({r["period"] for r in sub}, key=lambda p: (p != "IID", p))
        print(f"{corpus}")
        print(f"  {'period':8}" + "".join(f"{m:>12}" for m in MODELS))
        print("  " + "-" * (8 + 12 * len(MODELS)))
        for p in periods:
            line = f"  {p:8}"
            for m in MODELS:
                v = [r["inflation"] for r in sub if r["period"] == p and r["model"] == m]
                line += f"{v[0]:12.2f}" if v else f"{'-':>12}"
            print(line)
        # in-distribution vs drifted
        for m in MODELS:
            iid = [r["inflation"] for r in sub if r["model"] == m and r["period"] in IID]
            drift = [r["inflation"] for r in sub if r["model"] == m and r["period"] not in IID]
            if iid and drift:
                mi, md = sum(iid) / len(iid), sum(drift) / len(drift)
                print(f"    {m:10} in-distribution {mi:6.2f}   drifted {md:6.2f}   "
                      f"grows by {md - mi:+.2f}")
                print(f"RESULT corpus={corpus} model={m} inflation_iid={mi:.2f} "
                      f"inflation_drifted={md:.2f} growth={md - mi:+.2f}")
        print()

    rm = load_rate_matching()
    if rm:
        print("AND THE FIX -- mean oracle gap under each rule, from research/calibration-check/")
        print(f"  {'corpus':10} {'model':10} {'max-F1 on val':>15} {'rate matching':>15}")
        print("  " + "-" * 52)
        for (corpus, model), arms in sorted(rm.items()):
            a = arms.get("raw-valF1"); b = arms.get("quantile")
            if a and b:
                print(f"  {corpus:10} {model:10} {a[1]:15.2f} {b[1]:15.2f}")
        print()

    with open(HERE / "per-period.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["corpus", "period", "model", "achievable", "oracle", "inflation"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote per-period.csv ({len(rows)} rows)")

    allinf = [r["inflation"] for r in rows]
    drifted = [r["inflation"] for r in rows if r["period"] not in IID]
    print(f"\noverall: median inflation {sorted(allinf)[len(allinf)//2]:.2f} F1, "
          f"max {max(allinf):.2f}; on drifted periods only, median "
          f"{sorted(drifted)[len(drifted)//2]:.2f}")
    print(f"RESULT overall median={sorted(allinf)[len(allinf)//2]:.2f} max={max(allinf):.2f} "
          f"drifted_median={sorted(drifted)[len(drifted)//2]:.2f} n={len(allinf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
