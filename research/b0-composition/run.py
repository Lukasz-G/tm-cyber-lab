# QUESTION:  does the downloaded LAMDA baseline variant match the composition the published
#            baselines were computed on?
# SURPRISE:  nothing. This is CALIBRATION, not an experiment, and must not be written up as a
#            finding. It exists because every later comparison against 97.49 / 59.48 / 47.24 F1 is
#            meaningless if we hold a different release than those numbers came from.
# ARMS:      one. There is nothing to control for — this compares our copy against printed numbers.
# PASS/FAIL: PASS if totals match the paper: 1,008,381 APKs, 369,906 malware, 638,475 benign,
#            1,380 families, 150,604 singletons, no 2015. FAIL on any mismatch → stop and resolve
#            before anything downstream, because the pipeline would be uncalibrated.
#
# Also records, because later steps need them and nobody should re-derive them: per-year and
# per-month row counts, and how many months are too thin for the 100-background / 100-explained
# attribution protocol.

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import pandas as pd  # noqa: E402

from tmcyber import lamda  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PUBLISHED = {
    "total": 1_008_381,
    "malware": 369_906,
    "benign": 638_475,
    "families": 1_380,
    "singletons": 150_604,
}
MIN_ROWS_FOR_ATTRIBUTION = 200  # 100 background + 100 explained


def main(root):
    years = lamda.discover(root)
    print(f"years on disk: {[y.year for y in years]}")

    frames = []
    for yf in years:
        for kind, path in (("train", yf.train), ("test", yf.test)):
            df = lamda.read_meta(path)
            df["year"] = yf.year
            df["file_split"] = kind
            df["split"] = lamda.split_of(yf.year)
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # The paper labels benign at vt_detection == 0 and malware at >= 4, with [1,3] discarded, so the
    # stored label should already be that partition; check, not assume.
    df["label"] = df["label"].astype(int)
    total = len(df)
    malware = int((df["label"] == 1).sum())
    benign = int((df["label"] == 0).sum())

    # Family counting convention, which the paper does not spell out and which matters:
    # AVClass2 names every unclustered sample `singleton:<sha256>`, so those strings are per-sample
    # placeholders and not families. The paper's "1,380 families" counts distinct NAMED families
    # excluding both the singletons and the literal `unknown` bucket; its "150,604 singletons" is the
    # count of `singleton:*` strings. Counting naively instead gives 151,985 families and 151,075
    # singletons, because 471 named families happen to hold exactly one sample -- a real property of
    # the data, not an error, and the reason the two definitions diverge.
    fams = df.loc[df["label"] == 1, "family"].astype(str)
    fam_counts = Counter(fams)
    singleton_names = {k: v for k, v in fam_counts.items() if k.startswith("singleton:")}
    named = {k: v for k, v in fam_counts.items() if not k.startswith("singleton:")}
    families = len(named) - (1 if "unknown" in named else 0)
    singletons = len(singleton_names)
    named_of_size_one = sum(1 for v in named.values() if v == 1)

    got = {
        "total": total,
        "malware": malware,
        "benign": benign,
        "families": families,
        "singletons": singletons,
    }

    print("\n=== composition vs published ===")
    print(f"{'quantity':<12}{'ours':>12}{'published':>12}{'delta':>10}")
    ok = True
    for k, want in PUBLISHED.items():
        have = got[k]
        d = have - want
        ok &= d == 0
        print(f"{k:<12}{have:>12,}{want:>12,}{d:>+10,}")

    print(f"\nnamed families holding exactly one sample: {named_of_size_one} "
          f"(distinct family strings in total: {len(fam_counts):,})")
    print(f"\n2015 present: {2015 in set(df['year'])} (expected False)")
    ok &= 2015 not in set(df["year"])

    print("\n=== per year ===")
    print(f"{'year':<6}{'split':<11}{'rows':>10}{'malware':>10}{'benign':>10}{'mw %':>8}")
    for year, g in df.groupby("year", sort=True):
        mw = int((g["label"] == 1).sum())
        bn = int((g["label"] == 0).sum())
        print(f"{year:<6}{lamda.split_of(year):<11}{len(g):>10,}{mw:>10,}{bn:>10,}{100*mw/len(g):>7.1f}%")

    print("\n=== months too thin for 100+100 attribution ===")
    per_month = df.groupby("year_month").size().sort_index()
    thin = per_month[per_month < MIN_ROWS_FOR_ATTRIBUTION]
    print(f"months total: {len(per_month)}   under {MIN_ROWS_FOR_ATTRIBUTION} rows: {len(thin)}")
    for ym, n in thin.items():
        print(f"  {ym}: {n}")
    print(f"month row counts: min {per_month.min():,}  median {int(per_month.median()):,}  max {per_month.max():,}")

    print("\n=== label-lag years, the reason FAR is never a single mean ===")
    for year in lamda.LABEL_LAG_YEARS:
        g = df[df["year"] == year]
        if len(g):
            mw = int((g["label"] == 1).sum())
            print(f"  {year}: {mw:,} malware vs {int((g['label']==0).sum()):,} benign")

    feats = lamda.feature_names(years[0].train)
    print(f"\nfeature columns: {len(feats)} (expected 4,561 → {2*len(feats)} literals)")
    ok &= len(feats) == 4561

    print("\nRESULT composition=" + ("PASS" if ok else "FAIL"))
    print(f"RESULT total={total} malware={malware} benign={benign} families={families} "
          f"singletons={singletons} features={len(feats)} months={len(per_month)}")
    return 0 if ok else 1


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/lamda/raw"
    raise SystemExit(main(root))
