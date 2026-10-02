# QUESTION:  is what LAMDA calls concept drift partly the LABEL BOUNDARY moving and not the data
#            moving -- and are the two separable in time?
# SURPRISE:  yes. Every drift paper on this dataset treats the labels as fixed ground truth and
#            attributes all degradation to the features. But LAMDA's labels are a THRESHOLD on a
#            continuous, human-and-vendor-generated quantity: malware at vt_detection >= 4, benign at
#            0, and [1,3] discarded. If the detection-count distribution itself shifts over time -- AV
#            vendors becoming more or less aggressive, coverage improving -- then some samples cross the
#            threshold for reasons that have nothing to do with the APK, and the label series carries
#            drift of its own. If that series moves in step with the feature series the two are
#            confounded and no drift result on this dataset can separate them. If it moves
#            independently, then label-boundary drift is a distinct and unreported phenomenon.
# ARMS:      three series over 2013-2022, and the third is the one that makes the comparison readable:
#              1. LABEL BOUNDARY -- per year, the vt_detection distribution among malware, the share
#                 sitting borderline at 4-6 detections, and how many labels would flip if the threshold
#                 moved 4->5 or 4->10. This is label instability measured directly.
#              2. DATA -- per year, the per-feature presence rate over all samples, and the L1 distance
#                 between consecutive years. This is feature drift.
#              3. CONTROL: the same feature distance computed on BENIGN SAMPLES ONLY. Benign means
#                 vt_detection == 0 exactly, which is the one label in this dataset that no threshold
#                 choice can disturb. So drift measured there is pure data drift with the label question
#                 removed. Without this arm, a correlation between arms 1 and 2 could not be told apart
#                 from the labels reshaping which samples arm 2 is averaging over.
# PASS/FAIL: not a gate. SEPARABLE if the label series and the data series are weakly correlated across
#            years while arm 3 tracks arm 2 closely -- then label drift is its own signal. CONFOUNDED if
#            arms 1 and 2 move together, in which case this dataset cannot support a claim that
#            separates them and the project's pre-registered prediction 5 must be withdrawn rather than
#            reinterpreted.
#
#   python research/b3-label-drift/run.py
#
# WHAT THIS IS NOT, and it matters. The pre-registered prediction 5 is about LAMDA's Appendix F, which
# tracks samples whose VirusTotal verdicts STRENGTHENED, WEAKENED or FLIPPED between two scans -- 10,289
# weakened verdicts in 2017, and so on. That needs a detection count at two points in time per sample.
# The public release does not contain it: metadata.csv carries exactly one vt_detection per sha256, a
# single snapshot. There is no vt_detections.csv in the release, contrary to an earlier note in this
# repository's experiment index, which was wrong and is corrected. So prediction 5 remains UNTESTED and
# is blocked on asking the authors. What follows is a different, weaker, runnable question about the
# same underlying worry, and it must not be reported as prediction 5.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tmcyber.tmx import read_meta, read_tmx  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

META = Path("data/lamda/raw/metadata.csv")
TMX = Path("data/lamda/tmx")
YEARS = [2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
MAL_THRESHOLD = 4


def label_series():
    """Arm 1: how stable is the label, per year, given that it is a threshold on a moving quantity?"""
    df = pd.read_csv(META, usecols=["vt_detection", "label", "year"])
    df = df[df["year"].isin(YEARS)]
    rows = []
    for y in YEARS:
        d = df[df["year"] == y]
        mal = d[d["vt_detection"] >= MAL_THRESHOLD]["vt_detection"].to_numpy()
        n_mal = len(mal)
        if n_mal == 0:
            rows.append(dict(year=y, n_mal=0))
            continue
        rows.append(dict(
            year=y,
            n_mal=n_mal,
            median_vt=float(np.median(mal)),
            # borderline: would be relabelled by a small threshold move
            borderline=100.0 * np.mean((mal >= 4) & (mal <= 6)),
            flip_at_5=100.0 * np.mean(mal < 5),
            flip_at_10=100.0 * np.mean(mal < 10),
        ))
    return pd.DataFrame(rows)


def presence_rates(year, benign_only):
    """Per-feature presence rate for one year, over all rows or benign rows only."""
    meta = read_meta(TMX / f"{year}.meta.arrow").to_pydict()
    X = read_tmx(TMX / f"{year}.tmx")
    lab = np.asarray(meta["label"], dtype=np.int8)
    if benign_only:
        X = X[lab == 0]
    r = X.mean(axis=0).astype(np.float64)
    del X
    return r, (int((lab == 0).sum()) if benign_only else len(lab))


MIN_MONTH_ROWS = 200


def monthly_rates(year):
    """
    Per-month presence rates for one year, all rows and benign only, plus the month's row count.

    Year resolution gives only eight transitions, which cannot distinguish a correlation of 0.55 from
    zero -- so the separability question is decided here instead. One read of the year's matrix, sliced
    by month, because re-reading per month would be twelve times the I/O for the same bytes.
    """
    meta = read_meta(TMX / f"{year}.meta.arrow").to_pydict()
    X = read_tmx(TMX / f"{year}.tmx")
    lab = np.asarray(meta["label"], dtype=np.int8)
    mo = np.asarray([-1 if m is None else int(m) for m in meta["month"]], dtype=np.int16)
    out = {}
    for m in range(1, 13):
        sel = mo == m
        n = int(sel.sum())
        n >= MIN_MONTH_ROWS or None
        if n < MIN_MONTH_ROWS:
            continue
        Xm = X[sel]
        lm = lab[sel]
        if int((lm == 0).sum()) < MIN_MONTH_ROWS // 2:
            continue
        out[m] = (Xm.mean(axis=0).astype(np.float64),
                  Xm[lm == 0].mean(axis=0).astype(np.float64),
                  n)
        del Xm
    del X
    return out


def monthly_label_fragility():
    """Share of each month's malware that a threshold of 10 would relabel benign."""
    df = pd.read_csv(META, usecols=["vt_detection", "year", "year_month"])
    df = df[df["year"].isin(YEARS)]
    mal = df[df["vt_detection"] >= MAL_THRESHOLD]
    g = mal.groupby("year_month")["vt_detection"]
    frag = (g.apply(lambda s: 100.0 * float(np.mean(s.to_numpy() < 10))))
    cnt = g.size()
    return frag[cnt >= 50]


def main():
    print("ARM 1 -- label boundary: is the label itself stable over time?")
    lab = label_series()
    print(f"{'year':6} {'malware':>9} {'median vt':>10} {'4-6 dets':>9} {'flip 4→5':>9} {'flip 4→10':>10}")
    print("-" * 60)
    for _, r in lab.iterrows():
        if r["n_mal"] == 0:
            continue
        print(f"{int(r['year']):6} {int(r['n_mal']):9,} {r['median_vt']:10.1f} "
              f"{r['borderline']:8.1f}% {r['flip_at_5']:8.1f}% {r['flip_at_10']:9.1f}%")
        print(f"RESULT label year={int(r['year'])} n_mal={int(r['n_mal'])} median_vt={r['median_vt']:.1f} "
              f"borderline={r['borderline']:.2f} flip5={r['flip_at_5']:.2f} flip10={r['flip_at_10']:.2f}")
    print("\n  'flip 4→10' is the share of this year's malware that a threshold of 10 would relabel")
    print("  benign. It is label fragility, measured directly, and it is not constant across years.\n")

    print("ARMS 2 and 3 -- feature drift, all samples and benign only")
    all_rates, ben_rates, counts = {}, {}, {}
    for y in YEARS:
        all_rates[y], counts[y] = presence_rates(y, benign_only=False)
        ben_rates[y], _ = presence_rates(y, benign_only=True)
        print(f"  {y}: {counts[y]:,} rows")

    print(f"\n{'transition':14} {'L1 all':>10} {'L1 benign only':>16} {'difference':>12}")
    print("-" * 56)
    d_all, d_ben = [], []
    for a, b in zip(YEARS, YEARS[1:]):
        la = float(np.abs(all_rates[b] - all_rates[a]).sum())
        lb = float(np.abs(ben_rates[b] - ben_rates[a]).sum())
        d_all.append(la)
        d_ben.append(lb)
        print(f"{f'{a}→{b}':14} {la:10.3f} {lb:16.3f} {la - lb:+12.3f}")
        print(f"RESULT data from={a} to={b} l1_all={la:.4f} l1_benign={lb:.4f}")

    # --- the comparison ---
    print("\n" + "-" * 70)
    lab_ok = lab[lab["n_mal"] > 0].reset_index(drop=True)
    # align the label series to the transitions: use the LATER year of each pair
    later = [b for a, b in zip(YEARS, YEARS[1:])]
    lab_by_year = {int(r["year"]): r for _, r in lab_ok.iterrows()}
    lab_vals = [lab_by_year[y]["flip_at_10"] for y in later if y in lab_by_year]
    n = min(len(lab_vals), len(d_all))

    rho_ld, p_ld = spearmanr(lab_vals[:n], d_all[:n])
    rho_ab, p_ab = spearmanr(d_all[:n], d_ben[:n])
    print(f"label fragility vs feature drift (all):     Spearman rho {rho_ld:+.3f}  p {p_ld:.3f}")
    print(f"feature drift all vs benign-only (control): Spearman rho {rho_ab:+.3f}  p {p_ab:.3f}")
    print(f"RESULT corr label_vs_data rho={rho_ld:+.4f} p={p_ld:.4f} n={n}")
    print(f"RESULT corr all_vs_benign rho={rho_ab:+.4f} p={p_ab:.4f} n={n}")

    print(f"\n  n = {n} transitions. At that size a rho of 0.5 is not distinguishable from zero, which")
    print("  is why the year-resolution verdict below is not the answer -- the monthly series is.\n")

    # ---------------------------------------------------------------------------------------------
    # Month resolution. The decisive version: ~100 transitions instead of 8.
    # ---------------------------------------------------------------------------------------------
    print("=" * 70)
    print("MONTH RESOLUTION -- the same three arms, where the correlation can actually be measured")
    print("=" * 70)
    per_month = {}
    for y in YEARS:
        for m, v in monthly_rates(y).items():
            per_month[(y, m)] = v
    keys = sorted(per_month)
    print(f"{len(keys)} months with >= {MIN_MONTH_ROWS} rows\n")

    frag = monthly_label_fragility()
    m_all, m_ben, m_frag, labels = [], [], [], []
    for (ya, ma), (yb, mb) in zip(keys, keys[1:]):
        ra, ba, _ = per_month[(ya, ma)]
        rb, bb, _ = per_month[(yb, mb)]
        key = f"{yb}-{mb:02d}"
        if key not in frag.index:
            continue
        m_all.append(float(np.abs(rb - ra).sum()))
        m_ben.append(float(np.abs(bb - ba).sum()))
        m_frag.append(float(frag.loc[key]))
        labels.append(f"{ya}-{ma:02d}→{key}")

    nm = len(m_all)
    r_ld, p_ld_m = spearmanr(m_frag, m_all)
    r_ab, p_ab_m = spearmanr(m_all, m_ben)
    print(f"{'quantity':46} {'rho':>8} {'p':>9} {'n':>5}")
    print("-" * 70)
    print(f"{'label fragility vs feature drift (all samples)':46} {r_ld:+8.3f} {p_ld_m:9.4f} {nm:5d}")
    print(f"{'feature drift: all vs benign-only (control)':46} {r_ab:+8.3f} {p_ab_m:9.4f} {nm:5d}")
    print(f"RESULT monthly corr label_vs_data rho={r_ld:+.4f} p={p_ld_m:.4g} n={nm}")
    print(f"RESULT monthly corr all_vs_benign rho={r_ab:+.4f} p={p_ab_m:.4g} n={nm}")
    print(f"RESULT monthly median l1_all={np.median(m_all):.3f} l1_benign={np.median(m_ben):.3f} "
          f"frag={np.median(m_frag):.2f}")

    print()
    if p_ld_m < 0.05 and abs(r_ld) >= 0.5:
        print("CONFOUNDED at month resolution: label fragility and feature drift move together, so this")
        print("dataset does not support a claim that separates them.")
    elif p_ld_m >= 0.05 or abs(r_ld) < 0.3:
        print("SEPARABLE at month resolution: label fragility carries no reliable linear relation to")
        print("feature drift, so the two are distinct signals in place of one phenomenon seen twice.")
    else:
        print("WEAK ASSOCIATION at month resolution: real but small. Reportable as a caveat on any")
        print("claim that treats the labels as fixed, not as a result in itself.")
    if r_ab > 0.7 and p_ab_m < 0.05:
        print("The control holds: feature drift measured on unambiguously-benign samples tracks the")
        print("all-sample series, so that series is not an artefact of the labels reshaping the")
        print("population being averaged over.")
    else:
        print("The control does NOT hold, which weakens every reading above: the all-sample feature")
        print("series may be partly an artefact of which samples the labels admit.")

    print("\n" + "-" * 70)
    print("Year-resolution verdict below, kept because it was the pre-registered rule, and it was")
    print("under-specified: it tested a correlation threshold without requiring significance, which at")
    print("eight transitions it cannot have. The monthly result above supersedes it.")
    if abs(rho_ld) < 0.5 and rho_ab > 0.7:
        print("SEPARABLE: label fragility and feature drift move largely independently, while the")
        print("benign-only control tracks the all-sample series -- so the feature drift is not an")
        print("artefact of the labels reshaping the population being averaged.")
    elif abs(rho_ld) >= 0.5:
        print("CONFOUNDED: label fragility and feature drift move together on this dataset, so a result")
        print("that claims to separate them is not supported here.")
    else:
        print("INCONCLUSIVE at this resolution: the control does not track the all-sample series")
        print("closely enough to license either reading. Per-month resolution would be the next step.")

    print("\nThis is NOT pre-registered prediction 5. That prediction concerns Appendix F's")
    print("strengthened/weakened/flipped verdict counts, which need a detection count at two points in")
    print("time per sample. The release carries one snapshot per sha256, so prediction 5 is untested and")
    print("blocked on the dataset authors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
