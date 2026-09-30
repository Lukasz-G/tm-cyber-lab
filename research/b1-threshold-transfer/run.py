# QUESTION:  does the threshold-transfer advantage measured on APIGraph also hold on LAMDA, or is it
#            another single-corpus result?
# SURPRISE:  yes, and this is the check that should have been run before the claim was written down.
#            research/b5-fairness/ found on APIGraph that gradient boosting holds the better
#            precision/recall frontier while its OPERATING POINT does not survive drift -- choosing a
#            threshold honestly cost FPTM 1.73 F1 and the boosters 6 to 9 -- and that was described as an
#            advantage that had survived a cross-dataset check. It had not; b5-fairness runs on one
#            corpus. The prior from this project's own record is unkind: the LAMDA drift-robustness
#            advantage looked just as convincing and did not replicate on APIGraph.
#            There is also a mechanism that could break it. LAMDA's pool is ~48% malware against
#            APIGraph's ~10%. Threshold transfer is a claim about where a decision boundary sits, so a
#            four-fold difference in base rate is exactly the kind of thing that could reverse it.
# ARMS:      three models, all swept over their full score range, on identical rows:
#              1. FPTM margins, 3 seeds.
#              2. LightGBM predicted probabilities.
#              3. XGBoost predicted probabilities.
#            and three readings of each, the same three as on APIGraph so the two datasets are treated
#            identically: average precision (threshold-free), F1 at a threshold chosen on a held-out
#            slice of the TRAINING period, and oracle best-F1 on the test year. The oracle arm is not
#            decoration -- the quantity in dispute is the DIFFERENCE between it and the honest threshold,
#            so it has to be measured on both sides.
# PASS/FAIL: the claim REPLICATES if choosing the threshold honestly costs FPTM materially less F1 than
#            it costs both boosters, as on APIGraph. It FAILS if the costs are comparable, or if the
#            boosters lose less than FPTM does -- in which case threshold transfer joins drift robustness
#            as a LAMDA-or-APIGraph artifact and the manuscript's claim (iii) has to go the same way as
#            its predecessor.
#
#   julia --project=. -t 16 research/b1-threshold-transfer/export_margins.jl 3
#   python research/b1-threshold-transfer/run.py
#
# Features are read from the .tmx matrices rather than the parquet release, even though the boosters
# elsewhere in this project read parquet. The reason is alignment: the Julia side exported row indices
# into the .tmx files, so reading the same files makes "the same rows" provable rather than assumed. The
# .tmx is the binarised feature matrix the parquet was converted into, so nothing about the features
# changes. The label vectors are compared row for row before anything is computed.

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tmcyber.tmx import read_meta, read_tmx  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

TMX = Path("data/lamda/tmx")
MARGINS = Path("data/b1-threshold-transfer/fptm_margins.json")
TRAIN_YEARS = (2013, 2014)
TEST_YEARS = ("2016", "2017", "2018", "2019", "2020", "2021", "2022")


def year_arrays(year):
    meta = read_meta(TMX / f"{year}.meta.arrow").to_pydict()
    X = read_tmx(TMX / f"{year}.tmx")
    lab = np.asarray(meta["label"], dtype=np.int8)
    split = np.asarray([str(s) for s in meta["file_split"]])
    return X, lab, split


def best_f1(y, s):
    prec, rec, _ = precision_recall_curve(y, s)
    f = 2 * prec * rec / np.maximum(1e-12, prec + rec)
    return 100 * float(np.nanmax(f))


def pick_threshold(y, s):
    prec, rec, thr = precision_recall_curve(y, s)
    f = np.nan_to_num(2 * prec * rec / np.maximum(1e-12, prec + rec))[:-1]
    return float(thr[int(np.argmax(f))]) if len(thr) else 0.0


def f1_at(y, s, t):
    return 100 * f1_score(y, (s >= t).astype(np.int8), zero_division=0)


def main():
    if not MARGINS.exists():
        print(f"missing {MARGINS} -- run export_margins.jl first", file=sys.stderr)
        return 1
    M = json.loads(MARGINS.read_text())
    nseeds = M["n_seeds"]

    # --- rebuild the training pool exactly as Julia concatenated it: year by year, train rows in order
    Xparts, yparts = [], []
    for y in TRAIN_YEARS:
        X, lab, split = year_arrays(y)
        sel = np.flatnonzero(split == "train")
        Xparts.append(X[sel])
        yparts.append(lab[sel])
    Xpool = np.vstack(Xparts)
    ypool = np.concatenate(yparts)
    del Xparts, yparts

    tr = np.asarray(M["train_pos"], dtype=np.int64) - 1
    va = np.asarray(M["val_pos"], dtype=np.int64) - 1
    Xtr, ytr = Xpool[tr], ypool[tr]
    Xva, yva = Xpool[va], ypool[va]

    jl_val = np.asarray(M["labels"]["val"], dtype=np.int8)
    assert len(jl_val) == len(yva) and np.array_equal(jl_val, yva), (
        "validation labels disagree with Julia's export -- the two sides are not on the same rows"
    )
    print(f"train {len(ytr):,} ({100*ytr.mean():.1f}% malware)  "
          f"val {len(yva):,} ({100*yva.mean():.1f}%)")
    print("RESULT port_check val_labels=match")

    # --- evaluation sets, selected by the indices Julia exported ---
    evals = {}
    Xi, yi = [], []
    for y in TRAIN_YEARS:
        X, lab, split = year_arrays(y)
        sel = np.flatnonzero(split == "test")
        Xi.append(X[sel])
        yi.append(lab[sel])
    evals["IID"] = (np.vstack(Xi), np.concatenate(yi))
    del Xi, yi

    for y in TEST_YEARS:
        X, lab, _ = year_arrays(int(y))
        keep = np.asarray(M["eval_rows"][y], dtype=np.int64) - 1     # 1-based tmx row positions
        evals[y] = (X[keep], lab[keep])

    for k, (_, lab) in evals.items():
        jl = np.asarray(M["labels"][k], dtype=np.int8)
        assert np.array_equal(jl, lab), f"label mismatch on {k}"
    print(f"port check: IID and all {len(TEST_YEARS)} test years match row for row\n")

    # --- boosters on the identical training slice ---
    boosters = {}
    for name in ("lightgbm", "xgboost"):
        if name == "lightgbm":
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.02, num_leaves=256,
                                     n_jobs=-1, verbose=-1, random_state=0)
        else:
            import xgboost as xgb
            clf = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.02, max_leaves=256,
                                    tree_method="hist", n_jobs=-1, random_state=0,
                                    eval_metric="logloss")
        clf.fit(Xtr, ytr)
        boosters[name] = {"val": clf.predict_proba(Xva)[:, 1],
                          **{k: clf.predict_proba(X)[:, 1] for k, (X, _) in evals.items()}}
        print(f"trained {name}")
    print()

    order = ["IID"] + list(TEST_YEARS)

    print("(a) AVERAGE PRECISION -- threshold-free")
    print(f"{'eval':6} {'FPTM (mean±sd)':>18} {'LightGBM':>10} {'XGBoost':>10} {'gap':>8}")
    print("-" * 58)
    gaps = []
    for k in order:
        lab = evals[k][1]
        fp = [100 * average_precision_score(lab, np.asarray(M["margins"][k][s])) for s in range(nseeds)]
        al = 100 * average_precision_score(lab, boosters["lightgbm"][k])
        ax = 100 * average_precision_score(lab, boosters["xgboost"][k])
        g = max(al, ax) - float(np.mean(fp))
        gaps.append(g)
        print(f"{k:6} {np.mean(fp):11.2f}±{np.std(fp):5.2f} {al:10.2f} {ax:10.2f} {g:+8.2f}")
        print(f"RESULT ap eval={k} fptm={np.mean(fp):.2f} sd={np.std(fp):.2f} lgb={al:.2f} "
              f"xgb={ax:.2f} gap={g:+.2f}")
    print(f"\n  mean gap to the better booster: {np.mean(gaps):+.2f} AP")
    print(f"RESULT ap mean_gap={np.mean(gaps):+.2f}\n")

    print("(b) VALIDATION-CHOSEN threshold   vs   (c) ORACLE best-F1")
    print(f"{'eval':6} {'FPTM val':>9} {'FPTM orc':>9} {'LGB val':>8} {'LGB orc':>8} "
          f"{'XGB val':>8} {'XGB orc':>8}")
    print("-" * 62)
    c_f, c_l, c_x = [], [], []
    wins = 0
    for k in order:
        lab = evals[k][1]
        fv, fo = [], []
        for s in range(nseeds):
            t = pick_threshold(yva, np.asarray(M["margins"]["val"][s]))
            fv.append(f1_at(lab, np.asarray(M["margins"][k][s]), t))
            fo.append(best_f1(lab, np.asarray(M["margins"][k][s])))
        tl = pick_threshold(yva, boosters["lightgbm"]["val"])
        tx = pick_threshold(yva, boosters["xgboost"]["val"])
        lv, lo = f1_at(lab, boosters["lightgbm"][k], tl), best_f1(lab, boosters["lightgbm"][k])
        xv, xo = f1_at(lab, boosters["xgboost"][k], tx), best_f1(lab, boosters["xgboost"][k])
        c_f.append(np.mean(fo) - np.mean(fv)); c_l.append(lo - lv); c_x.append(xo - xv)
        if np.mean(fv) > max(lv, xv):
            wins += 1
        print(f"{k:6} {np.mean(fv):9.2f} {np.mean(fo):9.2f} {lv:8.2f} {lo:8.2f} {xv:8.2f} {xo:8.2f}")
        print(f"RESULT thresh eval={k} fptm_val={np.mean(fv):.2f} fptm_oracle={np.mean(fo):.2f} "
              f"lgb_val={lv:.2f} lgb_oracle={lo:.2f} xgb_val={xv:.2f} xgb_oracle={xo:.2f}")

    print(f"\n  what the oracle was worth: FPTM {np.mean(c_f):+.2f} F1, "
          f"LightGBM {np.mean(c_l):+.2f}, XGBoost {np.mean(c_x):+.2f}")
    print(f"  FPTM leads the honest-threshold F1 on {wins} of {len(order)} evaluation sets")
    print(f"RESULT oracle_cost fptm={np.mean(c_f):.2f} lgb={np.mean(c_l):.2f} xgb={np.mean(c_x):.2f}")
    print(f"RESULT honest_wins {wins}/{len(order)}")

    print("\n" + "-" * 62)
    apigraph = (1.73, 8.96, 6.14)
    print(f"On APIGraph the same three costs were FPTM {apigraph[0]}, LightGBM {apigraph[1]}, "
          f"XGBoost {apigraph[2]}.")
    if np.mean(c_f) < min(np.mean(c_l), np.mean(c_x)):
        print("REPLICATES: choosing the threshold honestly costs FPTM less than it costs either booster")
        print("here too, so threshold transfer is now a two-corpus result rather than a one-corpus one.")
    else:
        print("DOES NOT REPLICATE: the ordering seen on APIGraph is absent here, so threshold transfer")
        print("joins drift robustness as a single-corpus artifact and the manuscript claim must go.")
    print("Per-year figures only; no FAR mean is reported, because LAMDA's late-year malware counts are")
    print("antivirus label lag rather than drift and a mean across them reports the lag as a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
