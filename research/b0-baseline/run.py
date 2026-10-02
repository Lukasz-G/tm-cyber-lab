# QUESTION:  can our pipeline reproduce LAMDA's published LightGBM baseline on the same splits?
# SURPRISE:  nothing. CALIBRATION, not a finding. That gradient boosting degrades under temporal
#            drift is the published result; re-measuring it is not a contribution. This exists so
#            that "flat FPTM matches LightGBM" means something later, and it fails loudly if our
#            splits or preprocessing differ from theirs.
# ARMS:      LightGBM as the paper configures it, plus XGBoost as a cheap second point on the same
#            pipeline. No third arm needed: nothing is being attributed to a mechanism here.
# PASS/FAIL: PASS if IID and NEAR F1 land within ~1 point of 97.49 and 59.48. FAIL → our splits are
#            not theirs, and no later comparison against their numbers is valid. Stop and resolve.
#
# Deviation to record: the paper describes holding out the last month of each training year, while
# the release ships an 80/20 stratified within-year train/test split. We use the released split, so
# IID here is the 2013-14 test portions and not the two held-out months. NEAR and FAR use the
# whole year. Per-year FAR is reported; the single FAR mean is printed ONLY to demonstrate that we
# reproduce their figure, and is never used again.

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402

from tmcyber import lamda  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PUBLISHED = {"lightgbm": (97.49, 59.48, 47.24), "xgboost": (97.05, 55.84, 42.75)}
TOLERANCE = 1.0  # F1 points on IID and NEAR


def load(paths, cols):
    Xs, ys, yrs = [], [], []
    for p in paths:
        X, meta = lamda.read_features(p, cols)
        Xs.append(X)
        ys.append(meta["label"].to_numpy().astype(np.int8))
        yrs.append(meta["year_month"].to_numpy())
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(yrs)


def report(name, y, pred, ym):
    f1 = 100 * f1_score(y, pred)
    pr = 100 * precision_score(y, pred, zero_division=0)
    rc = 100 * recall_score(y, pred, zero_division=0)
    fnr = 100 * (1 - rc / 100)
    neg = y == 0
    fpr = 100 * float((pred[neg] == 1).mean()) if neg.any() else float("nan")
    print(f"  {name:<12} F1 {f1:6.2f}   precision {pr:6.2f}   recall {rc:6.2f}   "
          f"FNR {fnr:6.2f}   FPR {fpr:5.2f}   n={len(y):,}")
    return f1


def main(root):
    years = lamda.discover(root)
    by_year = {y.year: y for y in years}
    cols = lamda.feature_names(by_year[2013].train)
    print(f"features: {len(cols)}")

    t0 = time.time()
    Xtr, ytr, _ = load([by_year[y].train for y in lamda.TRAIN_YEARS], cols)
    print(f"train: {Xtr.shape} malware {ytr.sum():,} loaded in {time.time()-t0:.0f}s")

    evals = {}
    evals["IID"] = load([by_year[y].test for y in lamda.TRAIN_YEARS], cols)
    evals["NEAR"] = load([p for y in lamda.NEAR_YEARS for p in (by_year[y].train, by_year[y].test)], cols)
    for y in lamda.FAR_YEARS:
        evals[f"FAR {y}"] = load([by_year[y].train, by_year[y].test], cols)
    print(f"eval sets loaded in {time.time()-t0:.0f}s")

    results = {}
    for model_name in ("lightgbm", "xgboost"):
        print(f"\n=== {model_name} ===")
        t1 = time.time()
        if model_name == "lightgbm":
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.02, num_leaves=256,
                                     n_jobs=-1, verbose=-1, random_state=0)
        else:
            import xgboost as xgb
            clf = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.02, max_leaves=256,
                                    tree_method="hist", n_jobs=-1, random_state=0,
                                    eval_metric="logloss")
        clf.fit(Xtr, ytr)
        print(f"  trained in {time.time()-t1:.0f}s")

        scores = {}
        for name, (X, y, ym) in evals.items():
            scores[name] = report(name, y, clf.predict(X), ym)

        far_years = [scores[f"FAR {y}"] for y in lamda.FAR_YEARS]
        far_mean = float(np.mean(far_years))
        pub = PUBLISHED[model_name]
        print(f"  {'FAR mean':<12} F1 {far_mean:6.2f}   "
              f"(2018-2022 only; printed to compare with the published figure, then not used again)")
        print(f"  published:   IID {pub[0]}   NEAR {pub[1]}   FAR {pub[2]}")
        print(f"  delta:       IID {scores['IID']-pub[0]:+.2f}   NEAR {scores['NEAR']-pub[1]:+.2f}")
        results[model_name] = (scores["IID"], scores["NEAR"], far_mean)
        print(f"RESULT {model_name} iid={scores['IID']:.2f} near={scores['NEAR']:.2f} "
              f"far_mean={far_mean:.2f} " +
              " ".join(f"far{y}={scores[f'FAR {y}']:.2f}" for y in lamda.FAR_YEARS))

    lg = results["lightgbm"]
    ok = abs(lg[0] - PUBLISHED["lightgbm"][0]) <= TOLERANCE and \
         abs(lg[1] - PUBLISHED["lightgbm"][1]) <= TOLERANCE
    print(f"\nRESULT baseline_reproduction=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/lamda/raw"))
