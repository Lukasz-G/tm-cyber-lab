# QUESTION:  what do LightGBM and XGBoost score on AndroZoo 2020-2021, training on the 2019 pool?
# NOTE:      this is the DENSE regime -- 16,978 features at 39% density, against APIGraph's 1,159 at
#            1.75% and LAMDA's 4,561 at 3%. It is the reference point for whether a twenty-clause
#            Tsetlin machine is still competitive when negated literals stop being nearly free.
# SURPRISE:  nothing on its own. CALIBRATION, and the reference point the Tsetlin cross-check needs.
#            The result that matters is the COMPARISON in run.jl; this supplies its baseline on
#            identical row sets.
# ARMS:      two gradient-boosting models, same configuration as the LAMDA run so the two datasets
#            are treated identically. Determinism of LightGBM at this configuration was established
#            on LAMDA (five seeds, bit-identical), so single runs are means.
# PASS/FAIL: not pass/fail. It fails usefully only if the models cannot learn at all, which at a 10%
#            base rate would mean the protocol or the labels are being read wrongly.
#
#   python research/b5-androzoo/baselines.py
#
# Protocol note: this is the dataset's OWN temporal split -- one 2012 training pool, then every month
# of 2013-2018 as test -- not LAMDA's AnoShift arrangement. Forcing LAMDA's split onto it would be
# worse than useless, because the point of a cross-check is that the second dataset is genuinely
# different.
#
# Base rate: ~10% malware throughout, against LAMDA's ~37%. F1 at 10% positives is a harder and more
# realistic number than F1 at 37%, so the two datasets' absolute values are NOT comparable. Only the
# gap to the Tsetlin machine on the same rows is.

import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SRC = Path("data/apigraph/data/gen_androzoo_drebin")
MONTH_RE = re.compile(r"^(\d{4})-(\d{2})_selected\.npz$")
POOL = "2019-01to2019-12_selected.npz"


def load(path):
    f = np.load(path, allow_pickle=True)
    X = f["X_train"].astype(np.int8)
    y = (f["y_train"].astype(np.int64) > 0).astype(np.int8)   # 0 = benign, >0 = family index
    return X, y


def main():
    Xtr, ytr = load(SRC / POOL)
    print(f"train: {Xtr.shape} malware {ytr.sum():,} ({100*ytr.mean():.1f}%)")

    by_year = {}
    for p in sorted(SRC.glob("*_selected.npz")):
        m = MONTH_RE.match(p.name)
        if not m:
            continue
        by_year.setdefault(int(m.group(1)), []).append(p)

    evals = {}
    for year, paths in sorted(by_year.items()):
        Xs, ys = zip(*[load(p) for p in paths])
        evals[str(year)] = (np.vstack(Xs), np.concatenate(ys))
        print(f"  {year}: {evals[str(year)][0].shape[0]:,} rows, "
              f"{int(evals[str(year)][1].sum()):,} malware")

    out = {}
    for name in ("lightgbm", "xgboost"):
        print(f"\n=== {name} ===")
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
        scores = {}
        for k, (X, y) in evals.items():
            p = clf.predict(X)
            f1 = 100 * f1_score(y, p)
            scores[k] = f1
            print(f"  {k}  F1 {f1:6.2f}  precision {100*precision_score(y,p,zero_division=0):6.2f}  "
                  f"recall {100*recall_score(y,p,zero_division=0):6.2f}  n={len(y):,}")
        out[name] = scores
        print(f"RESULT {name} " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()))

    Path("research/b5-androzoo").mkdir(parents=True, exist_ok=True)
    Path("research/b5-androzoo/baselines.json").write_text(json.dumps(out, indent=2))
    print("\nwrote research/b5-androzoo/baselines.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
