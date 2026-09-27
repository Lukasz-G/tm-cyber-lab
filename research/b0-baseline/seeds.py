# QUESTION:  the B1 comparison puts FPTM's 10-seed mean against a SINGLE LightGBM run. Does
#            LightGBM's own seed variance matter, and is the comparison fair?
# SURPRISE:  probably not, and that is the useful answer. With feature_fraction and bagging_fraction
#            at their defaults of 1.0 there is no stochasticity left in LightGBM's tree building, so
#            every seed should give a bit-identical model — in which case one run IS the mean and the
#            caveat on B1 dissolves rather than needing to be measured away. If the seeds DO differ,
#            we need the mean and the spread before claiming FPTM beats it on the FAR years.
# ARMS:      five seeds of the same configuration. Nothing else varies, which is the point.
# PASS/FAIL: not pass/fail. Either the seeds are identical (report LightGBM as deterministic and
#            quote the single run) or they are not (quote mean±sd and redo the B1 gap arithmetic
#            against the mean).
#
#   python research/b0-baseline/seeds.py

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from tmcyber import lamda  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SEEDS = (0, 1, 2, 3, 4)


def load(paths, cols):
    Xs, ys = [], []
    for p in paths:
        X, meta = lamda.read_features(p, cols)
        Xs.append(X)
        ys.append(meta["label"].to_numpy().astype(np.int8))
    return np.vstack(Xs), np.concatenate(ys)


def main(root):
    import lightgbm as lgb

    years = lamda.discover(root)
    by = {y.year: y for y in years}
    cols = lamda.feature_names(by[2013].train)

    Xtr, ytr = load([by[y].train for y in lamda.TRAIN_YEARS], cols)
    sets = {"IID": load([by[y].test for y in lamda.TRAIN_YEARS], cols),
            "NEAR": load([p for y in lamda.NEAR_YEARS for p in (by[y].train, by[y].test)], cols)}
    for y in lamda.FAR_YEARS:
        sets[str(y)] = load([by[y].train, by[y].test], cols)

    scores = {k: [] for k in sets}
    preds0 = {}
    identical = True
    for seed in SEEDS:
        clf = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.02, num_leaves=256,
                                 n_jobs=-1, verbose=-1, random_state=seed)
        t = time.time()
        clf.fit(Xtr, ytr)
        line = f"seed {seed}  trained {time.time()-t:.0f}s  "
        for k, (X, y) in sets.items():
            p = clf.predict(X)
            scores[k].append(100 * f1_score(y, p))
            if seed == SEEDS[0]:
                preds0[k] = p
            elif not np.array_equal(p, preds0[k]):
                identical = False
            line += f"{k} {scores[k][-1]:.2f}  "
        print(line, flush=True)

    print(f"\nall seeds produced bit-identical predictions: {identical}")
    print(f"\n{'set':<6}{'mean':>9}{'sd':>8}{'min':>9}{'max':>9}")
    for k in sets:
        v = np.array(scores[k])
        print(f"{k:<6}{v.mean():>9.2f}{v.std():>8.4f}{v.min():>9.2f}{v.max():>9.2f}")

    print("\nRESULT lightgbm_deterministic=" + str(identical))
    for k in sets:
        v = np.array(scores[k])
        print(f"RESULT lgb_{k.lower()}_mean={v.mean():.2f} lgb_{k.lower()}_sd={v.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/lamda/raw"))
