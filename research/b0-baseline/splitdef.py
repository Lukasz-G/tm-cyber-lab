# QUESTION:  our LightGBM reproduces the published IID F1 to two decimals (97.49 vs 97.49) but comes
#            out 12.6 points ABOVE the published NEAR (72.12 vs 59.48). Which definition of the
#            NEAR and FAR evaluation sets reproduces the published figures?
# SURPRISE:  yes. An exact IID match with a 12-point NEAR gap means the model and splits are right
#            and the EVALUATION SET definition is not — and the paper does not state it unambiguously.
#            Every later claim of the form "flat FPTM matches LightGBM under drift" depends on
#            getting this right, and getting it wrong would flatter or damn us by 12 points.
# ARMS:      one trained model, six evaluation-set definitions. The model is held fixed so that only
#            the definition varies — that is the whole point.
# PASS/FAIL: PASS if exactly one definition reproduces NEAR 59.48 and FAR 47.24 within ~1 point.
#            If none does, the published numbers are not reconstructible from the release and B1 must
#            compare against numbers we computed ourselves, stating that plainly.
#
#   python research/b0-baseline/splitdef.py
#
# Trains once (~4 min) and caches the model, because the earlier run threw it away.

import pickle
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

PUB_NEAR, PUB_FAR = 59.48, 47.24
CACHE = Path("data/lamda/lightgbm_2013_2014.pkl")


def load(paths, cols):
    Xs, ys = [], []
    for p in paths:
        X, meta = lamda.read_features(p, cols)
        Xs.append(X)
        ys.append(meta["label"].to_numpy().astype(np.int8))
    return np.vstack(Xs), np.concatenate(ys)


def main(root):
    years = lamda.discover(root)
    by = {y.year: y for y in years}
    cols = lamda.feature_names(by[2013].train)

    if CACHE.is_file():
        clf = pickle.loads(CACHE.read_bytes())
        print(f"loaded cached model from {CACHE}")
    else:
        Xtr, ytr = load([by[y].train for y in lamda.TRAIN_YEARS], cols)
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.02, num_leaves=256,
                                 n_jobs=-1, verbose=-1, random_state=0)
        t = time.time()
        clf.fit(Xtr, ytr)
        print(f"trained in {time.time()-t:.0f}s")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(clf))
        del Xtr, ytr

    # per-year F1 under both portion choices, computed once and reused by every definition below
    per_year = {}
    for year in lamda.NEAR_YEARS + lamda.FAR_YEARS:
        for portion, paths in (("test", [by[year].test]),
                               ("all", [by[year].train, by[year].test])):
            X, y = load(paths, cols)
            per_year[(year, portion)] = (100 * f1_score(y, clf.predict(X)), len(y), int(y.sum()))
        del X, y

    print(f"\n{'year':<6}{'portion':<8}{'F1':>8}{'rows':>10}{'malware':>10}")
    for (year, portion), (f1, n, mw) in sorted(per_year.items()):
        print(f"{year:<6}{portion:<8}{f1:>8.2f}{n:>10,}{mw:>10,}")

    def pooled(years_, portion):
        Xs, ys = [], []
        for year in years_:
            paths = [by[year].test] if portion == "test" else [by[year].train, by[year].test]
            X, y = load(paths, cols)
            Xs.append(clf.predict(X)); ys.append(y)
        return 100 * f1_score(np.concatenate(ys), np.concatenate(Xs))

    defs = {}
    for portion in ("all", "test"):
        defs[f"pooled rows, {portion}"] = (
            pooled(lamda.NEAR_YEARS, portion), pooled(lamda.FAR_YEARS, portion))
        defs[f"mean of per-year F1, {portion}"] = (
            float(np.mean([per_year[(y, portion)][0] for y in lamda.NEAR_YEARS])),
            float(np.mean([per_year[(y, portion)][0] for y in lamda.FAR_YEARS])))
    # FAR as the paper's own window (2018-2025) rather than our 2018-2022
    full_far = [y for y in (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)]
    for portion in ("all", "test"):
        fy = []
        for year in full_far:
            if (year, portion) not in per_year:
                paths = [by[year].test] if portion == "test" else [by[year].train, by[year].test]
                X, y = load(paths, cols)
                per_year[(year, portion)] = (100 * f1_score(y, clf.predict(X)), len(y), int(y.sum()))
            fy.append(per_year[(year, portion)][0])
        defs[f"mean per-year F1, FAR 2018-2025, {portion}"] = (None, float(np.mean(fy)))

    print(f"\n{'definition':<42}{'NEAR':>9}{'vs pub':>9}{'FAR':>9}{'vs pub':>9}")
    best = None
    for name, (near, far) in defs.items():
        ns = f"{near:9.2f}" if near is not None else f"{'—':>9}"
        nd = f"{near-PUB_NEAR:+9.2f}" if near is not None else f"{'—':>9}"
        print(f"{name:<42}{ns}{nd}{far:>9.2f}{far-PUB_FAR:+9.2f}")
        if near is not None:
            score = abs(near - PUB_NEAR) + abs(far - PUB_FAR)
            if best is None or score < best[1]:
                best = (name, score, near, far)

    print(f"\nclosest definition: {best[0]}  (NEAR {best[2]:.2f}, FAR {best[3]:.2f})")
    ok = abs(best[2] - PUB_NEAR) <= 1.0 and abs(best[3] - PUB_FAR) <= 1.0
    print("RESULT split_definition=" + ("RECONSTRUCTED" if ok else "NOT_RECONSTRUCTIBLE"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/lamda/raw"))
