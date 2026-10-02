# QUESTION:  does the closed form work on a rule ensemble that is not a Tsetlin machine -- rules learned by
#            RIPPER, a classical and entirely unrelated learner?
# SURPRISE:  yes, and it is what decides whether this is a paper about Tsetlin machines or a paper about
#            rule ensembles. The derivation needs only that a rule's output depend on HOW MANY of its
#            literals are unsatisfied, not which, and that the model sum its rules. Nothing in it is
#            TM-specific. research/b2-generality/ verified that on constructed models; constructed models
#            are a weak demonstration, because they were built by the same person who derived the formula.
#            A ruleset induced by a 1990s separate-and-conquer algorithm on real malware data was not.
#            There is also a boundary to respect, and getting it wrong would be the obvious error here.
#            RIPPER'S NATIVE PREDICTION IS A DISJUNCTION -- positive if ANY rule covers the instance -- and
#            a disjunction is not a sum, so a raw RIPPER ruleset is OUTSIDE the class, for the same reason
#            ordered rule lists are. What is inside the class is the weighted rule ensemble built on those
#            rules, score = sum_r w_r * 1[rule r fires], which is a standard interpretable model in its own
#            right. This experiment is explicit about using the second and not the first.
# ARMS:      three, and the first is the one that makes the rest trustworthy:
#              1. BRUTE FORCE on a small ensemble -- RIPPER refit on 14 features, every one of the 2^14
#                 coalitions enumerated, exact Shapley computed from the definition and compared against
#                 the closed form. This is an INDEPENDENT reimplementation in Python of a formula derived
#                 and implemented in Julia, so agreement also cross-checks the derivation itself rather
#                 than only this port of it.
#              2. the FULL ensemble on all features, where brute force is impossible, to show the closed
#                 form runs at realistic width and what it costs.
#              3. SAMPLED vs EXACT on that full ensemble -- the paper's central comparison, repeated on a
#                 non-TM learner. If the sampled estimator is as far from the truth as from itself here
#                 too, that result stops being a fact about Tsetlin machines.
# PASS/FAIL: the generality claim HOLDS if arm 1 agrees with brute force to floating-point tolerance
#            (1e-9, the bar the Julia test uses). It FAILS if there is any systematic disagreement, in
#            which case the paper's scope claim is wrong and must be narrowed to models we have actually
#            verified.
#
#   python research/ripper-exact/run.py [--rows 8000] [--small-features 14]
#
# A conjunction is the ceiling-1 case of the fuzzy clause vote: max(0, 1 - misses) = 1[misses = 0]. So the
# same closed form serves, with the ceiling fixed at one and a per-rule weight multiplying it. No separate
# derivation is needed and none is used.

import argparse
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SRC = Path("data/apigraph/data/gen_apigraph_drebin")
POOL = "2012-01to2012-12_selected.npz"

# ---------------------------------------------------------------------------------------------
# the closed form, reimplemented here from the derivation, not ported from the Julia source

_LOGFACT = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, 1 << 16)))])


def _logbinom(n, k):
    if n < 0 or k < 0 or k > n:
        return -np.inf
    return _LOGFACT[n] - _LOGFACT[k] - _LOGFACT[n - k]


def _unit_value(beta, g, d, ceil, from_c):
    """Shared Shapley value of a C-member (from_c) or a D-member, for one rule."""
    tot = g + d
    if tot == 0:
        return 0.0
    if from_c and g == 0:
        return 0.0
    if (not from_c) and d == 0:
        return 0.0
    limit = ceil if from_c else ceil - 1
    nc = g - 1 if from_c else g
    nd = d if from_c else d - 1
    acc = 0.0
    for r in range(tot):
        ldenom = _logbinom(tot - 1, r)
        s = 0.0
        for k in range(max(0, r - nd), min(r, nc) + 1):
            if beta + g + r - 2 * k <= limit:
                s += math.exp(_logbinom(nc, k) + _logbinom(nd, r - k) - ldenom)
        acc += s
    return acc / tot


def rule_shapley(phi, rule, weight, x, b, ceil=1):
    """
    Accumulate one rule's exact Shapley values into phi.

    `rule` is a list of (feature, want) pairs: the literal is satisfied when the input equals `want`.
    """
    beta = 0
    C, D = set(), set()
    for f, want in rule:
        sx = (x[f] == want)
        sb = (b[f] == want)
        if sx and sb:
            pass                       # A: satisfied either way, contributes nothing
        elif (not sx) and (not sb):
            beta += 1                  # B: missed either way, a constant miss
        elif sx and not sb:
            C.add(f)
        else:
            D.add(f)
    both = C & D                       # a feature appearing with both polarities always misses once
    beta += len(both)
    C -= both
    D -= both
    g, d = len(C), len(D)
    if g + d == 0:
        return
    if g:
        v = weight * _unit_value(beta, g, d, ceil, True)
        for f in C:
            phi[f] += v
    if d:
        v = weight * _unit_value(beta, g, d, ceil, False)
        for f in D:
            phi[f] -= v


def exact_shapley(rules, weights, x, b, width):
    phi = np.zeros(width)
    for rule, w in zip(rules, weights):
        rule_shapley(phi, rule, w, x, b)
    return phi


def score_one(rules, weights, x):
    return float(sum(w for rule, w in zip(rules, weights)
                     if all(x[f] == want for f, want in rule)))


def score_matrix(rules, weights, X):
    X = np.asarray(X)
    out = np.zeros(X.shape[0])
    for rule, w in zip(rules, weights):
        m = np.ones(X.shape[0], dtype=bool)
        for f, want in rule:
            m &= (X[:, f] == want)
        out += w * m
    return out


# ---------------------------------------------------------------------------------------------

def brute_force_shapley(rules, weights, x, b, feats):
    """Shapley from the definition, over all subsets of `feats`. Exponential; small n only."""
    n = len(feats)
    idx = {f: i for i, f in enumerate(feats)}
    phi = np.zeros(len(x))

    def v(mask):
        z = b.copy()
        for f in feats:
            if mask >> idx[f] & 1:
                z[f] = x[f]
        return score_one(rules, weights, z)

    cache = {m: v(m) for m in range(1 << n)}
    for f in feats:
        i = idx[f]
        rest = [j for j in range(n) if j != i]
        total = 0.0
        for k in range(n):
            for comb in combinations(rest, k):
                mask = 0
                for j in comb:
                    mask |= 1 << j
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                total += w * (cache[mask | (1 << i)] - cache[mask])
        phi[f] = total
    return phi


def fit_ripper(X, y, seed=0):
    import wittgenstein as lw
    import pandas as pd
    clf = lw.RIPPER(random_state=seed, max_rules=40)
    df = pd.DataFrame(X.astype(np.int8), columns=[f"f{i}" for i in range(X.shape[1])])
    clf.fit(df, y)
    rules = []
    for rule in clf.ruleset_.rules:
        conds = []
        for c in rule.conds:
            f = int(str(c.feature)[1:])
            conds.append((f, int(c.val)))
        if conds:
            rules.append(conds)
    return rules


def weight_rules(rules, X, y):
    """score = sum_r w_r 1[rule fires]: a weighted rule ensemble, which IS in the class."""
    from sklearn.linear_model import LogisticRegression
    A = np.column_stack([
        np.all([(X[:, f] == want) for f, want in rule], axis=0) for rule in rules
    ]).astype(np.float64)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(A, y)
    return lr.coef_[0].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8000)
    ap.add_argument("--small-features", type=int, default=14)
    args = ap.parse_args()

    f = np.load(SRC / POOL, allow_pickle=True)
    X = f["X_train"].astype(np.int8)
    y = (f["y_train"].astype(np.int64) > 0).astype(np.int8)
    rng = np.random.default_rng(0)
    sel = rng.permutation(len(y))[:args.rows]
    X, y = X[sel], y[sel]
    width = X.shape[1]
    print(f"APIGraph pool subsample: {X.shape[0]:,} rows x {width} features, "
          f"{100*y.mean():.1f}% malware\n")

    # ---- arm 1: brute force on a small ensemble ----
    from sklearn.feature_selection import chi2
    sc, _ = chi2(X, y)
    sc = np.nan_to_num(sc)
    small = np.argsort(-sc)[:args.small_features]
    Xs = np.zeros_like(X)
    Xs[:, small] = X[:, small]                 # keep only those columns informative
    print(f"ARM 1 -- brute force verification on {args.small_features} features")
    rules_s = fit_ripper(Xs[:, small], y)
    rules_s = [[(small[f], want) for f, want in rule] for rule in rules_s]
    if not rules_s:
        print("  RIPPER induced no rules on the reduced problem; cannot verify", file=sys.stderr)
        return 1
    w_s = weight_rules(rules_s, X, y)
    print(f"  {len(rules_s)} rules, "
          f"{np.mean([len(r) for r in rules_s]):.1f} conditions each")

    worst = 0.0
    checks = 0
    for trial in range(6):
        i, j = rng.integers(0, len(y), 2)
        xi, bj = X[i].astype(int), X[j].astype(int)
        cf = exact_shapley(rules_s, w_s, xi, bj, width)
        bf = brute_force_shapley(rules_s, w_s, xi, bj, list(small))
        worst = max(worst, float(np.max(np.abs(cf - bf))))
        gap = score_one(rules_s, w_s, xi) - score_one(rules_s, w_s, bj)
        eff = abs(float(cf.sum()) - gap)
        worst = max(worst, eff)
        checks += 1
    print(f"  max |closed form - brute force| over {checks} (instance, background) pairs, "
          f"including the efficiency check: {worst:.3e}")
    print(f"RESULT brute_force max_abs_diff={worst:.3e} n_pairs={checks} "
          f"n_rules={len(rules_s)} verdict={'EXACT' if worst < 1e-9 else 'FAIL'}")
    if worst >= 1e-9:
        print("  FAIL: the closed form does not reproduce brute force on this model class.")
        return 1
    print("  EXACT: agreement at floating-point tolerance, on rules this project did not construct.\n")

    # ---- arm 2: the full ensemble ----
    print("ARM 2 -- the full ensemble, where brute force is impossible")
    t0 = time.time()
    rules = fit_ripper(X, y)
    print(f"  RIPPER induced {len(rules)} rules over {width} features in {time.time()-t0:.0f}s; "
          f"{np.mean([len(r) for r in rules]):.1f} conditions each")
    w = weight_rules(rules, X, y)
    used = sorted({f for rule in rules for f, _ in rule})
    print(f"  the ensemble reads {len(used)} distinct features")

    bg = X[rng.permutation(len(y))[:100]].astype(int)
    ex = X[rng.permutation(len(y))[:100]].astype(int)
    t0 = time.time()
    imp = np.zeros(width)
    for xi in ex:
        acc = np.zeros(width)
        for bj in bg:
            acc += exact_shapley(rules, w, xi, bj, width)
        imp += np.abs(acc / len(bg))
    imp /= len(ex)
    dt = time.time() - t0
    print(f"  exact attribution over 100x100: {dt:.1f}s, "
          f"{int((imp != 0).sum())} features with nonzero value")
    print(f"RESULT full n_rules={len(rules)} n_features_used={len(used)} "
          f"exact_seconds={dt:.1f} nonzero={int((imp != 0).sum())}")

    # ---- arm 3: sampled vs exact, on a non-TM model ----
    print("\nARM 3 -- sampled against exact, same model, same rows")
    import shap
    scorer = lambda Z: score_matrix(rules, w, Z)

    def topk(v, k):
        k = min(k, v.size)
        return np.argpartition(-v, k - 1)[:k]

    def jac(a, b):
        sa, sb = set(a.tolist()), set(b.tolist())
        u = len(sa | sb)
        return 1.0 if u == 0 else 1 - len(sa & sb) / u

    np.random.seed(0)
    t0 = time.time()
    expl = shap.KernelExplainer(scorer, bg.astype(float))
    v1 = np.abs(np.asarray(expl.shap_values(ex.astype(float), nsamples=100, silent=True))).mean(axis=0)
    np.random.seed(1)
    v2 = np.abs(np.asarray(expl.shap_values(ex.astype(float), nsamples=100, silent=True))).mean(axis=0)
    print(f"  two sampled runs at nsamples=100: {time.time()-t0:.0f}s")

    for k in (20, 50):
        a = jac(topk(v1, k), topk(imp, k))
        b = jac(topk(v1, k), topk(v2, k))
        print(f"  k={k:3d}   sampled vs EXACT {a:.3f}   sampled vs ITSELF {b:.3f}")
        print(f"RESULT arm3 k={k} sampled_vs_exact={a:.4f} sampled_vs_itself={b:.4f}")

    print("\nIf sampled-vs-exact is comparable to sampled-vs-itself here too, the variance-not-bias")
    print("finding is a property of the estimator at this budget and not of Tsetlin machines.")
    print("\nNote on scope: RIPPER's own prediction is a DISJUNCTION over its rules, which is not a sum")
    print("and so is outside the class, exactly as ordered rule lists are. What is verified above is the")
    print("weighted rule ensemble built on those rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
