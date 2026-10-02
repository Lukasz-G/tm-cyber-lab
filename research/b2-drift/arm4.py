# QUESTION:  on ONE fixed model, with the data and the attribution target held fixed, how far does
#            shap.KernelExplainer at nsamples=100 land from the exact Shapley values it is estimating?
# SURPRISE:  this is the arm no published work can run, because it needs a model whose exact Shapley
#            values are computable in closed form. research/b0-noise-floor/ showed the published
#            explanation-drift figure sits at its own noise floor, but it showed that by re-running the
#            estimator and watching the answer move -- it never had a ground truth to compare against.
#            Here there is one. If the sampled top-100 set overlaps the exact top-100 set well, then
#            nsamples=100 was adequate despite appearances and the estimator-variance explanation is
#            wrong (a pre-registered falsification condition). If it does not, the reported drift was
#            measuring the sampler.
# ARMS:      three, and the third is what makes the comparison honest:
#              a. sampled at nsamples=100 vs EXACT -- the published budget against ground truth.
#              b. sampled at nsamples=100 vs sampled at nsamples=100, different seed -- the
#                 estimator's own reproducibility, which bounds how much of (a) is variance rather
#                 than bias.
#              c. sampled at a LARGER budget vs exact -- without it, a poor (a) could be blamed on
#                 KernelExplainer being unsuitable here, not on the budget being too small.
#                 If (c) improves on (a), the budget is the cause, which is the claim.
# PASS/FAIL: not a gate. The pre-registered falsification is explicit: if sampled and exact agree
#            closely at nsamples=100, the estimator-variance hypothesis is wrong and that is reported.
#            The run is invalid and not informative if Python's own re-implementation of the clause
#            vote disagrees with the Julia scores exported alongside the model -- that is asserted
#            before any explaining happens, because a comparison against a mis-scored model measures
#            the port.
#
#   julia --project=. -t 16 research/b2-drift/export_arm4.jl 2016 6   # writes data/b2-arm4/
#   python research/b2-drift/arm4.py [--nsamples 100] [--big 2000]
#
# Budget note: KernelExplainer's own default for M features is 2*M + 2048, which at LAMDA's 4,561 is
# 11,170 coalitions. The LAMDA release uses 100. Arm (c) runs an intermediate budget and not the
# full default purely for wall clock, and says so in the output, because the point it has to make is
# only that the direction of travel is toward exact.

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import kendalltau

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join("data", "b2-arm4")


def load():
    with open(os.path.join(DATA, "model.json"), encoding="utf-8") as fh:
        model = json.load(fh)
    with open(os.path.join(DATA, "rows.json"), encoding="utf-8") as fh:
        rows = json.load(fh)
    bg = np.asarray(rows["background"], dtype=np.uint8)
    ex = np.asarray(rows["explained"], dtype=np.uint8)
    return model, bg, ex


def build_scorer(model):
    """
    The FPTM clause vote, re-implemented over a dense 0/1 matrix.

    A clause votes max(0, ceiling - misses), where misses counts its included literals that the input
    does not satisfy: a positive literal misses when the feature is 0, a negated literal when it is 1.
    The class score is the signed sum over both polarity banks. Nothing here is approximate -- it is
    the same arithmetic the Julia evaluator does, which is exactly why it can be checked against it.
    """
    width = model["width"]
    pos_idx, pos_neg, pos_ceil, signs = [], [], [], []
    for c in model["clauses"]:
        f = np.asarray(c["features"], dtype=np.int64) - 1          # Julia is 1-based
        n = np.asarray(c["negated"], dtype=bool)
        pos_idx.append(f)
        pos_neg.append(n)
        pos_ceil.append(int(c["ceiling"]))
        signs.append(int(c["sign"]))

    def score(X):
        X = np.asarray(X, dtype=np.uint8).reshape(-1, width)
        out = np.zeros(X.shape[0], dtype=np.float64)
        for f, n, ceil, sgn in zip(pos_idx, pos_neg, pos_ceil, signs):
            if f.size == 0:
                out += sgn * ceil          # an empty clause votes its ceiling on every input
                continue
            vals = X[:, f].astype(bool)
            missed = np.where(n, vals, ~vals).sum(axis=1)   # negated literal misses when value is 1
            out += sgn * np.maximum(0, ceil - missed)
        return out

    return score


def topk(imp, k):
    k = min(k, imp.size)
    return np.argpartition(-imp, k - 1)[:k]


def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    u = len(sa | sb)
    return 1.0 if u == 0 else 1 - len(sa & sb) / u


def kendall(a, b, imp_a, imp_b):
    union = sorted(set(a.tolist()) | set(b.tolist()))
    tau, _ = kendalltau([float(imp_a[i]) for i in union], [float(imp_b[i]) for i in union])
    return 1.0 if tau is None or np.isnan(tau) else (1 - tau) / 2


def sampled_importance(score, bg, ex, nsamples, seed):
    """mean over explained rows of |phi|, matching the reference's aggregation."""
    import shap

    np.random.seed(seed)
    expl = shap.KernelExplainer(score, bg)
    vals = expl.shap_values(ex, nsamples=nsamples, silent=True)
    vals = np.asarray(vals)
    if vals.ndim == 3:               # (rows, features, outputs) for some shap versions
        vals = vals[:, :, 0]
    return np.abs(vals).mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=100, help="the published budget")
    ap.add_argument("--big", type=int, default=2000, help="the larger budget for arm (c)")
    ap.add_argument("--topk", type=int, nargs="+", default=[100, 1000])
    args = ap.parse_args()

    model, bg, ex = load()
    score = build_scorer(model)
    exact = np.asarray(model["exact_importance"], dtype=np.float64)

    print(f"{model['year']}-{model['month']:02d}  width {model['width']}  "
          f"{len(model['clauses'])} clauses  background {bg.shape[0]}  explained {ex.shape[0]}")

    # --- the port check. Nothing below means anything if this fails. ---
    got_ex = score(ex)
    got_bg = score(bg)
    want_ex = np.asarray(model["score_ex"], dtype=np.float64)
    want_bg = np.asarray(model["score_bg"], dtype=np.float64)
    d = max(np.abs(got_ex - want_ex).max(), np.abs(got_bg - want_bg).max())
    print(f"port check: max |python score - julia score| = {d:g}")
    assert d == 0.0, (
        f"the Python clause vote does not reproduce the Julia scores (max diff {d:g}). "
        "Arm 4 compares an estimator against exact values ON THE SAME MODEL; if the port is wrong "
        "the comparison measures the port. Fix before reading any number below."
    )
    print("RESULT port_check max_abs_diff=0 exact")

    print(f"exact: {int((exact != 0).sum())} of {exact.size} features have nonzero attribution\n")

    runs = {}
    for label, ns, seed in (("sampled n=%d" % args.nsamples, args.nsamples, 0),
                            ("sampled n=%d (seed 2)" % args.nsamples, args.nsamples, 1),
                            ("sampled n=%d" % args.big, args.big, 0)):
        t0 = time.time()
        runs[label] = sampled_importance(score, bg, ex, ns, seed)
        print(f"  {label:26s} {time.time() - t0:7.1f}s")

    a = runs["sampled n=%d" % args.nsamples]
    a2 = runs["sampled n=%d (seed 2)" % args.nsamples]
    c = runs["sampled n=%d" % args.big]

    print()
    print(f"{'comparison':42s} {'k':>6} {'jaccard':>9} {'kendall':>9}")
    print("-" * 70)
    for k in args.topk:
        te = topk(exact, k)
        for name, x, y, ix, iy in (
            (f"a. sampled n={args.nsamples} vs EXACT", a, exact, a, exact),
            (f"b. sampled n={args.nsamples} vs itself, seed 2", a, a2, a, a2),
            (f"c. sampled n={args.big} vs EXACT", c, exact, c, exact),
        ):
            tx, ty = topk(ix, k), topk(iy, k)
            j, kd = jaccard(tx, ty), kendall(tx, ty, ix, iy)
            print(f"{name:42s} {k:6d} {j:9.3f} {kd:9.3f}")
            print(f"RESULT arm4={name} k={k} jaccard={j:.4f} kendall={kd:.4f}")
        print()

    print("-" * 70)
    print("Read (a) against (b): (a) is how far the published budget lands from ground truth, and")
    print("(b) is how far it lands from itself. If they are comparable, the error is variance rather")
    print("than bias. Read (c) against (a): if more coalitions move it toward exact, the budget is")
    print(f"the cause. KernelExplainer's own default here would be 2*{model['width']}+2048 = "
          f"{2 * model['width'] + 2048}.")


if __name__ == "__main__":
    main()
