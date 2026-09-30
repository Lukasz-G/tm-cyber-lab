# QUESTION:  with BOTH sides swept and the threshold chosen honestly, is FPTM still inside the
#            gradient-boosting precision/recall frontier on APIGraph -- and by how much?
# SURPRISE:  yes, either way. Three recorded unfairnesses all point the same direction, IN OUR FAVOUR:
#            earlier experiments compared a swept FPTM against boosters at their default 0.5 cut, took
#            dominance counts from a single seed while averaging F1 over five, and chose FPTM's
#            threshold on the same rows it was scored on. The standing conclusion -- that a 20-200
#            clause FPTM lies inside the booster frontier, with nine explanations eliminated -- was
#            reached under those conditions. Removing them can only move the comparison against us, so
#            the surprise would be the conclusion SURVIVING unchanged at full strength, which would
#            make it much stronger than it currently is. If instead the gap widens, the honest number
#            is larger than published anywhere in this repo and has to replace it.
# ARMS:      three models on identical rows and an identical 80/20 split of the 2012 pool:
#              1. FPTM, margins swept over their whole range.
#              2. LightGBM, predicted probabilities swept over their whole range.
#              3. XGBoost, likewise.
#            and three ways of reading them, because the metric is the thing in dispute:
#              a. AVERAGE PRECISION -- threshold-free, frontier against frontier, the honest headline.
#              b. VALIDATION-CHOSEN threshold -- picked on the held-out 20% of the 2012 pool, applied
#                 unchanged to every test year. The number a deployment would actually get.
#              c. ORACLE best-F1 on the test year itself -- kept only to quantify how much (b) costs,
#                 since every earlier F1 in this project is an oracle value and readers of those need
#                 to know the size of the correction.
#            Dominance is recomputed on EVERY seed rather than seed 1, with spread reported.
# PASS/FAIL: not a gate. It REPLACES numbers rather than deciding anything. The specific claim under
#            test is the standing conclusion that FPTM is dominated on APIGraph at every configuration
#            tried: that stands if boosters keep a higher average precision on every year once both
#            are swept, and must be softened if FPTM matches or beats them on any year.
#
#   julia --project=. -t 16 research/b5-fairness/export_margins.jl 5   # writes data/b5-fairness/
#   python research/b5-fairness/run.py
#
# The 80/20 split means this trains on less data than research/b5-apigraph/ did, so absolute values are
# not comparable with that experiment. Both models here see the same 80%, which is what makes THIS
# comparison fair. The port between the two languages is checked before anything is computed: the label
# vectors Julia exported must match the ones Python loads from the npz, row for row, or the two sides
# are not looking at the same data.

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SRC = Path("data/apigraph/data/gen_apigraph_drebin")
POOL = "2012-01to2012-12_selected.npz"
MARGINS = Path("data/b5-fairness/fptm_margins.json")
YEARS = [str(y) for y in range(2013, 2019)]


def load_npz(path):
    f = np.load(path, allow_pickle=True)
    X = f["X_train"].astype(np.int8)
    y = (f["y_train"].astype(np.int64) > 0).astype(np.int8)
    return X, y


def load_year(year):
    Xs, ys = [], []
    for p in sorted(SRC.glob(f"{year}-*_selected.npz")):
        X, y = load_npz(p)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def best_f1(y, s):
    """Oracle: the best F1 any threshold on THIS data achieves. An upper bound, not an operating point."""
    prec, rec, _ = precision_recall_curve(y, s)
    f = 2 * prec * rec / np.maximum(1e-12, prec + rec)
    return 100 * float(np.nanmax(f))


def pick_threshold(y, s):
    """The threshold maximising F1 on the validation slice. Returned to be applied elsewhere, unchanged."""
    prec, rec, thr = precision_recall_curve(y, s)
    f = 2 * prec * rec / np.maximum(1e-12, prec + rec)
    f = np.nan_to_num(f[:-1])          # last point of the curve has no threshold
    return float(thr[int(np.argmax(f))]) if len(thr) else 0.0


def f1_at(y, s, t):
    return 100 * f1_score(y, (s >= t).astype(np.int8), zero_division=0)


def frontier_beats(y, s_a, y_b=None, s_b=None):
    """
    Does frontier A reach ANY point strictly outside frontier B? Interpolates B's precision at A's
    recall levels, so this is a curve-against-curve test rather than a point comparison.
    """
    pa, ra, _ = precision_recall_curve(y, s_a)
    pb, rb, _ = precision_recall_curve(y, s_b)
    order = np.argsort(rb)
    rb_s, pb_s = rb[order], pb[order]
    pb_at_ra = np.interp(ra, rb_s, pb_s)
    return bool(np.any(pa > pb_at_ra + 1e-9)), float(np.max(pa - pb_at_ra))


def main():
    if not MARGINS.exists():
        print(f"missing {MARGINS} -- run export_margins.jl first", file=sys.stderr)
        return 1
    M = json.loads(MARGINS.read_text())
    nseeds = M["n_seeds"]

    Xpool, ypool = load_npz(SRC / POOL)
    # The split comes from Julia's exported indices, not from re-deriving the rule here. A contiguous
    # 80/20 was tried first and gave a validation slice with no malware at all, because the pool is
    # ordered by class; taking the indices removes any chance of the two sides disagreeing.
    tr_i = np.asarray(M["train_idx"], dtype=np.int64) - 1
    va_i = np.asarray(M["val_idx"], dtype=np.int64) - 1
    Xtr, ytr = Xpool[tr_i], ypool[tr_i]
    Xva, yva = Xpool[va_i], ypool[va_i]
    ntr = len(tr_i)
    assert yva.sum() > 0, "validation slice has no malware; a threshold cannot be selected"

    # --- port check: are both languages looking at the same rows? ---
    jl_val = np.asarray(M["labels"]["val"], dtype=np.int8)
    assert len(jl_val) == len(yva) and np.array_equal(jl_val, yva), (
        "the validation labels Julia exported do not match the ones loaded here, so the two sides are "
        "not on the same rows and nothing below is a comparison. Check that the .tmx was built from "
        "this npz and that the 80/20 split rule matches."
    )
    print(f"train {ntr:,} ({100*ytr.mean():.1f}% malware)  val {len(yva):,} ({100*yva.mean():.1f}%)")
    print(f"port check: validation labels match Julia's export, {len(yva):,} rows")
    print("RESULT port_check val_labels=match")

    evals = {}
    for y in YEARS:
        X, lab = load_year(y)
        jl = np.asarray(M["labels"][y], dtype=np.int8)
        assert np.array_equal(jl, lab), f"label mismatch on {y} between Julia export and npz"
        evals[y] = (X, lab)
    print(f"port check: all {len(YEARS)} test years match row for row\n")

    # --- boosters, on the identical 80% ---
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
        boosters[name] = {
            "val": clf.predict_proba(Xva)[:, 1],
            **{y: clf.predict_proba(evals[y][0])[:, 1] for y in YEARS},
        }
        print(f"trained {name}")
    print()

    # --- (a) average precision, threshold-free ---
    print("(a) AVERAGE PRECISION -- threshold-free, frontier against frontier")
    print(f"{'year':6} {'FPTM (mean±sd)':>18} {'LightGBM':>10} {'XGBoost':>10} {'best booster - FPTM':>21}")
    print("-" * 70)
    ap_gap = []
    for y in YEARS:
        fp = [100 * average_precision_score(evals[y][1], np.asarray(M["margins"][y][s]))
              for s in range(nseeds)]
        al = 100 * average_precision_score(evals[y][1], boosters["lightgbm"][y])
        ax = 100 * average_precision_score(evals[y][1], boosters["xgboost"][y])
        gap = max(al, ax) - float(np.mean(fp))
        ap_gap.append(gap)
        print(f"{y:6} {np.mean(fp):11.2f}±{np.std(fp):5.2f} {al:10.2f} {ax:10.2f} {gap:+21.2f}")
        print(f"RESULT ap year={y} fptm={np.mean(fp):.2f} sd={np.std(fp):.2f} "
              f"lgb={al:.2f} xgb={ax:.2f} gap={gap:+.2f}")
    print(f"\n  mean gap to the better booster: {np.mean(ap_gap):+.2f} AP\n")
    print(f"RESULT ap mean_gap={np.mean(ap_gap):+.2f}\n")

    # --- (b) validation-chosen threshold, and (c) the oracle it replaces ---
    print("(b) VALIDATION-CHOSEN threshold, applied unchanged   vs   (c) ORACLE best-F1 on the test year")
    print(f"{'year':6} {'FPTM val':>9} {'FPTM oracle':>12} {'LGB val':>9} {'LGB oracle':>11} "
          f"{'XGB val':>9} {'XGB oracle':>11}")
    print("-" * 78)
    cost_fptm, cost_lgb, cost_xgb = [], [], []
    for y in YEARS:
        lab = evals[y][1]
        fv, fo = [], []
        for s in range(nseeds):
            t = pick_threshold(yva, np.asarray(M["margins"]["val"][s]))
            fv.append(f1_at(lab, np.asarray(M["margins"][y][s]), t))
            fo.append(best_f1(lab, np.asarray(M["margins"][y][s])))
        tl = pick_threshold(yva, boosters["lightgbm"]["val"])
        tx = pick_threshold(yva, boosters["xgboost"]["val"])
        lv, lo = f1_at(lab, boosters["lightgbm"][y], tl), best_f1(lab, boosters["lightgbm"][y])
        xv, xo = f1_at(lab, boosters["xgboost"][y], tx), best_f1(lab, boosters["xgboost"][y])
        cost_fptm.append(np.mean(fo) - np.mean(fv))
        cost_lgb.append(lo - lv)
        cost_xgb.append(xo - xv)
        print(f"{y:6} {np.mean(fv):9.2f} {np.mean(fo):12.2f} {lv:9.2f} {lo:11.2f} {xv:9.2f} {xo:11.2f}")
        print(f"RESULT thresh year={y} fptm_val={np.mean(fv):.2f} fptm_oracle={np.mean(fo):.2f} "
              f"lgb_val={lv:.2f} lgb_oracle={lo:.2f} xgb_val={xv:.2f} xgb_oracle={xo:.2f}")
    print(f"\n  what the oracle was worth: FPTM {np.mean(cost_fptm):+.2f} F1, "
          f"LightGBM {np.mean(cost_lgb):+.2f}, XGBoost {np.mean(cost_xgb):+.2f}")
    print(f"RESULT oracle_cost fptm={np.mean(cost_fptm):.2f} lgb={np.mean(cost_lgb):.2f} "
          f"xgb={np.mean(cost_xgb):.2f}\n")

    # --- dominance, on every seed ---
    print("DOMINANCE, recomputed on every seed: does FPTM's frontier reach outside the booster's?")
    print(f"{'year':6} {'beats LGB somewhere':>22} {'beats XGB somewhere':>22} {'max precision excess':>21}")
    print("-" * 74)
    for y in YEARS:
        lab = evals[y][1]
        bl, bx, ex = 0, 0, []
        for s in range(nseeds):
            m = np.asarray(M["margins"][y][s])
            okl, dl = frontier_beats(lab, m, s_b=boosters["lightgbm"][y])
            okx, dx = frontier_beats(lab, m, s_b=boosters["xgboost"][y])
            bl += okl
            bx += okx
            ex.append(max(dl, dx))
        print(f"{y:6} {f'{bl}/{nseeds}':>22} {f'{bx}/{nseeds}':>22} {100*np.mean(ex):+20.2f}")
        print(f"RESULT dominance year={y} beats_lgb={bl}/{nseeds} beats_xgb={bx}/{nseeds} "
              f"max_prec_excess={100*np.mean(ex):+.2f}")

    print("\n" + "-" * 74)
    print("(a) is the honest headline: both sides swept, no threshold anywhere in it. (b) is what a")
    print("deployment gets. (c) exists only to size the correction owed to every earlier F1 in this")
    print("repo, all of which are oracle values. 'Beats somewhere' counts a frontier reaching outside")
    print("the booster's at ANY recall, which is a weaker and more generous test than dominance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
