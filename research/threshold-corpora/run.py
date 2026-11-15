# QUESTION:  how much does the decision-threshold rule change reported F1 on every corpus, when one code
#            path, one split and one set of rules is used on all of them, and does the training class
#            balance change the answer?
# SURPRISE:  predicted before the run, on every corpus and for all three models:
#            (1) the oracle gap (oracle F1 minus F1 at the validation threshold) is small on the
#                in-distribution split and larger on later years, and on AndroZoo it grows from 2020 to 2021;
#                it differs between FPTM and the boosters as on LAMDA in research/threshold-bias;
#            (2) the training condition moves the default operating point and not the best F1 or AP. Prior
#                evidence is research/b5-imbalance (FPTM only, APIGraph only): balancing moved default recall
#                from 55.7 to 75.5 while best swept F1 stayed at 74.69 against 74.09. Predicted here for all
#                three models: oracle F1 and AP do not change between B0, U and O; default minus validation
#                changes; oracle minus validation does not change.
#            Predictions (1) and (2) and both PASS/FAIL readings were reworded on 2026-11-13 for the B0/U/O
#            design, after a code review and after AndroZoo results from the superseded B0/B1/B2/W design had
#            been read. No threshold of the earlier wording was changed. The two clauses of (1) about AndroZoo's
#            growth from 2020 to 2021 and about FPTM against the boosters were in the first version of this
#            header (2026-11-13 00:16 UTC), before any AndroZoo result was read (02:46 UTC); readings (1b) and
#            (1c), which test them, were added on 2026-11-13 with their numbers fixed before any full result
#            is read.
# CONDITIONS (the same text in run.jl, run.py and research/b2-corpora): identical for FPTM, LightGBM and
#            XGBoost; no class weights, sample weights or scale_pos_weight in any model.
#              B0  the training split cut to exactly 1:9 on every corpus, LAMDA included: k = min(M, G div 9)
#                  malware rows and 9k goodware rows, drawn at random without replacement (the rule of
#                  research/b2-corpora). No other resampling. Every set meant to be at 10% (the B0 validation
#                  set, validation_te, the in-distribution split and every test year) is cut by the same rule,
#                  with a cap folded in as k = min(M, G div 9, cap div 10).
#              U   B0 with the majority class undersampled at random, without replacement, to the minority
#                  count: 50/50, size 2 x minority.
#              O   B0 with the minority class oversampled to the majority count: each minority row repeated
#                  floor(R) times (R = majority/minority), plus round((R - floor R) x minority) further
#                  minority rows drawn without replacement: 50/50, size 2 x majority. At exactly 10% that is
#                  x9.
#            Majority and minority are decided per training set, not assumed to be goodware and malware.
#            Seed s draws the U and O resample s and trains model s, for FPTM, LightGBM and XGBoost alike,
#            5 seeds. B0 has no resampling, and O on an exact 1:9 set draws no extra rows (R = 9), so under
#            both the training rows are identical for every seed: their boosters are trained once (they are
#            deterministic) and recorded as such. U boosters get 5 seeds = 5 draws. FPTM keeps 5 seeds in
#            every condition.
# ARMS:      the three conditions B0, U and O, test share 10% throughout. U against B0 changes the class
#            share by removing majority rows; O against B0 changes it by repeating minority rows; U against O
#            separates the two routes to the same 50/50 share. Each condition runs FPTM (margins from
#            run.jl), LightGBM and XGBoost (here). Six threshold rules per model: oracle, default, and
#            validation / rate on the validation set built like the condition's training set, and
#            validation_te / rate_te on the validation region resampled to the test share. Both validation
#            sets are reported; neither is preferred.
# PASS/FAIL: not a gate; per corpus and model, evaluated under each condition and under each validation rule
#            (validation and validation_te), every combination reported. A difference COUNTS when it exceeds
#            1.0 F1 point and its 95% paired-bootstrap interval excludes zero (the deciding criterion, below).
#            Reading (1) SURVIVES for a combination if the oracle gap on the in-distribution split is at most
#            2.0 F1 points and its mean over the later years exceeds that in-distribution gap by a difference
#            that counts. It is NOT SUPPORTED if the difference does not count (roughly constant), and
#            UNRESOLVED if the in-distribution gap exceeds 2.0.
#            Reading (1b), AndroZoo only: the oracle gap in 2021 exceeds the oracle gap in 2020 by a difference
#            that counts; it SURVIVES for a condition and validation rule if this holds for all three models.
#            Reading (1c), every corpus: the mean later-year oracle gap of each booster exceeds FPTM's by a
#            difference that counts, the direction research/threshold-bias found on LAMDA (FPTM 7.07 against
#            18.97 and 17.50); it SURVIVES for a condition and validation rule if this holds for both boosters.
#            Reading (2) SURVIVES for a corpus and model if, on every period, the differences between B0, U
#            and O in oracle F1, in AP and in oracle minus validation do not count, while the difference in
#            default minus validation counts on at least one period. It is NOT SUPPORTED otherwise.
#            Two noise criteria are computed for every difference and printed side by side. Stated: the
#            difference exceeds 1.0 F1 point and the range across the 5 seeds or draws of each side (a booster
#            trained once under B0 or O has no range). Bootstrap: the difference exceeds 1.0 F1 point and its
#            95% interval from a paired bootstrap over the shared test rows (2,000 resamples; scores averaged
#            over seeds or draws for every model; thresholds fixed on the validation sets) excludes zero. The
#            bootstrap criterion decides every reading and both controls, for every model (decided 2026-11-13
#            by Lukasz before any full result was read); the stated criterion is printed beside it as
#            secondary.
#            `--evaluate` applies both readings to the three conditions of a corpus.
# CONTROLS:  negative: on the in-distribution split at 1:9 the oracle gap is near zero (reading (1): at most
#            2.0 F1 points; under the bootstrap criterion, its upper 95% bound at most 2.0), with a verdict
#            line for every condition, rule and model. Positive, a planted prior shift with no covariate
#            drift: the in-distribution 1:9
#            set cut to exactly 1:1 from the same rows (all its malware rows and as many of its goodware rows,
#            drawn without replacement), scored from the margins and probabilities already computed, with no
#            extra training. Expected, fixed before the run: with the threshold chosen on the 1:9 validation
#            set (validation_te), the oracle gap on the 1:1 set exceeds the oracle gap on the 1:9
#            in-distribution set by a difference that counts under the bootstrap criterion (PASS/FAIL),
#            for every model and condition. The same comparison with the condition's own validation set (validation) is
#            reported beside it; under U and O that set is already 1:1, so there it is not a planted shift. A
#            control that fails is reported as a failure, and nothing is tuned to make it pass.
# EXPLORATORY (no pass/fail): the EM prior correction of Saerens, Latinne, Decaestecker, Neural
#            Computation 14(1), 2002, computed in --evaluate from the saved scores, with no retraining. Per
#            test period, starting from pi = pi_train (the condition's training share: 0.1 under B0, 0.5 under
#            U and O), every probability p is rescaled by the prior ratio, p' = (pi/pi_train) p /
#            [(pi/pi_train) p + ((1-pi)/(1-pi_train)) (1-p)], and pi is set to mean(p'), until |delta pi| <
#            1e-6 or 1,000 iterations. Reported per period: the estimated pi, and F1 of p' at 0.5 and at t_val,
#            the F1-optimal threshold chosen on validation in p-space at pi_train and applied unchanged to p'
#            (the label-free recalibration). This second rule replaces, before any full result, a vacuous one
#            that mapped the threshold through the same prior ratio as p and so reproduced the validation rule
#            exactly. The
#            boosters use their probabilities (the mean over draws under U); FPTM's seed-mean margin is first
#            Platt-scaled, fitted on the validation set built like the condition's training set.
# RESUMABLE: every unit (corpus, condition, model, seed) writes its output to a temporary file, renames it,
#            then writes a .sha256 sidecar. A unit whose output and matching sidecar exist, whose common hash
#            matches the current common content, and whose training rows, validation rows and settings hash
#            to the recorded values, is skipped on start. run.py reads only verified files.
# MEMORY:    before each job starts, MemAvailable (/proc/meminfo on Linux, the operating system's free-memory
#            figure elsewhere) must cover 1.25 x the measured peak RAM of that job class (peak_ram.json, from
#            the full-size timing runs); otherwise the job waits and re-checks every 60 s. It fails with an
#            error after 30 min of waiting, and at once if the need exceeds the machine's total memory.
#
#   julia --project=. research/threshold-corpora/run.jl <corpus> 1 5 --condition=B0|U|O [--test-share=0.10]
#   python -X utf8 research/threshold-corpora/run.py <corpus> --condition=.. [--test-share=..] [--threads=N] [--smoke | --timing --trees=N]
#   python -X utf8 research/threshold-corpora/run.py <corpus> --evaluate [--test-share=..] [--trees=N] [--smoke]
#
# The matrices are rebuilt from the same .tmx files and positions run.jl exported (the corpus directory and
# stems come from common.json, so a new corpus needs no change here). Every label vector is compared with
# Julia's row for row, and every FPTM seed file must carry the hash of the common content in common.json.
# Booster settings are those of research/b1-threshold-transfer and research/b5-fairness: LightGBM 5,000
# trees, learning rate 0.02, leaf-wise with 256 leaves; XGBoost 5,000 trees, learning rate 0.02, hist,
# depthwise with max_depth 6 (so at most 64 leaves; max_leaves=256 is passed as in the parents and does not
# bind). Row and column subsampling are off for both (LightGBM subsample 1.0, subsample_freq 0,
# colsample_bytree 1.0; XGBoost subsample 1.0, colsample_bytree 1.0) under every condition, so the boosters
# are deterministic given their training rows; random_state is set to the seed for the record only. n_jobs is
# the cores allotted per job (--threads=N or BOOSTER_THREADS, set by launch.sh; all cores if unset); it changes
# run time only, and is written into every results file.

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from tmcyber.tmx import read_meta, read_tmx  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
RULES = ("validation", "oracle", "default", "rate", "validation_te", "rate_te")
CONDITIONS = ("B0", "U", "O")
MODELS = ("fptm", "lightgbm", "xgboost")
N_SEEDS = 5
TOL = 1.0          # F1 points; a difference counts only above this AND above the seed range of each side
SMALL = 2.0        # reading (1): the in-distribution oracle gap must be at most this
CONTROL = "iid_1to1"   # positive control: the 1:9 iid set cut to exactly 1:1 from the same rows
PEAK = json.loads((HERE / "peak_ram.json").read_text(encoding="utf-8"))   # measured peaks, bound at import
MEM_FACTOR, MEM_POLL_S, MEM_MAX_WAIT_S = 1.25, 60, 1800
N_BOOT, BOOT_SEED = 2000, 20261113


def mem_available_mb():
    """MemAvailable in MB: /proc/meminfo on Linux, psutil elsewhere."""
    mi = Path("/proc/meminfo")
    if sys.platform.startswith("linux") and mi.exists():
        for ln in mi.read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1024
    import psutil
    return psutil.virtual_memory().available / 2**20


def mem_total_mb():
    """MemTotal in MB: /proc/meminfo on Linux, psutil elsewhere."""
    mi = Path("/proc/meminfo")
    if sys.platform.startswith("linux") and mi.exists():
        for ln in mi.read_text().splitlines():
            if ln.startswith("MemTotal:"):
                return int(ln.split()[1]) / 1024
    import psutil
    return psutil.virtual_memory().total / 2**20


def wait_for_memory(job, lang, corpus, cond):
    """Block until MemAvailable covers 1.25 x the measured peak of this job class; fail after 30 min or if it never can."""
    peak = PEAK.get(lang, {}).get(corpus, {}).get(cond, PEAK["default"][lang])
    need = MEM_FACTOR * peak
    if need > mem_total_mb():
        sys.exit(f"memory check {job}: need {need:.0f} MB exceeds total memory {mem_total_mb():.0f} MB")
    t_wait = time.time()
    while True:
        if time.time() - t_wait > MEM_MAX_WAIT_S:
            sys.exit(f"memory check {job}: need {need:.0f} MB not available after 30 min")
        avail = mem_available_mb()
        print(f"memory check {job}: need {need:.0f} MB (1.25 x {peak:.0f}), available {avail:.0f} MB"
              + ("" if avail >= need else f"; waiting {MEM_POLL_S}s"), flush=True)
        if avail >= need:
            return
        time.sleep(MEM_POLL_S)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verified(path):
    """True when `path` and its .sha256 sidecar exist and the sidecar matches the file."""
    path = Path(path)
    side = Path(str(path) + ".sha256")
    return path.exists() and side.exists() and side.read_text().strip() == sha256_file(path)


def rows_hash(tr, va):
    """Hash of a booster unit's training and validation rows (0-based positions into the pool)."""
    return hashlib.sha256(np.asarray(tr, dtype=np.int64).tobytes() + b"|" + np.asarray(va, dtype=np.int64).tobytes()).hexdigest()


def params_hash(name, n_trees, seed):
    """Hash of a booster's settings, without n_jobs, which changes run time only."""
    p = {k: v for k, v in booster_params(name, n_trees, seed).items() if k != "n_jobs"}
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()


def rebuild_results():
    """results.txt: every RESULT line of the verified results-*.txt and evaluation-*.txt files."""
    parts = [p for p in sorted(HERE.glob("results-*.txt")) + sorted(HERE.glob("evaluation-*.txt")) if verified(p)]
    allres = [ln for p in parts for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("RESULT")]
    (HERE / "results.txt").write_text("THRESHOLD RULES ON EVERY CORPUS -- RESULT lines of "
                                      + ", ".join(p.name for p in parts) + "\n\n" + "\n".join(allres) + "\n",
                                      encoding="utf-8")


def write_verified(path, write_fn):
    """Write through `write_fn(tmp)`, rename into place, then write the .sha256 sidecar."""
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)
    Path(str(path) + ".sha256").write_text(sha256_file(path))


def flag(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def run_tag(corpus, cond, te, smoke):
    """Directory tag shared with run.jl."""
    return f"{corpus}-{cond}-te{te}" + ("-smoke" if smoke else "")


def peak_ram_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2**20
    except Exception:  # noqa: BLE001
        return float("nan")


def load_stems(d, stems):
    Xs, ys = [], []
    for s in stems:
        Xs.append(read_tmx(d / f"{s}.tmx"))
        ys.append(np.asarray(read_meta(d / f"{s}.meta.arrow").column("label").to_pylist(), dtype=np.int8))
    return (Xs[0] if len(Xs) == 1 else np.vstack(Xs)), np.concatenate(ys)


def pick_threshold(y, s):
    """Threshold maximising F1 on (y, s); predictions are s >= threshold."""
    prec, rec, thr = precision_recall_curve(y, s)
    f = np.nan_to_num(2 * prec * rec / np.maximum(1e-12, prec + rec))[:-1]
    return float(thr[int(np.argmax(f))]) if len(thr) else 0.0


def rate_threshold(s, rate, fallback):
    """Threshold giving predicted-malware share `rate` on scores s."""
    return float(np.quantile(s, 1.0 - rate)) if 0.0 < rate < 1.0 else fallback


def f1_pred(y, pred):
    return 100 * f1_score(y, pred.astype(np.int8), zero_division=0)


def apply_rules(sva, yva, sva_te, yva_te, s, y, default_cut):
    """{rule: (f1, threshold, predicted-malware share)} for one model on one period.

    validation / rate use the validation set built like the condition's training set; validation_te /
    rate_te use the validation region resampled to the test share.
    """
    t_val = pick_threshold(yva, sva)
    t_val_te = pick_threshold(yva_te, sva_te)
    t_orc = pick_threshold(y, s)
    t_rate = rate_threshold(s, float(np.mean(sva >= t_val)), t_val)
    t_rate_te = rate_threshold(s, float(np.mean(sva_te >= t_val_te)), t_val_te)
    preds = {
        "validation": (s >= t_val, t_val),
        "oracle": (s >= t_orc, t_orc),
        "default": (s > default_cut, default_cut),       # strict: ties go to goodware
        "rate": (s >= t_rate, t_rate),
        "validation_te": (s >= t_val_te, t_val_te),
        "rate_te": (s >= t_rate_te, t_rate_te),
    }
    return {r: (f1_pred(y, p), t, float(np.mean(p))) for r, (p, t) in preds.items()}


def booster_threads():
    """Cores allotted to one booster job: --threads=N or BOOSTER_THREADS (set by launch.sh); all cores if unset."""
    v = flag("threads", os.environ.get("BOOSTER_THREADS", ""))
    return int(v) if v else -1


BOOSTER_THREADS = booster_threads()     # bound at import


def booster_params(name, n_estimators, seed):
    if name == "lightgbm":
        return dict(n_estimators=n_estimators, learning_rate=0.02, num_leaves=256, subsample=1.0,
                    subsample_freq=0, colsample_bytree=1.0, n_jobs=BOOSTER_THREADS, verbose=-1, random_state=seed)
    return dict(n_estimators=n_estimators, learning_rate=0.02, max_leaves=256, max_depth=6,
                grow_policy="depthwise", subsample=1.0, colsample_bytree=1.0, tree_method="hist", n_jobs=BOOSTER_THREADS,
                random_state=seed, eval_metric="logloss")


def train_booster(name, Xtr, ytr, n_estimators, seed):
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**booster_params(name, n_estimators, seed)).fit(Xtr, ytr)
    import xgboost as xgb
    return xgb.XGBClassifier(**booster_params(name, n_estimators, seed)).fit(Xtr, ytr)


def run_condition(corpus, cond, te_share, smoke, timing, n_trees):
    t_start = time.time()
    phase = {}
    mdir = Path("data/threshold-corpora") / run_tag(corpus, cond, te_share, smoke)
    cpath = mdir / "common.json"
    assert verified(cpath), f"{cpath} has no matching .sha256 sidecar; rerun run.jl with seed 1"
    C = json.loads(cpath.read_text())
    assert (C["condition"], C["test_share"]) == (cond, te_share), "common.json is from another run"
    seed_files = sorted(mdir.glob("fptm-seed*.json"), key=lambda p: int(p.stem.split("seed")[1]))
    unverified = [f.name for f in seed_files if not verified(f)]
    assert not unverified, f"FPTM seed files without a matching .sha256 sidecar: {unverified}"
    SF = [json.loads(f.read_text()) for f in seed_files]
    for s in SF:
        assert s["common_hash"] == C["common_hash"], f"seed {s['seed']} belongs to another common.json"
    have = [s["seed"] for s in SF]
    if not (smoke or timing) and have != list(range(1, N_SEEDS + 1)):
        sys.exit(f"run.py: refusing {corpus} {cond}: a full run needs verified FPTM seed files 1..{N_SEEDS} in {mdir}; "
                 f"found {have or 'none'}. Run run.jl for the missing seeds first.")
    if not SF and (smoke or cond != "B0"):
        sys.exit(f"run.py: refusing {corpus} {cond}: no verified FPTM seed files in {mdir}"
                 + ("" if smoke else f"; under {cond} the booster training rows come from the seed files")
                 + ". Run run.jl first.")
    if not SF:
        print(f"WARNING: no FPTM seed files in {mdir}; timing mode reports the boosters only", flush=True)
    cfg = C["config"]
    d = Path(C["dir"])

    # cheap checks above refuse at once; the memory gate holds only a job that will actually load data
    wait_for_memory(f"python {corpus} {cond}", "python", corpus, cond)
    # --- rebuild the pool and every evaluation set from Julia's positions; assert labels row for row
    t0 = time.time()
    Xp, yp = load_stems(d, C["pool_stems"])
    P = lambda key: np.asarray(C[key], dtype=np.int64) - 1  # noqa: E731
    shared = {"val_te": P("val_te_pos"), "iid": P("iid_pos")}
    sets = {k: (Xp[v], yp[v]) for k, v in shared.items()}
    years = [str(y) for y in C["years"]]
    for y in years:
        Xy, yy = load_stems(d, C["year_stems"][y])
        kp = np.asarray(C["year_pos"][y], dtype=np.int64) - 1
        sets[y] = (Xy[kp], yy[kp])
        del Xy
    for k, (_, lab) in sets.items():
        assert np.array_equal(np.asarray(C["labels"][k], dtype=np.int8), lab), f"label mismatch on {k}"
    assert np.array_equal(np.asarray(C["labels"]["b0_train"], dtype=np.int8), yp[P("b0_train_pos")])
    assert np.array_equal(np.asarray(C["labels"]["b0_val"], dtype=np.int8), yp[P("b0_val_pos")])
    idx11 = np.asarray(C["iid11_idx"], dtype=np.int64) - 1
    y11 = sets["iid"][1][idx11]
    assert np.array_equal(np.asarray(C["labels"]["iid11"], dtype=np.int8), y11)
    assert int(y11.sum()) * 2 == len(y11), "the positive-control set is not exactly 1:1"
    for s in SF:
        assert np.array_equal(np.asarray(s["train_labels"], dtype=np.int8), yp[np.asarray(s["train_pos"]) - 1])
        assert np.array_equal(np.asarray(s["val_labels"], dtype=np.int8), yp[np.asarray(s["val_pos"]) - 1])
    # every set meant to be at 10% is exactly 1:9 (goodware = 9 x malware)
    count_lines = []
    tenpct = [("B0 train", yp[P("b0_train_pos")]), ("B0 val = val_te", yp[P("b0_val_pos")])] + \
             [(k, lab) for k, (_, lab) in sets.items() if k != "val_te"]
    for k, lab in tenpct:
        m = int(lab.sum())
        g = len(lab) - m
        assert g == 9 * m, f"{k} is not 1:9: {m} malware, {g} goodware"
        count_lines.append(f"  {k:16} {m:7d} malware {g:8d} goodware (1:9)")
    for s in SF:
        tl = np.asarray(s["train_labels"])
        vl = np.asarray(s["val_labels"])
        count_lines.append(f"  seed {s['seed']} train  {int(tl.sum()):7d} malware {int(len(tl) - tl.sum()):8d} goodware "
                           f"({s['train_unique']} unique rows); val {int(vl.sum())} malware {int(len(vl) - vl.sum())} goodware")
    count_lines.append(f"  {'iid 1:1 control':16} {int(y11.sum()):7d} malware {int(len(y11) - y11.sum()):8d} goodware (1:1, same rows)")
    print("\n".join(count_lines))
    phase["load"] = time.time() - t0
    periods = ["iid"] + years
    print(f"{corpus} condition={cond} test-share={te_share}{'  SMOKE' if smoke else ''}{'  TIMING' if timing else ''}: "
          f"width {cfg['width']} S {cfg['S']} s {cfg['s']}; {len(SF)} FPTM seed files, hashes match; all label "
          f"vectors match Julia row for row (load {phase['load']:.0f}s)")

    # --- scores: (model, seed) -> {"val": ..., "val_te": ..., period: ...} and its validation labels
    scores, vlabels, trained = {}, {}, {}
    for s in SF:
        scores[("fptm", s["seed"])] = {k: np.asarray(v, dtype=float) for k, v in s["margins"].items()}
        vlabels[("fptm", s["seed"])] = np.asarray(s["val_labels"], dtype=np.int8)
    if cond == "O":       # at exactly 1:9, R = 9 and no extra rows are drawn: every seed has the same rows
        for s in SF[1:]:
            assert s["train_pos"] == SF[0]["train_pos"] and s["val_pos"] == SF[0]["val_pos"], \
                "O rows differ between seeds; the boosters would need one fit per seed"
    if cond in ("B0", "O"):   # identical rows for every seed and deterministic boosters: train once
        src = (C["b0_train_pos"], C["b0_val_pos"]) if cond == "B0" else (SF[0]["train_pos"], SF[0]["val_pos"])
        booster_seeds = [("once", np.asarray(src[0]) - 1, np.asarray(src[1]) - 1)]
    else:                     # U: seed s is draw s
        booster_seeds = [(s["seed"], np.asarray(s["train_pos"]) - 1, np.asarray(s["val_pos"]) - 1)
                         for s in (SF[:1] if timing else SF)]
    determinism = {}
    for name in ("lightgbm", "xgboost"):
        for seed, tr, va in booster_seeds:
            upath = mdir / f"{name}-seed{seed}-t{n_trees}.npz"
            rs = 0 if seed == "once" else int(seed)
            rh, ph = rows_hash(tr, va), params_hash(name, n_trees, rs)
            fresh = False
            if verified(upath):
                with np.load(upath) as Z:
                    fresh = (str(Z["common_hash"]) == C["common_hash"] and "rows_hash" in Z.files
                             and str(Z["rows_hash"]) == rh and str(Z["params_hash"]) == ph)
            if fresh:
                status = "skipped: verified output exists"
            else:
                t0 = time.time()
                clf = train_booster(name, Xp[tr], yp[tr], n_trees, rs)
                t_fit = time.time() - t0
                sc = {k: clf.predict_proba(X)[:, 1] for k, (X, _) in sets.items()}
                sc["val"] = clf.predict_proba(Xp[va])[:, 1]
                t_pred = time.time() - t0 - t_fit
                det = -1
                if smoke or timing:      # two runs on the same rows must give bit-identical probabilities
                    again = train_booster(name, Xp[tr], yp[tr], n_trees, rs)
                    det = int(all(np.array_equal(sc[k], again.predict_proba(X)[:, 1]) for k, (X, _) in sets.items()))
                meta = dict(common_hash=C["common_hash"], rows_hash=rh, params_hash=ph, model=name, seed=str(seed), rows=len(tr),
                            unique=len(np.unique(tr)), fit_s=t_fit, predict_s=t_pred, peak_mb=peak_ram_mb(),
                            trees=n_trees, determinism=det, n_jobs=BOOSTER_THREADS)

                def write_unit(tmp, sc=sc, va=va, meta=meta):
                    with open(tmp, "wb") as fh:
                        np.savez(fh, val_labels=yp[va], **{f"score_{k}": v for k, v in sc.items()},
                                 **{k: np.array(v) for k, v in meta.items()})
                write_verified(upath, write_unit)
                status = "trained"
            assert verified(upath), f"{upath} failed verification after writing"
            with np.load(upath) as Z:          # every number below comes from the verified file
                scores[(name, seed)] = {k[6:]: Z[k] for k in Z.files if k.startswith("score_")}
                vlabels[(name, seed)] = Z["val_labels"]
                trained[(name, seed)] = (int(Z["rows"]), int(Z["unique"]))
                phase[f"{name}_seed{seed}_fit"] = float(Z["fit_s"])
                phase[f"{name}_seed{seed}_predict"] = float(Z["predict_s"])
                if int(Z["determinism"]) >= 0:
                    determinism[name] = determinism.get(name, True) and bool(int(Z["determinism"]))
                print(f"  {name} seed {seed}: {status}; {int(Z['rows']):,} rows ({int(Z['unique']):,} unique), "
                      f"{int(Z['trees'])} trees, fit {float(Z['fit_s']):.0f}s, predict {float(Z['predict_s']):.0f}s, "
                      f"peak RAM {float(Z['peak_mb']):.0f} MB" + (f", bit-identical rerun {bool(int(Z['determinism']))}"
                                                                 if int(Z["determinism"]) >= 0 else ""))
    del Xp

    present = [m for m in MODELS if any(k[0] == m for k in scores)]

    # --- rules
    import lightgbm
    import sklearn
    import xgboost
    conf = (f"width={cfg['width']} S={cfg['S']} s={cfg['s']} tm_lab={cfg['tm_lab']} ceiling={cfg['ceiling']} "
            f"parallel={cfg['parallel']}")
    rows, lines = [], []
    for (model, seed), sc in scores.items():
        cut = 0.0 if model == "fptm" else 0.5
        yva = vlabels[(model, seed)]
        for p in periods:
            y, s = sets[p][1], sc[p]
            ap = 100 * average_precision_score(y, s)
            for rule, (f1, t, ps) in apply_rules(sc["val"], yva, sc["val_te"], sets["val_te"][1], s, y, cut).items():
                rows.append(dict(corpus=corpus, condition=cond, test_share=te_share, model=model, seed=seed, period=p,
                                 n=len(y), n_malware=int(y.sum()), ap=round(ap, 4), rule=rule, threshold=t,
                                 pred_share=round(ps, 6), f1=round(f1, 4)))
                lines.append(f"RESULT corpus={corpus} condition={cond} test_share={te_share} model={model} seed={seed} "
                             f"period={p} rule={rule} f1={f1:.2f} threshold={t:.6g} pred_share={ps:.4f} ap={ap:.2f} "
                             f"n={len(y)} n_malware={int(y.sum())} {conf}")
        # positive control: the same model's scores on the 1:1 cut of the 1:9 iid set (no extra training)
        s11 = sc["iid"][idx11]
        ap11 = 100 * average_precision_score(y11, s11)
        for rule, (f1, t, ps) in apply_rules(sc["val"], yva, sc["val_te"], sets["val_te"][1], s11, y11, cut).items():
            rows.append(dict(corpus=corpus, condition=cond, test_share=te_share, model=model, seed=seed, period=CONTROL,
                             n=len(y11), n_malware=int(y11.sum()), ap=round(ap11, 4), rule=rule, threshold=t,
                             pred_share=round(ps, 6), f1=round(f1, 4)))

    out_dir = HERE / ("timing" if timing else "smoke" if smoke else "")
    out_dir.mkdir(exist_ok=True)
    tag = f"{corpus}-{cond}-te{te_share}"
    def write_csv(tmp):
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write_verified(out_dir / f"per-period-{tag}.csv", write_csv)

    def vals(model, p, rule, key="f1"):
        return np.array([r[key] for r in rows if r["model"] == model and r["period"] == p and r["rule"] == rule])

    fptm_rows = [(s["seed"], len(s["train_pos"]), s["train_unique"], s["train_seconds"], s["maxrss_bytes"] / 2**20) for s in SF]
    summary = [
        f"THRESHOLD RULES -- {corpus}, condition={cond}, test share {te_share}{'  SMOKE' if smoke else ''}{'  TIMING' if timing else ''}",
        f"FPTM clauses={cfg['clauses']} T={cfg['T']} S={cfg['S']} L={cfg['L']} LF={cfg['LF']} epochs={cfg['epochs']}; "
        f"width={cfg['width']} s=round(width/S)={cfg['s']}; {conf}",
        f"pool {C['pool_rows']} rows at {100 * C['pool_share']:.2f}% malware; regions (malware, goodware): training "
        f"{C['train_region_counts']}, validation {C['val_region_counts']}, in-distribution {C['iid_region_counts']}",
        "sets (every 10% set exactly 1:9, asserted):", *count_lines,
        "FPTM seeds (seed, train rows, unique, train s, Julia peak RSS MB): "
        + "; ".join(f"{a}, {b}, {c}, {d:.1f}, {e:.0f}" for a, b, c, d, e in fptm_rows),
        "boosters: " + "; ".join(f"{m} seed {s}: {n} rows ({u} unique)" for (m, s), (n, u) in trained.items())
        + (f" ({cond}: trained once, deterministic, identical rows for every seed)" if cond in ("B0", "O") else
           " (U: one fit per seed, seed s = draw s)"),
        f"LightGBM {json.dumps(booster_params('lightgbm', n_trees, 's'))}",
        f"XGBoost {json.dumps(booster_params('xgboost', n_trees, 's'))} (depthwise, max_depth 6, at most 64 leaves)",
        f"versions: lightgbm {lightgbm.__version__}, xgboost {xgboost.__version__}, scikit-learn {sklearn.__version__}, "
        f"numpy {np.__version__}; threads: booster n_jobs={BOOSTER_THREADS} (-1 = all) on {os.cpu_count()} logical cores",
        "determinism (two runs on the same rows, bit-identical probabilities): "
        + (", ".join(f"{k} {v}" for k, v in determinism.items()) if determinism else "not checked in a full run"),
        "F1 points; mean over seeds, range across seeds in brackets. Later years: " + ", ".join(years) + ".", ""]
    gaps = (("oracle-validation", "oracle", "validation"), ("oracle-validation_te", "oracle", "validation_te"),
            ("oracle-rate", "oracle", "rate"), ("oracle-rate_te", "oracle", "rate_te"),
            ("default-validation", "default", "validation"), ("default-validation_te", "default", "validation_te"))
    for gname, a, b in gaps:
        summary.append(gname)
        summary.append(f"  {'model':9}" + "".join(f"{p:>16}" for p in periods) + f"   {'mean later yrs':>14}")
        for model in present:
            cells, later = [], []
            for p in periods:
                g = vals(model, p, a) - vals(model, p, b)
                cells.append(f"{g.mean():8.2f} ({np.ptp(g):5.2f})")
                if p != "iid":
                    later.append(g.mean())
                lines.append(f"RESULT gap corpus={corpus} condition={cond} model={model} period={p} gap={gname} "
                             f"mean={g.mean():.2f} range={np.ptp(g):.2f} seeds={len(g)}")
            summary.append(f"  {model:9}" + "".join(f"{c:>16}" for c in cells) + f"   {np.mean(later):14.2f}")
        summary.append("")
    summary.append("positive control (planted prior shift, no covariate drift): oracle gap on the iid set cut to 1:1 "
                   "against the 1:9 iid set; expected larger by more than the seed/draw range of either side "
                   "under validation_te (the criterion); validation reported beside it")
    for rule in ("validation_te", "validation"):
        for model in present:
            g19 = vals(model, "iid", "oracle") - vals(model, "iid", rule)
            g11 = vals(model, CONTROL, "oracle") - vals(model, CONTROL, rule)
            d = float(g11.mean() - g19.mean())
            ok = d > 0 and d > np.ptp(g11) and d > np.ptp(g19)
            verdict = ("PASS" if ok else "FAIL") if rule == "validation_te" else ("larger" if ok else "not larger")
            summary.append(f"  {rule:14} {model:9} 1:9 {g19.mean():6.2f} ({np.ptp(g19):4.2f})  1:1 {g11.mean():6.2f} "
                           f"({np.ptp(g11):4.2f})  diff {d:+6.2f} -> {verdict}")
            lines.append(f"RESULT control_positive corpus={corpus} condition={cond} rule={rule} model={model} "
                         f"gap_1to9={g19.mean():.2f} range_1to9={np.ptp(g19):.2f} gap_1to1={g11.mean():.2f} "
                         f"range_1to1={np.ptp(g11):.2f} diff={d:.2f} verdict={verdict}")
    summary.append("")
    summary.append("absolute F1 by rule, and AP (threshold-free), mean over seeds; columns " + ", ".join(periods))
    for model in present:
        for rule in RULES:
            summary.append(f"  {model:9} {rule:14}" + "".join(f"{vals(model, p, rule).mean():9.2f}" for p in periods))
        summary.append(f"  {model:9} {'AP':14}" + "".join(f"{vals(model, p, 'oracle', 'ap').mean():9.2f}" for p in periods))
    summary.append("")
    phase["total"] = time.time() - t_start
    summary.append("timing (s): " + ", ".join(f"{k} {v:.0f}" for k, v in phase.items())
                   + f"; FPTM train per seed {', '.join(f'{x[3]:.1f}' for x in fptm_rows)}; Python peak RAM {peak_ram_mb():.0f} MB")
    lines.append(f"RESULT timing corpus={corpus} condition={cond} trees={n_trees} "
                 + " ".join(f"{k}={v:.1f}" for k, v in phase.items()) + f" python_peak_mb={peak_ram_mb():.0f} "
                 + "julia_peak_mb=" + "/".join(f"{x[4]:.0f}" for x in fptm_rows))
    text = "\n".join(summary + [""] + lines) + "\n"
    write_verified(out_dir / f"results-{tag}.txt", lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))
    print("\n".join(summary))
    if not (smoke or timing):
        rebuild_results()


def condition_scores(corpus, cond, te_share, smoke, n_trees):
    """Seed-mean scores per model on the shared rows, with fixed thresholds, from verified unit files only.

    FPTM: the mean margin over its seeds. Boosters: the one fit under B0 and O, the mean over the draws under U.
    validation_te is chosen on the shared 1:9 validation set; validation on the condition's own validation set
    (B0: the same set; O: the set with malware repeated, identical for every seed; U: the draw of seed 1).
    """
    mdir = Path("data/threshold-corpora") / run_tag(corpus, cond, te_share, smoke)
    cpath = mdir / "common.json"
    assert verified(cpath), f"{cpath} has no matching .sha256 sidecar"
    C = json.loads(cpath.read_text())
    files = sorted(mdir.glob("fptm-seed*.json"), key=lambda p: int(p.stem.split("seed")[1]))
    assert files and all(verified(f) for f in files), f"unverified or missing FPTM seed files in {mdir}"
    SF = [json.loads(f.read_text()) for f in files]
    assert all(s["common_hash"] == C["common_hash"] for s in SF)
    years = [str(y) for y in C["years"]]
    keys = ["val_te", "iid"] + years
    labels = {k: np.asarray(C["labels"][k], dtype=float) for k in keys}
    where = {p: i for i, p in enumerate(C["val_te_pos"])}
    own_pos = C["b0_val_pos"] if cond == "B0" else SF[0]["val_pos"]
    own_idx = np.array([where[p] for p in own_pos])
    models = {"fptm": {k: np.mean([np.asarray(s["margins"][k], dtype=float) for s in SF], axis=0) for k in keys}}
    for name in ("lightgbm", "xgboost"):
        if cond in ("B0", "O"):
            src = (C["b0_train_pos"], C["b0_val_pos"]) if cond == "B0" else (SF[0]["train_pos"], SF[0]["val_pos"])
            expect = {"once": rows_hash(np.asarray(src[0]) - 1, np.asarray(src[1]) - 1)}
        else:
            expect = {str(s["seed"]): rows_hash(np.asarray(s["train_pos"]) - 1, np.asarray(s["val_pos"]) - 1) for s in SF}
        sc = []
        for seed, rh in expect.items():
            u = mdir / f"{name}-seed{seed}-t{n_trees}.npz"
            assert verified(u), f"{u} missing or unverified"
            with np.load(u) as Z:
                assert str(Z["common_hash"]) == C["common_hash"] and str(Z["rows_hash"]) == rh, f"{u} is stale"
                sc.append({k: np.asarray(Z[f"score_{k}"], dtype=float) for k in keys})
        models[name] = {k: np.mean([d[k] for d in sc], axis=0) for k in keys}
    out = {}
    for m, s in models.items():
        out[m] = dict(scores=s, cut=0.0 if m == "fptm" else 0.5,
                      own=(s["val_te"][own_idx], labels["val_te"][own_idx]),
                      t={"validation": pick_threshold(labels["val_te"][own_idx], s["val_te"][own_idx]),
                         "validation_te": pick_threshold(labels["val_te"], s["val_te"])})
    return labels, out, years, np.asarray(C["iid11_idx"], dtype=np.int64) - 1


def _sorted_groups(s):
    order = np.argsort(-s, kind="mergesort")
    ss = s[order]
    ends = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    return order, ends


def _weighted_stats(w, y, order, ends, preds):
    """Oracle F1, AP and F1 at fixed predictions, with row weights w (resample counts)."""
    wo, yo = w[order], y[order]
    ctp = np.cumsum(wo * yo)[ends]
    cfp = np.cumsum(wo * (1.0 - yo))[ends]
    pos = ctp[-1]
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    rec = ctp / max(pos, 1e-12)
    res = {"oracle": 100 * float(np.max(2 * prec * rec / np.maximum(prec + rec, 1e-12))),
           "ap": 100 * float(np.sum(np.diff(np.r_[0.0, rec]) * prec))}
    wy, wn = w * y, w * (1.0 - y)
    for r, pr in preds.items():
        tp, fp = float(wy @ pr), float(wn @ pr)
        res[r] = 100 * 2 * tp / max(2 * tp + fp + (pos - tp), 1e-12)
    return res


def bootstrap_stats(corpus, te_share, smoke, n_trees):
    """Paired bootstrap over the shared test rows: index 0 is the point estimate, 1..N_BOOT the resamples.

    Every model and condition sees the same resample of a period's rows; the 1:1 control uses the iid weights
    restricted to its rows. Periods are resampled independently.
    """
    data = {c: condition_scores(corpus, c, te_share, smoke, n_trees) for c in CONDITIONS}
    labels, _, years, idx11 = data["B0"]
    for c in CONDITIONS:
        for k in labels:
            assert np.array_equal(labels[k], data[c][0][k]), f"{c} does not share the {k} rows with B0"
    periods = ["iid"] + years
    setup = {}
    for p in periods + [CONTROL]:
        y = labels["iid"][idx11] if p == CONTROL else labels[p]
        for c in CONDITIONS:
            for m, d in data[c][1].items():
                s = d["scores"]["iid"][idx11] if p == CONTROL else d["scores"][p]
                preds = {"validation": (s >= d["t"]["validation"]).astype(float),
                         "validation_te": (s >= d["t"]["validation_te"]).astype(float),
                         "default": (s > d["cut"]).astype(float)}
                setup[(p, c, m)] = (y, *_sorted_groups(s), preds)
    stats = {key: {q: np.empty(N_BOOT + 1) for q in ("oracle", "ap", "validation", "validation_te", "default")}
             for key in setup}
    rng = np.random.default_rng(BOOT_SEED)
    for b in range(N_BOOT + 1):
        for p in periods:
            n = len(labels[p])
            w = np.ones(n) if b == 0 else np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            for c in CONDITIONS:
                for m in MODELS:
                    for p2, ww in ((p, w), (CONTROL, w[idx11])) if p == "iid" else ((p, w),):
                        y, order, ends, preds = setup[(p2, c, m)]
                        for q, v in _weighted_stats(ww, y, order, ends, preds).items():
                            stats[(p2, c, m)][q][b] = v
    return stats, years, data


def em_prior(p, pi_train, tol=1e-6, max_iter=1000):
    """Saerens et al. 2002: EM estimate of the test prior; returns (pi, p', iterations, prior ratios)."""
    pi = pi_train
    for it in range(1, max_iter + 1):
        r1, r0 = pi / pi_train, (1 - pi) / (1 - pi_train)
        pp = r1 * p / (r1 * p + r0 * (1 - p))
        new = float(np.mean(pp))
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    r1, r0 = pi / pi_train, (1 - pi) / (1 - pi_train)
    return pi, r1 * p / (r1 * p + r0 * (1 - p)), it, (r1, r0)


def exploratory_em(corpus, data, years):
    """EXPLORATORY, no pass/fail: EM prior correction per condition, model and test period."""
    from sklearn.linear_model import LogisticRegression
    out = ["EXPLORATORY (no pass/fail): EM prior correction (Saerens, Latinne, Decaestecker, Neural Computation "
           "14(1), 2002). pi = estimated test prior; F1 of the corrected probabilities p' at 0.5 and at t_val, the "
           "validation F1-optimal threshold chosen in p-space at pi_train, applied unchanged to p'."]
    for cond in CONDITIONS:
        labels, models, _, idx11 = data[cond]
        pi_train = 0.1 if cond == "B0" else 0.5
        for m in MODELS:
            d = models[m]
            s_own, y_own = d["own"]
            if m == "fptm":     # Platt-scale the seed-mean margin on the condition's own validation set
                lr = LogisticRegression(C=1e6, solver="lbfgs").fit(s_own.reshape(-1, 1), y_own.astype(int))
                prob = lambda s: lr.predict_proba(np.asarray(s, dtype=float).reshape(-1, 1))[:, 1]  # noqa: E731
            else:
                prob = lambda s: np.asarray(s, dtype=float)  # noqa: E731
            t_val = pick_threshold(y_own, prob(s_own))
            for p in ["iid"] + years + [CONTROL]:
                y = labels["iid"][idx11] if p == CONTROL else labels[p]
                s = d["scores"]["iid"][idx11] if p == CONTROL else d["scores"][p]
                pi, pp, it, _ = em_prior(prob(s), pi_train)
                f05 = 100 * f1_score(y.astype(int), (pp >= 0.5).astype(int), zero_division=0)
                ftv = 100 * f1_score(y.astype(int), (pp >= t_val).astype(int), zero_division=0)
                out.append(f"  {cond:3} {m:9} {p:9} true share {y.mean():.3f}  pi_hat {pi:.3f} ({it:4d} it)  "
                           f"F1@0.5 {f05:6.2f}  F1@t_val {ftv:6.2f}  (t_val {t_val:.4f})")
                out.append(f"RESULT exploratory_em corpus={corpus} condition={cond} model={m} period={p} "
                           f"pi_train={pi_train} true_share={y.mean():.4f} pi_hat={pi:.4f} iterations={it} "
                           f"f1_at_0.5={f05:.2f} f1_at_t_val={ftv:.2f} t_val={t_val:.6g}")
    return out


def evaluate(corpus, te_share, smoke, n_trees):
    """Readings (1), (1b), (1c), (2) and both controls, under the stated criterion and the paired bootstrap."""
    base = HERE / ("smoke" if smoke else "")
    R = {}
    for cond in CONDITIONS:
        f = base / f"per-period-{corpus}-{cond}-te{te_share}.csv"
        assert verified(f), f"{f} has no matching .sha256 sidecar"
        with open(f, encoding="utf-8") as fh:
            R[cond] = list(csv.DictReader(fh))
    periods = [p for p in dict.fromkeys(r["period"] for r in R["B0"]) if p != CONTROL]
    later = [p for p in periods if p != "iid"]
    B, years, data = bootstrap_stats(corpus, te_share, smoke, n_trees)
    assert years == later

    def per_seed(cond, model, p, rule, key="f1"):
        return {r["seed"]: float(r[key]) for r in R[cond]
                if r["model"] == model and r["period"] == p and r["rule"] == rule}

    def gap(cond, model, p, a, b):
        x, y = per_seed(cond, model, p, a), per_seed(cond, model, p, b)
        return np.array([x[k] - y[k] for k in x])

    def counts(a, b):
        """Stated: |mean difference| > TOL and > the seed/draw range of each side."""
        d = float(np.mean(a) - np.mean(b))
        return abs(d) > TOL and abs(d) > np.ptp(a) and abs(d) > np.ptp(b), d

    def bgap(p, cond, model, rule):
        return B[(p, cond, model)]["oracle"] - B[(p, cond, model)][rule]

    def bcounts(diff):
        """Bootstrap: |point difference| > TOL and the 95% interval excludes zero."""
        lo, hi = np.percentile(diff[1:], [2.5, 97.5])
        return abs(diff[0]) > TOL and (lo > 0 or hi < 0), float(diff[0]), float(lo), float(hi)

    def bq(p, cond, model, q):
        if q == "oracle F1":
            return B[(p, cond, model)]["oracle"]
        if q == "AP":
            return B[(p, cond, model)]["ap"]
        a, r = q.split("-", 1)
        return B[(p, cond, model)][a] - B[(p, cond, model)][r]

    out = [f"PASS/FAIL -- {corpus}, test share {te_share}{'  SMOKE' if smoke else ''}. Stated criterion: |difference| > "
           f"{TOL} F1 and > the seed/draw range of each side. Bootstrap criterion: |difference| > {TOL} F1 and its 95% "
           f"interval ({N_BOOT} paired resamples of the shared test rows, seed-mean scores) excludes zero. The "
           f"bootstrap criterion decides every verdict (decided 2026-11-13 before any full result was read); the "
           f"stated criterion is secondary. 'Small' in-distribution gap <= {SMALL}.", ""]

    out.append("Reading (1): oracle gap small in distribution, larger on later years")
    for cond in CONDITIONS:
        for rule in ("validation", "validation_te"):
            for model in MODELS:
                g_iid = gap(cond, model, "iid", "oracle", rule)
                g_later = np.mean([gap(cond, model, p, "oracle", rule) for p in later], axis=0)
                c, d = counts(g_later, g_iid)
                v_s = "UNRESOLVED" if g_iid.mean() > SMALL else "SURVIVES" if c and d > 0 else "NOT SUPPORTED"
                bi = bgap("iid", cond, model, rule)
                bl = np.mean([bgap(p, cond, model, rule) for p in later], axis=0)
                cb, db, lo, hi = bcounts(bl - bi)
                v_b = "UNRESOLVED" if bi[0] > SMALL else "SURVIVES" if cb and db > 0 else "NOT SUPPORTED"
                out.append(f"  {cond:3} {rule:14} {model:9} stated: iid {g_iid.mean():6.2f} later {g_later.mean():6.2f} "
                           f"diff {d:+6.2f} -> {v_s:13} (secondary) | bootstrap (deciding): iid {bi[0]:6.2f} later {bl[0]:6.2f} diff {db:+6.2f} "
                           f"[{lo:+.2f}, {hi:+.2f}] -> {v_b}")
                out.append(f"RESULT reading1 corpus={corpus} condition={cond} rule={rule} model={model} "
                           f"stated_iid_gap={g_iid.mean():.2f} stated_later_gap={g_later.mean():.2f} stated_diff={d:.2f} "
                           f"stated_verdict={v_s.replace(' ', '_')} boot_iid_gap={bi[0]:.2f} boot_later_gap={bl[0]:.2f} "
                           f"boot_diff={db:.2f} boot_lo={lo:.2f} boot_hi={hi:.2f} boot_verdict={v_b.replace(' ', '_')} verdict={v_b.replace(' ', '_')}")
    out.append("")

    if corpus == "androzoo" and {"2020", "2021"} <= set(later):
        out.append("Reading (1b): AndroZoo, oracle gap in 2021 larger than in 2020 (all three models)")
        for cond in CONDITIONS:
            for rule in ("validation", "validation_te"):
                ok_s = ok_b = True
                for model in MODELS:
                    c, d = counts(gap(cond, model, "2021", "oracle", rule), gap(cond, model, "2020", "oracle", rule))
                    cb, db, lo, hi = bcounts(bgap("2021", cond, model, rule) - bgap("2020", cond, model, rule))
                    ok_s &= c and d > 0
                    ok_b &= cb and db > 0
                    out.append(f"  {cond:3} {rule:14} {model:9} stated diff {d:+6.2f} counts {c} | bootstrap diff "
                               f"{db:+6.2f} [{lo:+.2f}, {hi:+.2f}] counts {cb}")
                v_s, v_b = ("SURVIVES" if ok_s else "NOT SUPPORTED"), ("SURVIVES" if ok_b else "NOT SUPPORTED")
                out.append(f"  {cond:3} {rule:14} -> {v_b} (bootstrap, deciding) | stated (secondary) {v_s}")
                out.append(f"RESULT reading1b corpus={corpus} condition={cond} rule={rule} "
                           f"stated_verdict={v_s.replace(' ', '_')} boot_verdict={v_b.replace(' ', '_')} verdict={v_b.replace(' ', '_')}")
        out.append("")

    out.append("Reading (1c): each booster's mean later-year oracle gap larger than FPTM's (both boosters)")
    for cond in CONDITIONS:
        for rule in ("validation", "validation_te"):
            f_s = np.mean([gap(cond, "fptm", p, "oracle", rule) for p in later], axis=0)
            f_b = np.mean([bgap(p, cond, "fptm", rule) for p in later], axis=0)
            ok_s = ok_b = True
            for model in ("lightgbm", "xgboost"):
                g_s = np.mean([gap(cond, model, p, "oracle", rule) for p in later], axis=0)
                c, d = counts(g_s, f_s)
                cb, db, lo, hi = bcounts(np.mean([bgap(p, cond, model, rule) for p in later], axis=0) - f_b)
                ok_s &= c and d > 0
                ok_b &= cb and db > 0
                out.append(f"  {cond:3} {rule:14} {model:9} minus fptm: stated {d:+6.2f} counts {c} | bootstrap "
                           f"{db:+6.2f} [{lo:+.2f}, {hi:+.2f}] counts {cb}")
            v_s, v_b = ("SURVIVES" if ok_s else "NOT SUPPORTED"), ("SURVIVES" if ok_b else "NOT SUPPORTED")
            out.append(f"  {cond:3} {rule:14} -> {v_b} (bootstrap, deciding) | stated (secondary) {v_s}")
            out.append(f"RESULT reading1c corpus={corpus} condition={cond} rule={rule} "
                       f"stated_verdict={v_s.replace(' ', '_')} boot_verdict={v_b.replace(' ', '_')} verdict={v_b.replace(' ', '_')}")
    out.append("")

    out.append("Reading (2): the condition moves default minus validation, not oracle F1, AP or oracle minus validation")
    for rule in ("validation", "validation_te"):
        for model in MODELS:
            qs = ("oracle F1", "AP", f"oracle-{rule}", f"default-{rule}")
            moved_s, moved_b = {q: [] for q in qs}, {q: [] for q in qs}
            for p in periods:
                for c1, c2 in (("B0", "U"), ("B0", "O"), ("U", "O")):
                    stated = {
                        "oracle F1": (np.array(list(per_seed(c1, model, p, "oracle").values())),
                                      np.array(list(per_seed(c2, model, p, "oracle").values()))),
                        "AP": (np.array(list(per_seed(c1, model, p, "oracle", "ap").values())),
                               np.array(list(per_seed(c2, model, p, "oracle", "ap").values()))),
                        f"oracle-{rule}": (gap(c1, model, p, "oracle", rule), gap(c2, model, p, "oracle", rule)),
                        f"default-{rule}": (gap(c1, model, p, "default", rule), gap(c2, model, p, "default", rule)),
                    }
                    for q, (a, b) in stated.items():
                        c, d = counts(a, b)
                        if c:
                            moved_s[q].append(f"{p} {c1}-{c2} {d:+.2f}")
                        cb, db, lo, hi = bcounts(bq(p, c1, model, q) - bq(p, c2, model, q))
                        if cb:
                            moved_b[q].append(f"{p} {c1}-{c2} {db:+.2f}")
            verdicts = []
            for moved in (moved_s, moved_b):
                ok = not moved["oracle F1"] and not moved["AP"] and not moved[f"oracle-{rule}"] and moved[f"default-{rule}"]
                verdicts.append("SURVIVES" if ok else "NOT SUPPORTED")
            out.append(f"  {rule:14} {model:9} -> {verdicts[1]} (bootstrap, deciding) | stated (secondary) {verdicts[0]}")
            for q in qs:
                out.append(f"      {q:22} counted: stated {len(moved_s[q])}, bootstrap {len(moved_b[q])}: "
                           + "; ".join(moved_b[q][:6]) + (" ..." if len(moved_b[q]) > 6 else ""))
            out.append(f"RESULT reading2 corpus={corpus} rule={rule} model={model} "
                       f"stated_verdict={verdicts[0].replace(' ', '_')} boot_verdict={verdicts[1].replace(' ', '_')} verdict={verdicts[1].replace(' ', '_')} "
                       + " ".join(f"{q.replace(' ', '_')}_counted_stated={len(moved_s[q])} "
                                  f"{q.replace(' ', '_')}_counted_boot={len(moved_b[q])}" for q in qs))
    out.append("")

    out.append(f"Negative control: oracle gap on the 1:9 iid set at most {SMALL} (stated: mean; bootstrap: upper 95% bound)")
    for cond in CONDITIONS:
        for rule in ("validation_te", "validation"):
            for model in MODELS:
                g19 = gap(cond, model, "iid", "oracle", rule)
                b19 = bgap("iid", cond, model, rule)
                hi = float(np.percentile(b19[1:], 97.5))
                v_s = "PASS" if g19.mean() <= SMALL else "FAIL"
                v_b = "PASS" if hi <= SMALL else "FAIL"
                out.append(f"  {cond:3} {rule:14} {model:9} stated {g19.mean():6.2f} -> {v_s} (secondary) | bootstrap (deciding) {b19[0]:6.2f} "
                           f"(upper {hi:6.2f}) -> {v_b}")
                out.append(f"RESULT control_negative corpus={corpus} condition={cond} rule={rule} model={model} "
                           f"stated_gap={g19.mean():.2f} stated_verdict={v_s} boot_gap={b19[0]:.2f} boot_hi={hi:.2f} "
                           f"boot_verdict={v_b} verdict={v_b}")
    out.append("")

    out.append("Positive control: oracle gap on the 1:1 cut of the iid set larger than on the 1:9 iid set by a "
               "difference that counts (criterion rule: validation_te; validation reported beside it)")
    n_s = n_b = n_all = 0
    for cond in CONDITIONS:
        for rule in ("validation_te", "validation"):
            for model in MODELS:
                g19, g11 = gap(cond, model, "iid", "oracle", rule), gap(cond, model, CONTROL, "oracle", rule)
                c, d = counts(g11, g19)
                cb, db, lo, hi = bcounts(bgap(CONTROL, cond, model, rule) - bgap("iid", cond, model, rule))
                ok_s, ok_b = c and d > 0, cb and db > 0
                if rule == "validation_te":
                    n_all += 1
                    n_s += ok_s
                    n_b += ok_b
                lab = (("PASS", "FAIL") if rule == "validation_te" else ("larger", "not larger"))
                v_s, v_b = lab[0] if ok_s else lab[1], lab[0] if ok_b else lab[1]
                out.append(f"  {cond:3} {rule:14} {model:9} stated 1:9 {g19.mean():6.2f} 1:1 {g11.mean():6.2f} diff "
                           f"{d:+6.2f} -> {v_s:10} (secondary) | bootstrap (deciding) diff {db:+6.2f} [{lo:+.2f}, {hi:+.2f}] -> {v_b}")
                out.append(f"RESULT control_positive corpus={corpus} condition={cond} rule={rule} model={model} "
                           f"stated_gap_1to9={g19.mean():.2f} stated_gap_1to1={g11.mean():.2f} stated_diff={d:.2f} "
                           f"stated_verdict={v_s.replace(' ', '_')} boot_diff={db:.2f} boot_lo={lo:.2f} boot_hi={hi:.2f} "
                           f"boot_verdict={v_b.replace(' ', '_')} verdict={v_b.replace(' ', '_')}")
    out.append(f"  positive control passed (validation_te): {n_b} of {n_all} by the bootstrap criterion (deciding); "
               f"stated (secondary) {n_s} of {n_all}")
    out.append(f"RESULT control_positive_summary corpus={corpus} passed={n_b} boot_passed={n_b} stated_passed={n_s} "
               f"of={n_all}")
    out.append("")
    out += exploratory_em(corpus, data, years)
    text = "\n".join(out) + "\n"
    write_verified(base / f"evaluation-{corpus}-te{te_share}.txt", lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))
    print(text)
    if not smoke:
        rebuild_results()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    smoke, timing = "--smoke" in sys.argv, "--timing" in sys.argv
    corpus, te_share = args[0], flag("test-share", "0.10")
    if "--evaluate" in sys.argv:
        evaluate(corpus, te_share, smoke, int(flag("trees", "200" if smoke else "5000")))
        return 0
    cond = flag("condition", "")
    assert cond in CONDITIONS, "--condition must be B0, U or O"
    n_trees = int(flag("trees", "200" if smoke else "5000"))
    run_condition(corpus, cond, te_share, smoke, timing, n_trees)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
