# B0 — pipeline calibration

**Purpose: calibration, not a finding.** Nothing in B0 is reportable as a result. Its job is to
establish that our numbers can be compared to published ones at all, and to fail loudly and early
if they cannot. Budget: one working day, most of it download and waiting.

**Prerequisite decisions** are listed at the end; B0.1 and B0.2 cannot start without them.

## Storage

Datasets and derived matrices go on **E:** (1.5 TB free; D: has 79 GB and holds the repo).

```
E:\TM-Cyber-data\
  lamda\raw\           HuggingFace download, baseline variant
  lamda\tmx\           derived .tmx matrices and Arrow sidecars
  apigraph\            B5, later
  vt_detections.csv    label-drift source, 27 MB, fetched separately
```

The repo's gitignored `data\` becomes a junction to that directory, so paths in scripts stay
relative and nothing large ever sits on D:.

Rough sizes: LAMDA baseline variant 222 MB compressed; 1,008,381 × 4,561 bits packed is **574 MB**
as `.tmx`, so the whole dataset fits in memory comfortably and the per-month slices are trivial.
Allow 10 GB for the raw download, intermediates and per-year derivatives.

---

## B0.1 — Dataset acquisition and composition check

Fetch the baseline (4,561-feature) variant from `IQSeC-Lab/LAMDA` via `huggingface_hub`, plus
`vt_detections.csv` from their GitHub — that one is standalone and small, and is the raw material
for B3, so it is worth having before we need it.

**Then reproduce the composition table**, per year and per month: total APKs, malware, benign,
family counts. Expected: 1,008,381 total, 369,906 malware, 638,475 benign, 2013–2025 excluding
2015, 1,380 families, 150,604 singletons.

**Pass:** counts match the paper. **Fail:** a mismatch means we have a different release than the
one the published baselines were computed on, and everything downstream is uncalibrated — stop and
resolve before continuing.

Also record, because later steps depend on them and nobody should re-derive them: per-month row
counts (several months are thin, and the SHAP protocol takes 100 rows per month, so months with
fewer than ~200 usable rows need an explicit policy), the exact feature ordering, and the feature
names.

## B0.2 — Baseline reproduction

LightGBM on the baseline variant, AnoShift-style splits, same hyperparameters the paper reports
(up to 5000 estimators, learning rate 0.02, max 256 leaves).

**Target:** F1 of **97.49 IID / 59.48 NEAR / 47.24 FAR**. Report FAR per year as well as the single
figure, the latter only to demonstrate that we reproduce their number, alongside the per-year
breakdown that shows why it should not be used again.

**Pass:** within ~1 F1 point on IID and NEAR. **Fail:** our splits or preprocessing differ from
theirs, and no comparison in B1 would mean anything. This is the single most important step in B0
and the reason it exists.

XGBoost is a cheap second point on the same pipeline; run it if LightGBM matches, skip if not.

## B0.3 — Matrix boundary at full scale

The boundary is already verified bit-exact at 1, 64, 65 and 4,561 columns on small fixtures. What is
untested is scale: write the whole dataset to `.tmx` with Arrow sidecars, read it back in Julia,
and check that row counts, per-row popcounts and the label column agree end to end. Also time a
per-month slice read, because B2 does ~135 of them.

**Pass:** bit-exact, and a month slice loads in well under a second.

## B0.4 — First TM sanity run, and the readability diagnostic

Not the gate — a single serial run at the gate configuration (20 clauses per class, `T` ≈ 10,
`LF` = 10, `L` = 64, `:none`, `LiteralCapped`) on the 2013–14 training split, scored on IID.

Its real purpose is the two measurements that decide what can be claimed later, and both are
cheap once a model exists:

- **`LF` / included-literals.** The readability diagnostic. The prediction on record is that LAMDA
  lands near IMDb's 1.7% rather than MNIST's 21%, in which case the contribution is the *exactness*
  of the drift measurement and not "interpretable detector".
- **Sign composition.** Fraction of core literals that are positive versus negated. The prediction
  on record is that the clauses are near-pure blacklists, which would mean they give an analyst no
  indicators of compromise.

Both predictions are written down in advance and both may be wrong. Record the number either way.

## B0.5 — The noise-floor probe *(promoted from B2)*

This one is out of order on purpose. It is the decisive experiment for the whole contribution, it
needs no TM at all, and it costs an afternoon.

Re-run LAMDA's own SHAP pipeline unchanged — their MLP, their `KernelExplainer`, `nsamples=100`,
100 background and 100 explained rows, three runs per month — and compute the one quantity they
never did: **Jaccard between runs of the same month**.

- If the within-month between-run Jaccard is **near 0.9**, their reported explanation drift is
  substantially estimator variance, and the central B2 result is established before we train a
  single Tsetlin machine.
- If it is **near 0**, the churn is real, B2 becomes the exact-decomposition result instead
  (arms 2 and 3), and we will have found that out for the price of one afternoon rather than after
  building everything.

Either outcome reshapes B2, which is why it comes before B1 rather than after. It is also the only
part of B0 that could produce something publishable, so if it does, that changes the paper's
structure and should be discussed rather than absorbed.

---

## What B0 deliberately does not do

- No hyperparameter search. B1's gate is at a stated configuration; tuning before the gate turns a
  gate into a target.
- No feature-variant comparison. 925 vs 4,561 vs 25,460 differ in width, which needs the `s`
  rescaling and dead-channel controls; that is its own experiment.
- No re-running of LAMDA's Appendix E. The VT-threshold noise confound is already settled there;
  cite it.
- No parallel mode other than `:none`, and no threading claims. Cores go to arms.

## Ordering and cost

| | step | cost | blocks |
|---|---|---|---|
| 1 | Python env, storage junction | minutes | everything |
| 2 | B0.1 acquisition + composition | ~1 h, mostly download | B0.2–B0.4 |
| 3 | B0.5 noise floor | afternoon | nothing — run in parallel with 2 |
| 4 | B0.2 LightGBM reproduction | ~1 h | B1 |
| 5 | B0.3 boundary at scale | ~30 min | B1 |
| 6 | B0.4 sanity run + diagnostics | ~30 min | B1 |

Everything here fits on one machine. The rented CPU box is not needed until B1's seed and
hyperparameter arms, and B2's per-month series — and B2's exact attribution is 15 minutes
single-threaded for the full series, so it may never be needed at all.

## Decisions required before starting

1. **Python environment** — create `.venv` in the repo and `pip install -r requirements.txt`?
   Nothing is installed yet; lightgbm, xgboost, shap, wittgenstein and huggingface_hub are all
   missing. Recommendation: yes.
2. **LAMDA variant** — baseline (4,561) only for now. Recommendation: yes; the other two variants
   are a later width comparison.
3. **B0.5 promoted ahead of B1** — it is a departure from the agreed order, justified by it being
   decisive, cheap, and independent of everything else. Recommendation: yes, run it in parallel
   with the download.
4. **Storage layout on E:** as above, with `data\` as a junction. Recommendation: yes.
