# b0-noise-floor — the reported explanation drift against its estimator's noise

LAMDA reports Jaccard distance "close to 0.9" between consecutive months over the top-100 SHAP
features and reads it as **explanation drift**: the features the model relies on almost entirely turn
over month to month, while classification performance is comparatively stable.

A Jaccard of 0.9 is equally consistent with real drift and with a noisy estimator. This measures
which.

## Answer: the estimator

Nine months of the benchmark, three independent runs per month, top-100 features.

| arm | Jaccard | sd | n |
|---|---|---|---|
| **(a)** same month, two independent runs, model refit each time | **0.926** | 0.024 | 27 |
| **(b)** same month, **same model**, only the explainer re-run | **0.926** | 0.021 | 18 |
| **(c)** consecutive months, within a run — *their protocol, re-run here* | 0.958 | 0.012 | 24 |

Kendall distance: 0.723 for (a) against 0.753 for (c).

**The noise floor is 97% of the reported signal.** The gap between them, 0.032, is about 1.3 standard
deviations of the noise distribution. Running the published protocol twice on *identical data*
reproduces almost exactly the churn that is reported as drift between *different months*.

**And (a) equals (b) to three decimal places.** Retraining the model every month — which the released
script does, and which the paper does not mention — contributes nothing measurable. With the model,
the data and the explained rows all held fixed, re-running only the explainer's coalition sampling
turns over 93% of the top-100 set. The churn is the sampling.

That arm (b) exists is why this is interpretable. Arms (a) and (c) alone would have shown the floor
was high without saying whether the cause was refitting or sampling, and those have different
implications: refitting noise would be a property of the protocol, sampling noise is a property of the
estimator at the budget used.

## The estimator's mechanism

The released script calls `shap.KernelExplainer(...).shap_values(X_test[:100], nsamples=100)` on
4,561 features. `KernelExplainer`'s own default is `2·M + 2048`, which here is **11,170** coalitions.
The published measurement uses **100** — fewer coalitions than there are features, by a factor of 45.

SHAP itself says so. Every call in this experiment emitted the library's warning that the number of
samples is too small to determine a regular solution, advising to "turn up the number of samples" or
reduce the number of inputs. Estimating 4,561 coefficients from 100 observations is underdetermined by
construction, and the top-100 ranking taken from it is substantially arbitrary.

The published analysis reports mean ± standard error over its three runs. That puts an error bar on
the *drift curve* but never compares the runs to each other, which is the one comparison that would
have revealed the floor. The data to do it already existed.

## Protocol

Transcribed from the authors' released script
(`code/section_4_concept_drift_analysis/4_5_shap_explanation_monthly_lamda.py`):

| | |
|---|---|
| model | `ChenEncoderMLP`: 4561→512→384→256→128 encoder, 128→100→100→2 head, dropout 0.2 |
| training | 20 epochs, batch 64, Adam lr 1e-3, `CrossEntropyLoss`, fresh model per month per run |
| explainer | `shap.KernelExplainer` over softmax probabilities |
| background | first 100 rows of that month's training split |
| explained | first 100 rows of that month's test split |
| coalitions | `nsamples=100` |
| importance | mean over explained rows of `abs(shap[:, 1])` |
| metric | Jaccard `1 − |∩|/|∪|` over top-k; Kendall `(1 − τ)/2` over the union's rankings |

Months: 2016-02 through 2016-10, consecutive. 2016-11 had to be **dropped** — 105 training rows,
where the protocol needs 100 background and 100 explained.

## Deviations, stated because they bound the claim

1. **The monthly partitions are ours.** The authors' script reads monthwise `.npz` files from a path
   local to their machine; those are not in the release. Months here are derived from the released
   `year_month` column. The noise floor is a *within-month* quantity, so it does not depend on our
   month boundaries agreeing with theirs — but arm (c) does, and it is arm (c) that reproduces their
   figure. Anyone checking this should know that.
2. **Six of the nine usable months train on under 2,500 rows** (2016-09 has 585). That is the
   corpus's own shape, and their protocol faces it too. Arm (b) isolates the concern anyway: with the
   model held fixed, however well or badly it was trained, the attribution still churns.
3. **GPU, single machine, three runs per month.** More runs would tighten the standard deviations;
   they are already small relative to the effect.

## Claims licensed

- The statement that **the published explanation-drift figure sits at its own noise floor**, with the
  cause localised to coalition sampling, not to monthly refitting.
- It does **not** license the statement that there is no explanation drift on this dataset. The
  measurement cannot see a signal underneath a floor this high; a different estimator might. That is
  what exact attribution is for, and it is the reason this result belongs *inside* a paper about exact
  attribution instead of standing alone as a correction.

## Reproduce

```
python research/b0-noise-floor/run.py --months 10 --runs 3 --topk 100
```

Raw output in `results.txt`. Takes roughly 40 minutes on one GPU; cost is dominated by the explainer,
at roughly 100 s per call. Needs `torch` and `shap`.
