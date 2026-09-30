# b5-fairness — both sides swept, thresholds chosen honestly, dominance on every seed

Fixes the three recorded unfairnesses in the APIGraph comparison at once. All three flattered us, so
the expectation was that removing them would widen the gap. **One of them reverses the conclusion
instead**, and the reason is worth more than the number.

What was wrong, and is now fixed:

1. FPTM's threshold was swept while the boosters were read at their default 0.5 cut.
2. Dominance counts came from seed 1 while F1 averaged five seeds.
3. The swept threshold was chosen on the same rows it was scored on — an oracle, not an operating point.

## The two answers disagree, and both are real

APIGraph, 2012 pool split 80/20, train on the 80%, threshold chosen on the held-out 20%, tested per
year. 5 seeds. ~10% malware throughout.

### (a) Threshold-free: the boosters are ahead on every year

Average precision — the whole frontier, no threshold anywhere in it.

| year | FPTM | LightGBM | XGBoost | gap to better booster |
|---|---|---|---|---|
| 2013 | 93.88 ± 0.50 | **94.87** | 94.26 | +0.99 |
| 2014 | 78.67 ± 1.49 | **81.41** | 80.59 | +2.74 |
| 2015 | 76.96 ± 2.27 | **81.23** | 78.98 | +4.27 |
| 2016 | 74.59 ± 1.07 | **79.52** | 77.60 | +4.93 |
| 2017 | 76.44 ± 1.72 | **80.46** | 75.89 | +4.02 |
| 2018 | 76.38 ± 0.91 | 76.43 | **82.06** | +5.67 |

**Mean gap +3.77 AP, ahead on 6 of 6.** The standing conclusion — that a 20-clause FPTM lies inside the
gradient-boosting frontier on APIGraph — **survives** the fair comparison. It is not a threshold
artifact.

### (b) At a threshold you could actually deploy: FPTM is ahead on 5 of 6

Threshold picked on the held-out 20% of the 2012 pool, then applied unchanged to every test year — the
number a deployment gets, with no access to the future.

| year | FPTM | LightGBM | XGBoost |
|---|---|---|---|
| 2013 | **87.30** | 85.33 | 83.47 |
| 2014 | **72.25** | 66.88 | 67.75 |
| 2015 | **67.34** | 63.68 | 64.23 |
| 2016 | 66.60 | 64.30 | **67.07** |
| 2017 | **72.65** | 71.19 | 71.58 |
| 2018 | **70.95** | 58.23 | 70.07 |

FPTM beats both boosters on five years and loses 2016 to XGBoost by 0.47.

### Why they disagree: the oracle was worth far more to the boosters

| model | F1 given up by choosing the threshold honestly |
|---|---|
| FPTM | **+1.73** |
| LightGBM | **+8.96** |
| XGBoost | **+6.14** |

This is the mechanism and it is the actual finding. The boosters have the better frontier, but **their
operating point does not survive the drift** — a threshold chosen on 2012 is badly wrong by 2014, and
recovering their frontier's quality needs a threshold chosen on the test year itself, which a
deployment cannot do. FPTM's margin keeps its calibration: 1.7 F1 between its oracle and its honest
threshold, against 6–9 for the boosters.

So both statements are true and neither may be quoted without the other:

- **The boosters' frontier is better.** +3.77 AP, on every year.
- **FPTM's operating point transfers under drift and theirs does not.** That is why it wins on F1 once
  neither side is allowed to see the future.

Under temporal drift, threshold transfer *is* the deployment problem, so the second statement is the
one an operator cares about.

**It is one dataset.** This experiment runs on APIGraph only. The threshold-transfer result therefore
carries exactly the caveat that sank the drift-robustness claim — which also looked good on one corpus
and did not replicate. Until the same measurement is made on LAMDA it is provisional, and it must not be
described as cross-dataset.

## The correction owed to every earlier number in this repo

Every F1 previously reported here is an oracle value, FPTM's included. The size of that correction is
now measured: **+1.73 F1 for FPTM**. The earlier estimate, from `b5-threshold/`, was that the default
argmax cost ~8.6 F1 — that is a different quantity (default versus oracle, not honest versus oracle)
and both are now on the record.

`b5-apigraph/baselines.json` holds booster F1 at the default 0.5 cut on the *full* 2012 pool. Those
numbers are not comparable with the table above, which trains on 80% of the pool. Nothing here
supersedes that experiment's pre-registered verdict; it supersedes the *comparison basis*.

## Dominance, on every seed instead of one

| year | FPTM frontier reaches outside LightGBM's | outside XGBoost's | max precision excess |
|---|---|---|---|
| 2013 | 5/5 | 5/5 | +6.48 |
| 2014 | 5/5 | 5/5 | +11.22 |
| 2015 | 5/5 | 5/5 | +9.63 |
| 2016 | 5/5 | 5/5 | +11.34 |
| 2017 | 5/5 | 5/5 | +31.64 |
| 2018 | 5/5 | 5/5 | +24.02 |

Consistent across seeds, which settles recorded issue 2: the earlier one-seed counts were not
misleading, just underpowered.

**Read this row carefully, because it is the most over-claimable table in the experiment.** "Reaches
outside" means FPTM's curve exceeds the booster's precision at *some* recall — a far weaker and more
generous test than dominance. Since FPTM's average precision is *lower* on every year, the curves must
cross: FPTM is better in one region and worse overall. The region is the high-precision, low-recall end,
consistent with the 96%-precision-at-10%-recall operating point found in `b5-diagnosis/`. This table
does **not** say FPTM dominates; it says neither frontier contains the other.

## Arms

| arm | swept over |
|---|---|
| FPTM | decision margin, whole range, 5 seeds |
| LightGBM | `predict_proba`, whole range |
| XGBoost | `predict_proba`, whole range |

and three readings: average precision (threshold-free), validation-chosen threshold (deployable),
oracle best-F1 (upper bound, kept only to size the correction).

## A bug the design caught, worth recording

The first version split the 2012 pool 80/20 **by row order**. The validation slice came out at **0.0%
malware** — the pool is ordered by class, so every positive landed in the training half and the
honest-threshold fix would have had nothing to select a threshold on. It would not have crashed; it
would have silently produced a meaningless threshold.

The split is now interleaved (every 5th row), 10.0% malware in both halves, and the **exact indices are
exported from Julia rather than the rule re-derived in Python**, so the two languages cannot disagree.
Both sides also assert that their label vectors match row for row before anything is computed — that
check is what would have caught a silent misalignment, and it passed on the validation slice and all
six test years.

## Configuration

FPTM: `clauses_per_class` 20, `T` 10, `S` 25, `L` 64, `LF` 10, 30 epochs, `LiteralCapped()`,
`parallel = :none`, tm-lab **43dba5f**, width 1,159, 5 seeds.
Boosters: `n_estimators` 5000, `learning_rate` 0.02, 256 leaves, `random_state` 0 — the same
configuration used on LAMDA, so the two datasets are treated identically.
Pool 30,533 rows → 24,427 train / 6,106 validation, both 10.0% malware.

## Reproduce

```
julia --project=. -t 16 research/b5-fairness/export_margins.jl 5   # -> data/b5-fairness/
python research/b5-fairness/run.py                                 # -> results.txt
```

Under two minutes for the export, a few minutes for the boosters.
