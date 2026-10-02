# threshold-bias — the decision threshold at up to 33 F1, growing with drift

A reanalysis, not a new run: it reads the `RESULT` lines already written by `b5-fairness/`,
`b1-threshold-transfer/` and `calibration-check/`. The question is not which model wins but how much the
*choice of decision threshold* — usually an unstated free parameter — is worth in a temporal evaluation.

## The finding

$F_1$ inflation from an **oracle** threshold, one chosen to maximise $F_1$ on the very period being
scored, against an **achievable** one chosen on held-out data from the training period and applied
unchanged:

| corpus | model | in-distribution | drifted periods | growth |
|---|---|---|---|---|
| LAMDA | FPTM | **0.03** | 7.07 | +7.04 |
| LAMDA | LightGBM | **0.03** | **18.97** | +18.94 |
| LAMDA | XGBoost | **0.03** | 17.50 | +17.47 |
| APIGraph | FPTM | 0.78 | 1.91 | +1.13 |
| APIGraph | LightGBM | 3.82 | 9.99 | +6.17 |
| APIGraph | XGBoost | 5.07 | 6.36 | +1.29 |

Worst single period: **33.38 F1** (LightGBM, LAMDA 2018). Median across all 42 model-period
combinations: 6.21; on drifted periods only, 8.61.

**In distribution the inflation is essentially zero — 0.03 for all three models on LAMDA. It is a
drift-specific bias**, and it grows precisely where a temporal-evaluation paper locates its contribution.

## A field observation in place of a model comparison

Three things make it more than a curiosity about our own numbers.

**It affects every model.** A bias hitting one learner would be a fact about that learner. This hits a
rule ensemble and two gradient boosters alike.

**It is unequal between model families, so it can reorder a comparison.** On LAMDA it is worth 7.07 to
FPTM and 18.97 to LightGBM. A table built from oracle thresholds and a table built from achievable ones
do not rank the same models in the same order — and that is not hypothetical, it is what happened to this
project: an advantage we measured on two corpora dissolved once the threshold rule changed
(`calibration-check/`).

**Both common choices are wrong, in opposite directions.** The oracle inflates by up to 33. The other
common default — the model's own argmax — *deflates*: `b5-threshold/` measured it costing **8.6 F1** on
average on APIGraph. So the span between two defensible-looking reporting choices exceeds most published
effect sizes, and neither is the number a deployment gets.

## The fix, effective for every model family

Mean oracle gap by rule, from `calibration-check/`:

| corpus | model | max-F1 on validation | **rate matching** |
|---|---|---|---|
| APIGraph | FPTM | 2.55 | 0.82 |
| APIGraph | LightGBM | 8.96 | 2.26 |
| APIGraph | XGBoost | 6.14 | **0.60** |
| LAMDA | FPTM | 5.82 | 3.35 |
| LAMDA | LightGBM | 18.97 | **1.78** |
| LAMDA | XGBoost | 17.50 | 3.53 |

Rate matching sets the threshold so the predicted positive rate on the test period equals the rate the
validation threshold produced. It needs **only unlabelled test features**, so a deployment can run it, and
it closes most of the gap for every model. Note it helps the boosters more than it helps us — which is
why this is advice about thresholding in place of a claim about model class.

## Limits of the claim

**We do not claim that specific published work reports oracle thresholds.** We checked the benchmark used
here: its paper states that "no task-specific tuning or dataset-specific hyperparameter adjustments are
performed", which reads as defaults and not oracles. The claim is narrower and still worth making:

- the threshold rule is a free parameter worth more than the effects typically reported;
- it is usually **unstated**, so a reader cannot tell which of the three numbers a table contains;
- and the spread grows with drift, so temporal papers are the ones most exposed.

Every $F_1$ this project reported before 2026-09-30 was an oracle value, which is the concrete instance we
can verify, and it is why the measurement was made.

## Recommendation

State the threshold rule. Prefer one that uses no test labels. Of those tested, **rate matching** is the
best, and reporting both it and the oracle bounds what a reader needs to know.

## Reproduce

```
python research/threshold-bias/analyse.py
python paper/figures/threshold-bias.py
```

No training; it reparses the three source experiments' output. Writes `per-period.csv`.
