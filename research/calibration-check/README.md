# calibration-check — the threshold-transfer advantage under a better threshold rule

`b5-fairness/` and `b1-threshold-transfer/` reported that gradient boosting holds the better
precision/recall frontier while its *operating point* does not survive drift, and that this was the one
claim in the project holding on two corpora. The obvious objection is that boosters are famously
miscalibrated and that Platt scaling or isotonic regression would fix it.

**The objection as stated is wrong, and a stronger version of it is right.**

## Part 1 — calibration as a no-op, exactly as predicted

Stated before the run, from the structure and not from data: Platt and isotonic are both **monotone**,
so they cannot change a ranking. Average precision is invariant, and an F1-maximising threshold chosen on
the validation slice is the same decision rule however the scores are relabelled.

| corpus | model | raw | Platt | isotonic |
|---|---|---|---|---|
| APIGraph | FPTM | 72.51 | 72.51 | 72.51 |
| APIGraph | LightGBM | 68.27 | 68.27 | 68.27 |
| APIGraph | XGBoost | 70.69 | 70.69 | 70.69 |
| LAMDA | FPTM | 70.93 | 70.93 | 70.93 |
| LAMDA | LightGBM | 58.20 | 58.20 | 58.20 |
| LAMDA | XGBoost | 60.27 | 60.27 | 60.27 |

Identical to the decimal in all six model-corpus combinations. **Calibration changes nothing about
threshold transfer, and this settles that objection structurally.** Arm 3 fixes `p = 0.5` after Platt scaling, which is the one arm where calibration can act on its own.
It is slightly *worse* than the validation threshold everywhere except XGBoost on APIGraph.

## Part 2 — the selection rule as the whole effect

Arm 5 sets the threshold so the **predicted positive rate** on each test period matches the rate the
validation threshold produced. It uses only unlabelled test features, so a deployment can run it.

F1 given up against each model's own oracle — lower is better:

| corpus | model | max-F1 on validation | **quantile matching** |
|---|---|---|---|
| APIGraph | FPTM | 2.55 | **0.82** |
| APIGraph | LightGBM | 8.96 | **2.26** |
| APIGraph | XGBoost | 6.14 | **0.60** |
| LAMDA | FPTM | 5.82 | **3.35** |
| LAMDA | LightGBM | 18.97 | **1.78** |
| LAMDA | XGBoost | 17.50 | **3.53** |

And mean F1 under quantile matching:

| corpus | FPTM | LightGBM | XGBoost |
|---|---|---|---|
| APIGraph | 74.23 | 74.98 | **76.23** |
| LAMDA | 73.40 | **75.38** | 74.23 |

**On both corpora a booster now leads.** The pre-registered criterion — the claim fails if any calibrated
booster arm closes the gap to FPTM — fires on both.

## Verdict — the advantage as a property of the threshold rule

The boosters' operating point does transfer badly under a max-F1-on-validation rule, and that is what the
earlier experiments measured. It is not a fact about gradient boosting; it is a fact about choosing a
threshold by maximising F1 on a slice drawn from a different period. Once the threshold is set by rate
matching instead, the boosters' transfer problem largely disappears — 18.97 → 1.78 on LAMDA — and they
overtake.

So **the manuscript's claim (iii) must be rewritten again**, and this time weakened, not narrowed.
What can honestly be said:

- The rule ensemble's margin is **more robust to a naive threshold rule**. That is a real difference and
  it is measured on two corpora.
- It has **no operating-point advantage once a drift-aware threshold rule is used**, and under rate
  matching it is behind on mean F1 on both corpora.
- The deployable recommendation is therefore **"match the predicted positive rate"**, which is advice
  about thresholding and not about model class. It improves every model here, including ours.

This leaves the project with **no surviving detection advantage**: the drift-robustness claim died on the
cross-dataset check, and threshold transfer dies here. The attribution results are unaffected and remain
the paper's contribution.

## The non-uniform cost to FPTM

Rate matching helps FPTM too (2.55 → 0.82 and 5.82 → 3.35) but unevenly: on LAMDA 2017 it *hurts*,
38.10 → 29.37, while on 2018 it helps sharply, 34.01 → 48.07. Those are the two periods where
`b3-label-drift/` finds the label boundary most fragile, so the rate observed on the validation period is
least informative about them. We note the coincidence and do not build on it — the one time this project
built on that coincidence it was refuted (`prune-vs-fragility/`).

## The five arms

| arm | what it isolates |
|---|---|
| 1. raw score, max-F1 on validation | the published comparison |
| 2. Platt + max-F1 on validation | the literal objection |
| 3. isotonic + max-F1 on validation | the same, non-parametrically |
| 4. Platt + fixed `p = 0.5` | the strongest form: the rule calibration is supposed to make meaningful, and the only arm where calibration alone can move anything |
| 5. quantile matching | the technique that can genuinely beat it — included so the answer is not "calibration does not help" when something else does |

Arm 5 is the one that mattered, and leaving it out would have produced a comfortable and wrong conclusion.

## Configuration

Reuses the FPTM margins exported by `b5-fairness/` and `b1-threshold-transfer/`, so that side is the
identical model on identical rows with nothing retrained. Boosters refitted here because their
probabilities were not saved: 5000 estimators, learning rate 0.02, 256 leaves, `random_state` 0. Label
vectors are asserted to match the Julia exports row for row before anything is computed.

Per-period figures only; no mean across LAMDA's late years.

## Reproduce

```
python research/calibration-check/run.py
```

Requires `data/b5-fairness/fptm_margins.json` and `data/b1-threshold-transfer/fptm_margins.json` from the
two export scripts. About 25 minutes, dominated by refitting the boosters on LAMDA.
