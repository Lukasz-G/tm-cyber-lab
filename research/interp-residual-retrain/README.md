# interp-residual-retrain — pruning the residual, retrained and over every year

`interp-residual/` found that forcing the 38 attributed-but-not-frequent features to zero *at inference*
improves drifted-year F1 — by 10.32 on 2021. Two things were missing before that could be a
recommendation: it was two years, and inference ablation cannot distinguish "this model uses them" from
"a better model ignores them". Both are supplied here: every period, and the model **retrained** from
scratch per arm.

## Answer: real, substantial, and conditional on the period

5 seeds. Every arm retrained. F1, higher is better.

| eval | baseline | zero-residual | zero-random | **resid − random** | zero-freqonly |
|---|---|---|---|---|---|
| IID | 95.63 | 95.18 | 95.52 | −0.34 | 95.53 |
| 2016 | 84.52 | 83.75 | 83.57 | +0.18 | 83.95 |
| 2017 | 34.33 | 31.56 | 34.60 | **−3.04** | 34.35 |
| 2018 | 27.90 | 26.52 | 27.03 | −0.51 | 30.37 |
| 2019 | 77.25 | **79.42** | 75.81 | **+3.61** | 76.67 |
| 2020 | 75.93 | **80.30** | 74.91 | **+5.39** | 76.61 |
| 2021 | 68.08 | **73.26** | 66.49 | **+6.77** | 68.97 |
| 2022 | 74.69 | **78.83** | 73.49 | **+5.35** | 74.72 |

**The claim is the `resid − random` column**, not the comparison against baseline: arms 2–4 each zero
exactly 38 features and so share the dead-channel perturbation, while the baseline does not carry it.
Only residual-against-random isolates *which* features were removed from the fact of removing some.

### It survives retraining, at about half the magnitude

On 2021, inference ablation gave +10.32; retraining gives **+6.77** over the matched control and **+5.18**
over the unpruned baseline. The direction holds and the effect is real, but the retrained model partially
relearns the shortcut, so the inference figure overstated it. That is the expected relationship, and it is
the reason this arm was necessary.

### The split by period, and its distinctness from noise

| periods | resid − random | resid − baseline |
|---|---|---|
| **2019–2022** | **+5.28** | **+3.96** |
| 2016–2018 | −1.12 | −1.64 |

Four consecutive later years all land between +3.6 and +6.8. Three earlier years are flat or negative,
with 2017 costing 3.04. "5 of 7" undersells the structure: this is a clean regime split, not a majority.

**So it is not yet a blanket recommendation.** A pruning step you cannot know in advance whether to apply
is not deployable as a rule, and the pre-registered criterion — a majority of drifted years — is met on a
count while concealing that the failures are contiguous.

### A hypothesis about the two regimes, since refuted

2017 and 2018 are exactly the years `b3-label-drift/` found the label boundary most fragile: 51% and 58%
of malware sitting at 4–6 detections, against 21–27% in the years where pruning works. 2018 is also the
collapse year for every model here (baseline 27.90). A plausible reading is that pruning the
non-generalising features helps when the labels are stable enough for "generalise" to mean something, and
cannot help when the evaluation labels are themselves unreliable.

**It was tested, and it is refuted in sign.** `prune-vs-fragility/` conditioned the pruning gain on label
fragility over 70 months and found $\rho = +0.253$ ($p = 0.030$): pruning helps roughly *twice as much*
where the labels are least reliable, which is the opposite of the prediction above. A month-size control
clears ($-0.147$, $p = 0.22$), so thin months are not the explanation either. The hypothesis is dead and no
replacement is offered — we have no account of the regime split.

## Claims now permitted

- **Confirmed:** exact attribution, computed from training data alone, identifies a 38-feature subset
  whose removal improves detection on later drifted periods by ~4 F1 over the unpruned model and ~5 over a
  size-matched control. It survives retraining, so it is a property of the features and not of one model.
- **Confirmed:** the mechanism in `interp-residual/` — that the attributed-but-not-frequent features are
  where the model fails to generalise — holds under the stronger test.
- **Not claimable:** that pruning should be applied unconditionally. It hurts on 2017 and is neutral on
  2016 and 2018.
- **Refuted:** the label-fragility explanation for the regime split, in sign, over 70 months (`prune-vs-fragility/`).
- **Still not claimable:** anything about readable rules. This makes a feature set diagnostic, not a
  clause legible.

## Zeroing in place of dropping

Deleting 38 of 4,561 columns changes the input width, and two things in this model depend on width: the
effective specificity `s = width/S`, which in a sibling measurement accounted for more of an apparent
effect than the thing being studied, and the `L` growth gate. Zeroing holds width, `S` and `s` identical
across all four arms.

Zeroing is not free either — an always-zero feature is satisfied for free by its negation, so a clause
includes it at no evaluation cost, inflating its literal count and shifting the `L` gate. That effect was
worth +0.0109 accuracy for 32 dead bits in the sibling project. The defence is that **arms 2, 3 and 4 each
zero exactly 38 features**, so the perturbation is identical between them, which is precisely why the
verdict rests on arm 2 against arm 3 and not on arm 2 against the baseline.

**That defence is incomplete, and `residual-density-control/` is the repair.** Zeroing is not
sign-neutral: it removes the evidence for a positive literal but *satisfies* a negated one. The residual
features sit at a training-period presence rate of 11.2%, roughly ten times the uniformly-drawn control's,
and presence rate is what sets sign composition — a feature that is almost always absent is included
negated. So arm 3 is matched on count but not on the thing that governs how much the zeroing does. The
control matched on presence rate is in `residual-density-control/`, and the verdict there supersedes the
`resid - random` column here.

## The mirror arm's unpredicted behaviour

`interp-residual/` found that ablating the frequency-only features *hurt* under drift, suggesting they
were the more drift-stable set. Retrained, `zero-freqonly` is indistinguishable from baseline on most
years and **better** on 2018 (30.37 against 27.90). So the mirror prediction does not survive retraining,
and the inference-time reading of that arm should be treated as a property of the fixed model.

## Configuration

Flat FPTM, 20 clauses per class, `T` 10, `S` 100, `L` 64, `LF` 10, 30 epochs, `LiteralCapped()`,
`parallel = :none`, tm-lab **43dba5f**, 5 seeds. Per seed: train on all features, attribute with 100
background and 100 explained rows from the 2013–14 pool, take `k = 100`, then retrain each arm with its
38 features zeroed in both training and evaluation inputs. Support 447–482 features; residual, frequency-only
and random sets are 38 each in every seed. Evaluation subsampled to 20,000 rows per period, fixed seed.

No test period is touched when the residual is chosen, so the procedure is one a deployment could run.

## Reproduce

```
julia --project=. -t 16 research/interp-residual-retrain/run.jl 5
```

About 45 minutes: 20 trainings on 150k rows plus attribution per seed.
