# residual-density-control — the pruning gain as a matter of identity

`interp-residual-retrain/` found that zeroing the 38 attributed-but-not-frequent features and retraining
improves F1 over 2019–2022 by **+5.28** against a size-matched random control. That control was matched on
count and on nothing else, and this experiment exists because count is not the variable that decides how
much the zeroing does.

## Insufficiency of the published control

**Zeroing is not sign-neutral.** It removes the evidence for a positive literal and *satisfies* a negated
one, so what the intervention actually does to a clause depends on the sign its literals carry. And sign
is set by presence rate: a feature that is almost always absent gets included negated, because a negated
literal is then satisfied for free on nearly every row.

The two sets are far apart on exactly that axis. The residual sits at a **11.1%** training-period presence
rate; a uniform draw from the nonzero-attribution support sits at **2.6%**. `blacklist-anatomy/` measured
the consequence at 79.1% negated against 94.8%. So the uniform control was receiving a systematically
*weaker* intervention than the residual, and the published +5.28 could in principle have been reporting
density and not which features were chosen.

## Answer: identity, with a larger effect under the better control

Arms 1–3 reproduce `interp-residual-retrain/` **to the decimal on all eight periods**, so arm 4 is read on
exactly the same scale. Arm 4 replaces the uniform draw with 38 features matched *feature by feature* on
presence rate, from the same pool.

| eval | baseline | zero-resid | zero-unif | zero-dens | resid − unif | **resid − dens** |
|---|---|---|---|---|---|---|
| IID | 95.63 | 95.18 | 95.52 | 95.45 | −0.34 | −0.28 |
| 2016 | 84.52 | 83.75 | 83.57 | 83.55 | +0.18 | +0.20 |
| 2017 | 34.33 | 31.56 | 34.60 | 35.80 | −3.04 | **−4.24** |
| 2018 | 27.90 | 26.52 | 27.03 | 28.63 | −0.51 | **−2.11** |
| 2019 | 77.25 | **79.42** | 75.81 | 75.87 | +3.61 | **+3.55** |
| 2020 | 75.93 | **80.30** | 74.91 | 74.20 | +5.39 | **+6.10** |
| 2021 | 68.08 | **73.26** | 66.49 | 64.61 | +6.77 | **+8.65** |
| 2022 | 74.69 | **78.83** | 73.49 | 72.87 | +5.35 | **+5.97** |

| block | resid − unif | resid − dens |
|---|---|---|
| **2019–2022** | +5.28 | **+6.07** |
| 2016–2018 | −1.12 | **−2.05** |

**The verdict is the direction, not just the sign.** Closing most of the density gap *raises* the
residual's advantage from +5.28 to +6.07. Density was masking part of the effect instead of manufacturing it, so the published figure was conservative. Had the matched control gained as much as the
residual, the drift-forensics claim would have been withdrawn here; instead it survives its strongest
available objection.

The regime split survives too, and deepens: the early block goes from −1.12 to −2.05, with 2017 at −4.24.
Both failing periods are ones where the density-matched control *beats the unpruned baseline* (35.80
against 34.33 on 2017, 28.63 against 27.90 on 2018), which the uniform control did not do. So on the
early periods it is not that pruning the residual is merely useless — pruning almost any dense set helps
there, and pruning the residual specifically helps less. We still have no account of the split.

## Quality of the match, on which the verdict depends

Partial, and the shortfall is reported because the verdict depends on reading it correctly.

| | presence rate | negated | included literals |
|---|---|---|---|
| residual (target) | **11.11%** | 79.1% | 132–168 |
| zero-dens | 7.65% | 86.6% | 71–85 |
| zero-unif (published control) | 2.58% | 94.8% | 62–74 |

The match closes about **60%** of the presence-rate gap and about **52%** of the sign gap, and only about
a fifth of the literal-count gap. Median per-feature density error 0.39–1.76 pp across five seeds. The
pool simply does not contain 38 unused features at 11% presence, which is itself informative: the
residual is unusually dense for a set drawn from the attribution support.

**That is why the direction matters more than the magnitude.** The confound is controlled only partway,
and over that partial range it moves the effect *up*. Extrapolating a monotone trend, a perfect match
would not reverse the sign. We state that as the reading of a partial match and not as a measurement
of a complete one.

**One seed went the other way on sign.** On seed 4 the density-matched set landed at 94.59% negated,
slightly *worse* than that seed's uniform control at 93.24%, because presence rate predicts sign on
average and not per draw. The sign improvement is a mean over seeds, not a property of every seed.

## Claims licensed

- **Confirmed, against the strongest control we can build:** exact attribution computed from training-period
  data alone identifies a feature subset whose removal improves detection on 2019–2022 by ~6 F1 over a
  control matched on the variable that governs the intervention's strength. The effect is about *which*
  features, not how dense they are.
- **The caveat in the manuscript is now a result.** The paper previously said a control matched on size
  *and* sign was the right comparison and that we had not run it. It has been run, in the stronger
  upstream form, and the direction is against the confound.
- **Still not claimable:** that pruning should be applied unconditionally. The regime split is larger under
  this control, not smaller, and remains unexplained after the label-fragility account was refuted in sign
  (`prune-vs-fragility/`).
- **Still not claimable:** anything about readable clauses. This makes a feature set diagnostic.

## Configuration

Flat FPTM, 20 clauses per class, `T` 10, `S` 100, `L` 64, `LF` 10, 30 epochs, `LiteralCapped()`,
`parallel = :none`, tm-lab **43dba5f**, 5 seeds. Per seed: train on all features, attribute with 100
background and 100 explained rows from the 2013–14 pool, take `k = 100`, then retrain each arm with its
features zeroed in training and evaluation inputs alike. Features are **zeroed, not dropped**, so width,
`S` and `s = width/S` are identical in every arm and arms 2–4 share the dead-channel perturbation. Set
sizes are 34–39 depending on seed and are equal across arms *within* a seed, which is what the comparison
needs. Evaluation subsampled to 20,000 rows per period, fixed seed. No test period is touched when any
set is chosen.

Sign composition is **reported, never selected on**. Matching directly on the negated share would have
answered the objection as stated whilst leaving the mechanism beneath it untested; presence rate is
model-free and upstream, so matching on it tests the cause and lets the sign fall where it falls.

## Reproduce

```
julia --project=. -t 16 research/residual-density-control/run.jl 5
```

About 40 minutes: 20 retrainings on 150k rows plus attribution per seed.
