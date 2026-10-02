# interp-residual — the 38 features frequency misses, and what they turn out to be

`interp-dataset-control/` retracted the readability claim: a document-frequency count recovers 62 of the
model's exact-attribution top-100. Its pre-registration said the remaining 38 licenses a narrow claim
*only if shown to matter*. This shows whether it does.

Ablation is at inference — the selected features are forced to zero — which is the right test for an
attribution claim, since attribution asserts these features drive *this* model's output.

## Answer: a load-bearing residual, bearing the model's overfitting

$F_1$ **drops** from forcing each set to zero. Positive means the ablation hurt; negative means it
*helped*. 3 seeds, sets sized 38 and 62 with size-matched random controls drawn from the
nonzero-attribution support.

| eval | base $F_1$ | residual (38) | random (38) | overlap (62) | random (62) | freq-only (38) |
|---|---|---|---|---|---|---|
| IID | 95.59 | **+2.17** | +0.12 | **+88.85** | +0.30 | +0.33 |
| 2019 | 76.85 | **−2.63** | −0.25 | +72.59 | −0.66 | +2.32 |
| 2021 | 66.66 | **−10.32** | −1.24 | +66.50 | −1.59 | +3.71 |

### 1. Sixty-two features as the model, all found by a frequency count

Ablating the overlap set destroys the classifier: 95.59 → 6.74 on IID, and on 2021 it removes
essentially all skill. Sixty-two features out of 4,561 carry the model. **They are exactly the features
identifiable without any model at all**, which is the retraction in `interp-dataset-control/` restated as
a causal fact in place of a rank correlation. A size-matched random set costs 0.30.

### 2. A real residual at 18× random, and the locus of failed generalisation

On IID the residual costs **2.17** against **0.12** for random at the same size, so the attribution is
finding genuine model-specific structure that frequency misses. It is not ranking noise.

But on both drifted years **ablating the residual improves $F_1$**, by 2.63 on 2019 and **10.32 on 2021**,
against 0.25 and 1.24 for random. The 38 features the attribution finds and a frequency count does not
are the features that **do not survive the drift**. They are the model's overfitting to its training
period, and exact attribution localises them.

This is worth more than the readability claim it replaces, and it is a drift-forensics claim, not an interpretability one, which fits this project's thesis better. **It is also an intervention:** the
residual is identified from training-period data alone — the attribution uses background and explained
rows drawn from the 2013–14 pool and never touches a test year — so dropping those 38 features is
something a deployment could actually do, for a 10-point gain on the worst year measured.

### 3. The dismissed features as the more drift-stable ones

`freq-only` is frequency's top-100 minus the model's: features a univariate ranking hands an analyst and
the attribution ranks below the top 100. Ablating them costs almost nothing in-distribution (0.33) but
**2.32 and 3.71 on the drifted years** — the mirror image of the residual. The model does use them, and
what it draws from them travels better than what it draws from its own top-ranked idiosyncrasies.

## Claims licensed and refused

- **Claimable:** exact attribution separates, from training data alone, the features a model relies on
  that generalise from those that do not. The non-generalising set is small (38 of 4,561) and removing it
  improves drifted-year $F_1$.
- **Claimable:** the readability retraction is confirmed causally. The features that carry the model are
  the ones a frequency count already supplies.
- **Not claimable yet:** the 10.32-point gain. Three seeds and two drifted years is not enough, the
  effect is much smaller on 2019 than on 2021, and the obvious stronger arm — *retrain* without those
  features, not ablate at inference — has not been run. Ablation answers "does this model use
  them"; retraining answers "is there a better model that ignores them", and only the second supports a
  recommendation.
- **Not claimable:** anything about human-readable rules. Nothing here makes a clause legible; it makes a
  feature set diagnostic.

## Provenance of the random control

4,065 of the 4,561 features have exactly zero exact attribution, so ablating a random selection from the
full width is guaranteed to do almost nothing. A random baseline over all features would have made any
real set look load-bearing by comparison. The controls here are drawn from the nonzero-attribution
support outside the top-100, so they are matched on being in play as well as on size — which is why
their drops are small but non-zero.

## Configuration

Flat FPTM, `clauses_per_class` 20, `T` 10, `S` 100, `L` 64, `LF` 10, 30 epochs, `LiteralCapped()`,
`parallel = :none`, tm-lab **43dba5f**, 3 seeds. Attribution: 100 background and 100 explained rows from
the 2013–14 training pool, $k = 100$. Support runs 447–482 features across seeds; residual is 38 and
overlap 62 in all three, which is itself a stability check on the partition. Evaluation subsampled to
20,000 rows per set with a fixed seed.

## Reproduce

```
julia --project=. -t 8 research/interp-residual/run.jl 3
```
