# local-feature-strength — Blakely & Granmo's *local* expression against exact Shapley

`arXiv:2007.13885` gives **two** closed forms for interpreting a Tsetlin machine, and until now this
project had tested only one of them.

- **Global Feature Strength** (their Eqn 4–6) is the inclusion frequency of a feature across
  positive-polarity clauses. It reads the inclusion sets and nothing else, so it depends on the model
  alone. [`interp-dataset-control/`](../interp-dataset-control/) measured it: 49 of the exact top-100.
- **Local Feature Strength** (their Eqn 7–8) is evaluated **per input**. For an input `X` and
  predicted class `i`,

  ```
  l[k, X] = Σ over positive-polarity clauses j of  C_ji(X)    where x_k = 1 and k ∈ I_ji
  ```

  a bit scores when it is *present* in `X` and included *non-negated* in a clause that fires,
  weighted by that clause's output. Eqn 8 aggregates bits into features.

The paper claimed that "inclusion frequency is data-independent given a fixed model, so it registers
zero explanation drift by construction". That is true of Eqn 4–6 and **false of Eqn 7–8**, which
varies with the input and can therefore drift. The claim was broader than what had been measured.
This experiment measures the rest.

## Implementation

Faithful to their text, including the restriction they state explicitly — *"we are only interested in
indices pertaining to the positive polarity clauses"*. Two arms, because the as-written form is a weak
test on this model:

| arm | what it credits |
|---|---|
| `local (Eqn 7)` | as written: bits present in `X`, included non-negated |
| `local (+negated)` | adds the mirror term: bits absent from `X`, included negated |

About 80% of this model's included literals are negated, so the as-written form ignores most of what
the clauses encode. Reporting only it would be a strawman; reporting only the variant would not be
their method. Both are given.

`C_ji(X)` is the clause output. For the classical machine of their paper that is Boolean; for the
fuzzy vote used here it is `max(0, ceiling − misses)`, which is the faithful reading for this model
and the same quantity the Shapley derivation solves.

## Part 1 — agreement with the exact values

Three seeds, B1 gate configuration, 100 explained rows against 100 background rows from the training
period, which is [`interp-dataset-control/`](../interp-dataset-control/)'s protocol exactly.

| k | local (Eqn 7) | local (+negated) | global (Eqn 4) | document frequency |
|---|---|---|---|---|
| 20 | **11.0** / 20 | 6.0 / 20 | 8.3 / 20 | — |
| 100 | **53.3** / 100 | 37.7 / 100 | 49.3 / 100 | **62** / 100 |
| 500 | 139.0 / 500 | 348.3 / 500 | 348.7 / 500 | — |

**Their local expression beats their global one** at the sizes an analyst reads, 53.3 against 49.3 at
k=100 and 11.0 against 8.3 at k=20. That is a point in its favour and the paper now says so.

**It is still below a document-frequency count that never looks at the model**, which recovers 62.
The sentence that survives is not "inclusion frequency is a poor proxy" but "neither of their
expressions reaches what counting the raw data gives, at this width".

**At k=500 it collapses to 139**, because its support is tiny: the as-written form is nonzero on only
**62 to 79 features** against the exact values' **447 to 482**. A bit scores only where it is present
*and* non-negated, and on Drebin data a row activates a few hundred of 4,561 features. Most of a
top-500 drawn from it is ties.

Adding the negated mirror term reverses the ordering: it is worse at small k (6.0 at k=20) and equal
to the global measure at k=500. The negated literals are satisfied on almost every row, so crediting
them spreads the score over the features the clauses are *not* really keyed on.

## Part 2 — the drift series

One model trained on 2013–14 and held fixed, attributions recomputed for each of 88 months, the
month's own background. This is [`b2-drift/`](../b2-drift/) arm 2's protocol, so the exact column is
directly comparable to its 0.294. It comes out at **0.2943**, which is that number reproduced by a
separate script.

| series | mean | sd | vs exact, Pearson | vs exact, Spearman |
|---|---|---|---|---|
| local (Eqn 7) | **0.138** | 0.103 | +0.550 | +0.526 |
| local (+negated) | **0.331** | 0.248 | **+0.842** | +0.808 |
| exact (arm 2) | 0.294 | 0.159 | — | — |

**It drifts.** The paper's claim that this family of measures "registers zero explanation drift by
construction" was true of the global expression and false of this one, and is now scoped.

**As written it under-reports the movement by half**, 0.138 against 0.294, and tracks it only
moderately at ρ = +0.53. With the negated mirror term it over-reports the level, 0.331, and its
variance is half again as large, but it follows the month-to-month shape closely at **ρ = +0.84**.
So the local expression with negated literals included is a usable indicator of *when* the
attribution moves, and neither arm gives the right *size* of the movement.

**Both are far below the reported figure**, 0.138 and 0.331 against a 0.958 that the benchmark's own
protocol produces and a 0.926 noise floor. Had the benchmark attributed a Tsetlin machine with
Blakely & Granmo's own local expression instead of sampling a neural network, it would have reported
roughly a third of what it did. That is a second, independent route to this paper's central result.

## The scope of the correction

The local expression **can** drift, so the paper's claim had to be restated. It is now scoped to
Global Feature Strength, with the local expression reported on its own numbers.

Three structural differences remain, and none of them is settled by a number:

1. **Positive-polarity clauses only.** The class score here is a signed difference between two banks.
   Their expression reads one of them.
2. **No background, so no counterfactual.** A Shapley value here is defined against a background
   instance — what the model would have done otherwise. Eqn 7 reads the clauses that fired on `X` and
   has nothing to be *as against*, so there is no efficiency property: the values need not sum to
   `f(x) − f(b)`. Efficiency is the axiom that caught a real bug in our own derivation.
3. **Their own claim is correspondence, not identity** — reported on a 30-feature problem, explicitly
   not an equality.

## Reproducing

```
julia --project=. -t 16 research/local-feature-strength/run.jl [nmonths] [nseeds]
```

`results.txt` is the full output, `series.csv` the per-month drift. tm-lab pin `43dba5f`, ceiling
policy `LiteralCapped`, parallel mode `:none` for training, attribution threaded and bit-identical.
