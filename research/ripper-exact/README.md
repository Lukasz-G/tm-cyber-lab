# ripper-exact — the closed form on a rule ensemble that is not a Tsetlin machine

`b2-generality/` proved the closed form needs only count-symmetry plus additivity, and verified it on
constructed models. Constructed models are a weak demonstration: they were built by the same person who
derived the formula. This runs it on rules induced by **RIPPER** — a 1990s separate-and-conquer algorithm
with no connection to this work — on real malware data.

## Scope, stated first because getting it wrong would be the obvious error

**RIPPER's own prediction is a disjunction** — positive if *any* rule covers the instance. A disjunction
is not a sum, so **a raw RIPPER ruleset is outside the class**, for exactly the reason ordered rule lists
are (`b2-generality/` measures that boundary at 1.1e−1 error).

What is inside the class, and what is verified here, is the **weighted rule ensemble** built on those
rules: `score(x) = Σ_r w_r · 1[rule_r fires]`, with weights fitted by logistic regression on the rule
activations. That is a standard interpretable model in its own right, and each rule is a conjunction —
which is the ceiling-1 case of the fuzzy clause vote, since `max(0, 1 − misses) = 1[misses = 0]`. No
separate derivation is needed and none is used.

## Arm 1 — brute force says EXACT, on rules this project did not construct

RIPPER refit on the 13 strongest features, every one of the 2¹³ coalitions enumerated, Shapley computed
from the definition and compared against the closed form, over 6 (instance, background) pairs, with the
efficiency check included.

**Max |closed form − brute force| = 1.29 × 10⁻¹⁴**, against the 1e−9 bar the Julia test uses. 18 rules,
3.9 conditions each.

This is also a **cross-language, cross-implementation check**: the formula was derived and implemented in
Julia, and reimplemented here in Python from the derivation. Agreement at 1e−14 tests the derivation, not
only the port.

## Arm 2 — it runs at realistic width

| | |
|---|---|
| rules induced | 21, over 1,159 features, in 40 s |
| conditions per rule | 3.4 |
| distinct features the ensemble reads | 38 |
| **exact attribution, 100 × 100** | **0.7 s** |
| features with nonzero value | 34 |

Cheaper than on the Tsetlin machine (6.7 s at the same budget), and for the same structural reason: cost
scales with rule size, not with the feature count, and RIPPER's rules are short.

## Arm 3 — the paper's central finding is not Tsetlin-specific

Sampled `KernelExplainer` at `nsamples=100` against the exact values, on this identical non-TM model:

| k | sampled vs **exact** | sampled vs **itself** (different seed) |
|---|---|---|
| 20 | 0.621 | 0.667 |
| 50 | **0.765** | **0.765** |

The estimator lands as far from the truth as it lands from a second run of itself — identical to three
decimals at k = 50. **The error is variance rather than bias here too**, which is the paper's central
attribution result reproduced on a learner with no relationship to a Tsetlin machine.

That matters for how the contribution should be read: it is a statement about sampled attribution at a
realistic budget, and about what exactness buys, not a statement about one model family.

## What this licenses

- **The scope claim in the paper is verified rather than asserted.** "Weighted rule ensembles" is in the
  class, demonstrated on an externally-induced ruleset, checked against brute force at 1e−14.
- **The variance-not-bias result generalises** beyond Tsetlin machines.
- It does **not** extend the class. Disjunctive RIPPER and ordered rule lists remain outside it, and the
  README says so before it says anything else.

## Configuration

APIGraph 2012 pool, 6,000-row subsample, 1,159 features, ~10% malware. `wittgenstein` RIPPER,
`max_rules=40`, `random_state=0`. Weights by `LogisticRegression(C=1.0)` on the rule-activation matrix.
Attribution over 100 background and 100 explained rows, matching the budget used everywhere else in this
project.

## Reproduce

```
python research/ripper-exact/run.py --rows 6000 --small-features 13
```

About two minutes, dominated by RIPPER induction and the two `KernelExplainer` passes.
