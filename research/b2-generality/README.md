# b2-generality — the reach of the closed-form Shapley result

The derivation of exact Shapley values for a fuzzy clause used exactly one property of the clause:
that its output depends on **how many** of its literals are unsatisfied, not on which ones. Nothing
else about the response `max(0, ceiling − misses)` was needed.

If that is really the only requirement, the result is not about Fuzzy-Pattern Tsetlin machines. This
finds out which.

## Answer: count-symmetry as the whole requirement

All 2¹³ coalitions brute-forced per case, six random models per arm, compared against the closed form.

| response function `g(misses)` | model class it represents | max &#124;closed − brute&#124; |
|---|---|---|
| `max(0, ceiling − m)` | Fuzzy-Pattern TM clause *(control)* | 5.4e−14 |
| `1[m = 0]` | a conjunction; classical Tsetlin machine | 1.7e−14 |
| `1[m ≤ t]` | m-of-n threshold rule | 4.1e−14 |
| **arbitrary value per miss count** | count-symmetric and nothing else | **2.2e−13** |

The fourth arm is the result. `g` there is a fresh random number for each miss count — no monotonicity,
no threshold, no structure of any kind beyond depending only on the count. The closed form is still
exact. So the construction needs **count-symmetry and additivity**, and nothing whatsoever about
fuzziness, tolerance, or Tsetlin machines.

### The claim this licenses

> For any model of the form **f(x) = Σ_r w_r · g_r(misses_r(x))**, where each `misses_r` counts
> unsatisfied literals in a conjunction of literals and each `g_r` is an arbitrary function of that
> count, exact Shapley values have a closed form costing `O((g+d)²)` per rule, independent of the
> number of features.

Instances of that form include: Fuzzy-Pattern Tsetlin machines; classical Tsetlin machines (a clause
is the `g(m) = 1[m = 0]` case, so the classical machine is a corollary in place of a separate
derivation); m-of-n threshold ensembles; and **weighted rule ensembles of the RuleFit kind**, where a
prediction is a sparse linear combination of indicator functions of conjunctive rules.

That last one is what widens this from a Tsetlin result to a rule-learning result, and it reaches the
rule-based drift literature directly.

## The boundary, established by measurement and not assertion

An **ordered rule list** — RIPPER's "first matching rule wins" — is *not* in the class. The prediction
is not a sum over rules, so Shapley's linearity over components, which the whole construction rests
on, does not apply. Applying the formula anyway gives an error of **1.1e−1**, six orders of magnitude
above the matching arms.

This is in the experiment on purpose. A method claim needs its boundary demonstrated, not asserted,
and "we checked and it genuinely breaks here" is a stronger statement than "we expect it would not
apply." It also means RIPPER itself is **not** covered, only unordered weighted rule ensembles — a
distinction that has to be stated precisely, because it would be easy and wrong to claim otherwise.

## The mechanism

The derivation only ever used the miss count. For a coalition `S`, partitioning a rule's literals by
behaviour at the explained instance `x` versus the background `b`:

```
misses(S) = β + (g − |C ∩ S|) + |D ∩ S|
```

where `β` counts literals unsatisfied under both, `C` those satisfied only under `x` (size `g`), and
`D` those satisfied only under `b` (size `d`). This depends on the two counts alone. Therefore every feature outside `C ∪ D` has Shapley value exactly zero. Members of `C` share one value and members of `D` another.
Averaging over a uniform permutation, with `r` relevant predecessors and a hypergeometric split `k`, collapses the whole thing to `O((g+d)²)` terms.

Generalising from the fuzzy clause changes exactly one line: the marginal contribution of adding a
feature, which was `±1`, becomes `g(M−1) − g(M)` for a member of `C` and `g(M+1) − g(M)` for a member
of `D`. Both are still functions of the count `M = β + g + r − 2k`. Nothing else in the construction
moves, which is why the arbitrary-`g` arm passes.

Two subtleties that are easy to get wrong, both of which cost a debugging cycle when this was first
derived for the fuzzy case:

- A feature whose **both polarities** appear in the same rule lands in `C` and `D` simultaneously.
  Flipping it repairs one literal and breaks the other, so the miss count cannot see it and its value
  is zero — but exactly one of the pair is always unsatisfied, so it contributes a constant miss and
  must be folded into `β`. Omitting that term leaves per-feature values that look plausible and
  violate efficiency.
- **Efficiency is not a sufficient check.** An earlier version summed to the right total while
  dividing it wrongly between features. Both properties are tested separately here and in
  `../../test/shapley_closed_form.jl`.

## Consequence for the paper

The method contribution is now "exact Shapley values for additive ensembles of count-symmetric rules",
with the Tsetlin machine as one instance and the classical machine as a corollary —, not a
result about one model family. The malware application becomes the demonstration of what exactness
buys, which is the ordering the project had been planning for on weaker grounds.

## Reproduce

```
julia --project=. research/b2-generality/run.jl
```

Raw output in `results.txt`. The production implementation, specialised to Tsetlin machines, is
[`julia/shapley.jl`](../../julia/shapley.jl); this script carries its own generic implementation so
that the generalisation is tested, not assumed from the specialised one.
