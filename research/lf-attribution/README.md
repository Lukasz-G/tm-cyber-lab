# lf-attribution — the effect of tolerance on credit assignment

`LF` is the ceiling in the clause vote `max(0, ceiling − misses)`. At `LF = 1` the vote is
`1[misses = 0]` — an ordinary conjunction, and so a **classical** Tsetlin machine. As `LF` grows the
clause keeps voting while several literals fail.

The closed form is built directly on that structure, so it can measure what tolerance does to *where
credit goes*. This is the one question in the project that a classical TM cannot be asked.

## Answer: representational, saturating by LF ≈ 5

LAMDA, 2 seeds, everything but `LF` and `T` held at the gate configuration. "support" is how many of the
4,561 features carry nonzero exact attribution; "top10%" is the share of attribution mass in the ten
largest; gini is 0 for an even spread and 1 for all mass on one feature.

| LF | T policy | T | literals | tolerance | **support** | **top10%** | gini | IID F1 | 2021 F1 |
|---|---|---|---|---|---|---|---|---|---|
| **1** | fixed | 10 | 66 | 1.5% | **314** | **48.3%** | 0.785 | 94.07 | 59.68 |
| **1** | scaled | 3 | 64 | 1.6% | **260** | **54.4%** | 0.827 | 92.58 | 52.80 |
| 2 | fixed | 10 | 67 | 3.0% | 431 | 31.4% | 0.742 | 95.47 | 68.81 |
| 2 | scaled | 4 | 66 | 3.1% | 363 | 37.6% | 0.790 | 94.72 | 55.36 |
| 5 | fixed | 10 | 70 | 7.2% | 498 | 24.8% | 0.738 | **95.81** | 67.35 |
| 5 | scaled | 7 | 68 | 7.4% | 478 | 27.2% | 0.758 | 95.64 | 64.10 |
| 10 | fixed | 10 | 68 | 14.6% | 454 | 26.0% | 0.737 | 95.53 | 62.73 |
| 10 | scaled | 10 | 68 | 14.6% | 454 | 26.0% | 0.737 | 95.53 | 62.73 |
| 20 | fixed | 10 | 70 | 28.4% | 457 | 25.6% | 0.759 | 95.05 | 68.09 |
| 20 | scaled | 14 | 70 | 28.6% | 493 | 25.2% | 0.744 | 95.39 | **71.07** |
| 40 | fixed | 10 | 73 | 54.8% | 421 | 24.2% | 0.727 | 94.28 | 63.00 |
| 40 | scaled | 20 | 75 | 53.3% | 474 | 24.1% | 0.741 | 95.04 | 67.02 |

**Tolerance spreads attribution.** Going from the classical conjunction to `LF = 5`, the attribution
support grows from 260–314 features to 478–498, and the mass held by the top ten halves, from 48–54% to
25–27%. Gini falls from 0.79–0.83 to 0.74–0.76. The clause is genuinely scoring sub-patterns and not one dominant conjunction.

**And it stops.** Beyond `LF ≈ 5` nothing moves: support sits between 420 and 500 and the top-ten share
between 24% and 27% all the way to `LF = 40`, an eightfold further increase in tolerance. The effect is
real and it is bounded.

**Clause size barely changes** — the median stays between 64 and 75 literals throughout. Tolerance
redistributes credit among roughly the same-sized clauses instead of growing them.

## The control's contribution

Sweeping `LF` alone would confound it with a mis-scaled `T`, since the published relation is
`T ≈ sqrt(CLAUSES/2 · LF)`. Both policies were run, and **they agree on every direction**: support up,
concentration down, saturating in the same place. So the effect is attributable to `LF`.

They also coincide exactly at `LF = 10`, where the relation returns `T = 10` and the two arms are the same
configuration — which they reproduce to the digit (454 support, 26.0%, 0.737), an internal consistency
check that cost nothing to include.

Where the policies *do* differ is at the bottom: at `LF = 1` the scaled policy gives `T = 3`, and that arm
is the worst in the table (92.58 IID, 52.80 on 2021). The relation is not reliable at the strict end.

## An extension of the sibling project's result

That project measured that a fuzzy vote buys about **3× the resolution of its own binarisation, not `LF`×,
and the factor does not grow with `LF`**. The same shape appears here in a different quantity:
attribution spread is about **1.6×** the classical model's, and it likewise does not grow with `LF`.

Two independent measurements, on different quantities, both say the useful range of tolerance is small and
bounded. **A large `LF` is not buying a proportionally richer model.**

## Claims licensed

- That tolerance is a **representational** parameter, not merely a capacity knob: it changes which
  features carry the model's decisions, not only how well it classifies.
- That the useful range is **`LF` between 2 and 10** on this data, with IID F1 peaking at 5 and the effect
  on attribution saturating there too.
- It does **not** license a claim about why it saturates. The clause size does not grow, so the ceiling
  stops binding on something, but we have not measured what.

## Configuration

Flat FPTM, `clauses_per_class` 20, `S` 100, `L` 64, 30 epochs, `LiteralCapped()`, `parallel = :none`,
tm-lab **43dba5f**, 2 seeds. Attribution over 100 background and 100 explained rows from the training
pool. IID is the held-out test portion of 2013–14 subsampled to 20,000 rows; the drifted column is 2021.

## Reproduce

```
julia --project=. -t 16 research/lf-attribution/run.jl 2
```

About 35 minutes: 24 trainings on 150k rows plus attribution for each.
