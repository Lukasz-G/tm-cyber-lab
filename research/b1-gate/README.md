# b1-gate — flat Fuzzy-Pattern TM against gradient boosting on LAMDA

**This was a real gate with a stated stopping condition**: match the baselines at roughly twenty
clauses per class, or stop and reconsider the domain. The criterion and the configuration were written
into the script header before the run, and the clause budget was not raised afterwards.

## Answer: PASS, with the drift behaviour as the interesting part

Flat FPTM, **20 clauses per class** across both polarities (ten per polarity), `T` = 10, `S` = 100,
`L` = 64, `LF` = 10, `LiteralCapped` ceiling, `parallel = :none`, 30 epochs, mean ± sd over **ten
seeds**. Trained on the released 2013–14 training portions (150,090 rows). tm-lab pin 43dba5f.

| set | FPTM | LightGBM | XGBoost | vs. LGB | gap / SEM |
|---|---|---|---|---|---|
| IID | 95.67 ±0.14 | 97.49 | 97.18 | −1.82 | — |
| NEAR | 72.35 ±1.22 | 72.12 | 71.46 | +0.23 | 0.6 |
| 2018 | 28.10 ±3.21 | 28.32 | 25.31 | −0.22 | 0.2 |
| 2019 | 76.40 ±2.11 | 72.22 | 70.73 | **+4.18** | 6.3 |
| 2020 | 74.51 ±3.76 | 67.77 | 65.89 | **+6.74** | 5.7 |
| 2021 | 65.61 ±7.41 | 53.87 | 56.25 | **+11.74** | 5.0 |
| 2022 | 74.34 ±3.42 | 64.90 | 62.22 | **+9.44** | 8.7 |

IID gap 1.82 against a criterion of 3. Mean absolute gap across NEAR and the five FAR years 5.43
against a criterion of 10.

So: **1.8 F1 points behind gradient boosting on in-distribution data, and 4 to 12 points ahead of it
on four of the five drifted years**, at a model holding a few hundred literals per class. The FAR
advantages are 5 to 9 standard errors out, so they are not seed noise.

## A deliberate comparison against our own gradient boosting

Not against the published figures. Our LightGBM reproduces the published IID F1 to two decimal places, 97.49 against 97.49.
The published NEAR figure turns out to be a **mean of per-year F1** and not a pooled one.
The FAR figure is not reconstructible from the release at all — see
`../b0-baseline/`. Comparing FPTM against numbers we computed on identical row sets removes that
ambiguity completely. The relationship between our gradient boosting and theirs is reported
separately, where it belongs.

## Non-monotone drift in time, for all three models

This is the most surprising thing in the table and it is not about Tsetlin machines.

2018 — the year immediately after the training window — is **the worst year for every model**:
LightGBM 28.32, XGBoost 25.31, FPTM 28.10. Every later year is better, and 2019 is roughly two and a
half times better than 2018 for all three. Precision stays at 82–99% throughout while recall
collapses, so the failure mode is missed malware, not false alarms, as the benchmark's authors
describe — but the *shape over time* is nothing like a decay curve.

A single FAR mean averages 28 and 76 into a number that describes neither year. That is why nothing
here is reported as one.

An honest reading of the 2018 anomaly is that it is a property of that year's data and not of
drift: something about 2018 makes it hard for a model trained on 2013–14, and whatever it is does not
persist into 2019. It is not explained here, and it is worth explaining — a model-agnostic effect this
large is more interesting than the margins between the models.

## Variance grows with difficulty

FPTM's standard deviation runs from 0.14 on IID to 7.41 on 2021 — a fifty-fold spread across the same
model and configuration, differing only in which year it is scored on. Any single-seed result on the
FAR years would have been close to meaningless, and a three-seed run (kept in `results.txt`) already
differed from the ten-seed run by up to a point.

Gradient boosting is quoted from a single run because it is **deterministic at this configuration** —
five seeds give bit-identical predictions on all seven evaluation sets, standard deviation exactly
0.0000. So the single number is the mean, and the comparison above is a one-sample test of the Tsetlin
mean against a fixed constant in place of a shortcut. See `../b0-baseline/` for the check that
establishes this instead of assuming it in either direction.

## Interpretability diagnostics, measured on the same models

- **`LF` / median included literals = 14.5%** (`LF` = 10, median 69 literals per clause). For
  reference, a ratio of 21% is where clauses have been found to decompose into a readable core plus a
  tolerance tail, and 1.7% is where extraction returns something precise and useless.
- **83.8% of literals are negated** — the clauses are blacklist-dominated — but no clause is purely
  so; a typical 69-literal clause carries around eleven "feature present" literals.
- Clauses run **2.2× over `L` = 64**, which independently confirms that `L` gates clause *growth*
  instead of capping clause size.

None of this is an interpretability claim. A claim requires the dataset control — whether a χ² or
frequency ranking of the raw features would produce the same feature list — and that has not been run.

## Scope

One dataset, one feature variant, one clause budget, binary detection, 2013–2022. The cross-dataset
check on APIGraph has not been run, and until it has, this is a LAMDA result, not a result about
Fuzzy-Pattern Tsetlin machines. Three of four comparable positives in the sibling algorithm project
failed on their second dataset, so that qualifier is not boilerplate.

## Reproduce

```
julia --project=. -t 16 research/b1-gate/run.jl 10
```

`results_10seed.txt` is the ten-seed run, `results.txt` the earlier three-seed one. Threads affect `predict` only, and `predict` is bit-identical across thread counts.
Training is serial for two reasons: parallelising across classes caps at the class count, which is two here, and parallelising across clauses is slower than serial at this clause budget on this width.
