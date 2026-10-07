# b2-drift — exact clause-level explanation drift against the sampled estimator's target

The measurement this project exists to make. `b0-noise-floor/` showed that LAMDA's reported explanation drift sits at its own estimator's noise floor.
It showed that by re-running the estimator and watching the answer move. That establishes the published number cannot see a signal, without saying what the signal is. This measures the signal, exactly.

Protocol fixed in advance in [`docs/clause-attribution.md`](../../docs/clause-attribution.md),
pre-registered 2026-09-19 before any LAMDA model existed.

## Answer: an explanation far more stable than reported, with most churn from refitting

LAMDA, 88 months 2013-06 to 2022-12, flat FPTM at the B1 gate configuration, top-100 features.
Jaccard `1 − |∩|/|∪|`, so **lower means more stable**.

| | arm | Jaccard | sd | n |
|---|---|---|---|---|
| **2** | fixed model, data moves — **pure data drift** | **0.294** | 0.159 | 87 |
| 5 | fixed model, whole test split instead of 100 rows | 0.274 | 0.163 | 87 |
| 1 | same month, two seeds — model variation only | 0.547 | 0.139 | 88 |
| 3 | refitted per month, as the reference does | 0.661 | 0.124 | 174 |
| — | *LAMDA's protocol re-run here, MLP + KernelExplainer at `nsamples=100`; they publish ≈0.9* | *0.958* | | |
| — | *its own noise floor, from `b0-noise-floor/`* | *0.926* | | |

At k=1000: arm 2 = 0.125, arm 5 = 0.122, arm 1 = 0.352, arm 3 = 0.451.

Four readings, in order:

**Pure data drift is 0.294 against the 0.958 their protocol gives here.** The feature set a fixed model relies on turns
over about 30% month to month, not 96%. The published figure was not measuring the malware.

**Monthly refitting contributes more than the data does.** Arm 3 − arm 2 = **+0.367** at k=100 and
+0.326 at k=1000. The released script trains a fresh MLP inside its per-month function — a detail
that is in the code and not in the paper — so the published number folds refitting into "drift", and
that component is the larger one.

**Model variation exceeds data drift at this clause budget.** Arm 1 (0.547) is above arm 2 (0.294):
two seeds on the *same month* disagree more than one model does across *different* months. Stated
plainly because it bounds what the method can claim — at 20 clauses per class, which clauses get
learned is less stable than the data is. Arm 3 (0.661) sits above arm 1, and the gap between them,
0.114, is the genuine data contribution to refit churn. So of arm 3's 0.661, most is seed variance.

**The 100-row sample is not the cause.** Arm 5 explains the month's whole test split — up to 7,161
rows instead of 100 — and moves the answer by 0.02. Whatever residual churn arm 2 shows is data, not
the row sample.

## Arm 4 — beyond the reach of published work

Sampled `KernelExplainer` against the closed form, **on the identical model**, with the data and the
attribution target held fixed. This requires a model whose exact Shapley values are computable, which
is why nobody has run it. 2016-06, top-100.

| comparison | Jaccard | Kendall |
|---|---|---|
| **a.** sampled `nsamples=100` vs **exact** | **0.851** | 0.614 |
| **b.** sampled `nsamples=100` vs *itself*, different seed | **0.889** | 0.681 |
| **c.** sampled `nsamples=1000` vs **exact** | **0.496** | 0.300 |

**(a) ≈ (b): the error is variance, not bias.** At the published budget the estimator lands as far
from the truth as it lands from itself. It is not systematically wrong about which features matter;
it is barely determined at all.

**(c) < (a): the budget is the cause.** Ten times the coalitions nearly halves the distance to exact.
`KernelExplainer`'s own default here would be `2·4561 + 2048 = 11170`, which is 112× the budget used.

Every call at `nsamples=100` emitted the library's own warning that the sample count is too small to
determine a regular solution.

### Sparsity of the exact attribution

Only **496 of 4,561** features have nonzero exact Shapley value on this model. A top-1000 set drawn
from it is therefore more than half arbitrary — ties among zeros. The k=1000 arm-4 rows in
`results-arm4.txt` are reported for completeness and should not be interpreted; k=100 is inside the
support and is the comparison that means anything. This does not affect arms 1–3 and 5, which compare
two exact rankings against each other under the same sparsity.

Sparsity is itself a consequence of the closed form: a clause's vote depends only on how many of its
literals flip, so every feature outside the clauses' literal sets has value exactly zero. It is not a
threshold or a regularisation choice.

## Claims licensed and refused

- That the published explanation-drift figure is **dominated by estimator variance and monthly
  refitting**, with a measured decomposition and not an inference from re-running.
- That exact attribution on a nonlinear malware model is **cheap**: the whole 88-month series, three
  attribution passes per month, runs in about 25 minutes on 16 threads.
- It does **not** license any claim that the clauses are human-readable. The tolerance ratio is
  measured below and is favourable, but the dataset control — whether a χ² or frequency ranking of the
  raw features produces the same list — has not been run, and the equivalent control retracted the
  sibling project's one interpretability positive. See `docs/open-issues.md` entry 5.
- Arms 1–3 and 5 are one dataset. The cross-dataset picture for *detection* did not replicate
  (`b5-apigraph/`), so a drift-forensics result on LAMDA alone is provisional in the same way.

## Pre-registered falsification conditions against the outcome

Quoted from the pre-registration so they cannot drift:

- *"If arm 1's noise floor is near zero and arms 2–3 still show Jaccard ≈ 0.9, the churn is real."*
  **Did not fire** — arms 2 and 3 are 0.294 and 0.661, not ≈ 0.9.
- *"If arm 4 shows sampled and exact attributions agreeing closely, then `nsamples=100` was adequate
  despite appearances, and the estimator-variance hypothesis is wrong."* **Did not fire** — 0.851.
- *"If the TM's own clause-level Jaccard is ≈ 0.9 with a near-zero noise floor, explanation drift is a
  property of the problem, not of the estimator."* **Did not fire.**
- The run's own invalidation condition — arm 1 landing at the same level as arms 2 and 3, meaning seed
  variation swamps everything — **did not fire**, but came closest: arm 1 sits between them and above
  arm 2, which is why that is reported above and not buried.

## Interpretability diagnostic, recorded because any rule quoted later needs it

Fixed model, 2013–14: median **69 literals per clause**, `LF` = 10, so tolerance is **14.5%** of the
clause. Pre-registered prediction 2 expected ~2% — IMDb's regime, where extracted rules are precise
and useless — and is **wrong**, for the second time after `b0-sanity/`. 14.5% is nearer MNIST's 21%,
which is the regime that decomposed into genuine rules. That makes the dataset control worth running
in place of a formality.

## Configuration

| | |
|---|---|
| model | flat FPTM, `clauses_per_class` 20, `T` 10, `S` 100, `L` 64, `LF` 10, 30 epochs |
| ceiling policy | `LiteralCapped()` — the FPTM paper's |
| training parallel mode | `:none` |
| attribution | threaded across explained rows, 16 threads, bit-identical at any thread count |
| Tsetlin stack | tm-lab **43dba5f** |
| window | 2013-06 to 2022-12, 88 months with ≥100 train and ≥100 test rows |
| budget | 100 background from the month's train portion, 100 explained from its test portion |
| importance | mean over explained rows of `abs(phi)`, matching the reference's aggregation |

2013 months before June, and 2015 entirely, are absent: LAMDA has no 2015, and months with fewer than
100 rows in either portion are skipped. 2023–2025 are excluded because LAMDA's 2024 and 2025 malware
counts are 794 and 23 against ~45,000 benign per year, which is antivirus label lag, not drift.

## A bug in shared code, worth knowing about

`julia/shapley.jl` originally computed its binomial ratio in exact rational arithmetic with
`binomial(big(n), big(k))`. Correct, and it is what the closed form was first verified against.
But it allocates a BigInt per term, and on a real attribution pass over LAMDA it produced **4.6 billion
allocations and crashed the garbage collector inside a threaded region** (`EXCEPTION_ACCESS_VIOLATION`
in `ijl_gc_collect`, reached from `BigFloat`). It now routes every binomial through a log-factorial
table in Float64. The summed terms are hypergeometric probabilities, all positive, so there is no
cancellation.

The brute-force check was re-run against the new implementation, not inherited:
**max |closed form − brute force| = 1.78e-15** over 144 configurations, against the 1e-9 the test
enforces. The rewrite made the 88-month series **about 25× faster** — one month went from 363 s to
7–41 s depending on its test-split size.

## Reproduce

```
julia --project=. -t 16 research/b2-drift/run.jl                 # arms 1, 2, 3, 5 -> results.txt
julia --project=. -t 16 research/b2-drift/export_arm4.jl 2016 6  # -> data/b2-arm4/
python research/b2-drift/arm4.py --big 1000                      # arm 4 -> results-arm4.txt
```

`run.jl` takes optional `[nmonths] [nseeds]` for a short pass. `arm4.py` asserts that its own re-implementation of the clause vote reproduces the exported Julia scores exactly before it explains anything.
A comparison against a mis-scored model would measure the port. That check is therefore a hard failure and not a warning. It passed at max difference **0**.
