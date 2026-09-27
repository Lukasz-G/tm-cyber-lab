# Experiment index

Every experiment, its verdict, and whether it earns its place. Written so that pruning is a decision
about which rows to delete rather than an archaeology exercise.

Conventions and the required script header are in [README.md](README.md). Known-unfair comparisons and
other outstanding gaps are in [../docs/open-issues.md](../docs/open-issues.md).

## Calibration — B0. Not findings.

| | verdict | keep? |
|---|---|---|
| [b0-composition](b0-composition/) | **PASS exactly.** All five published quantities to the sample. Recovered the family-counting convention: `singleton:*` are AVClass2 placeholders, and 471 named families hold one sample. | keep — the convention is not in the paper and the next user needs it |
| [b0-boundary](b0-boundary/) | **PASS**, 12 years bit-exact against digests computed from Parquet by separate code. Surfaced irregular month coverage (2017-07 has 2 samples). | keep — the coverage finding bears on any month-over-month series |
| [b0-baseline](b0-baseline/) | IID reproduces to **two decimals** (97.49). Published NEAR recovered as a **mean of per-year F1**, not pooled rows (59.09 vs 59.48). FAR **not reconstructible**. LightGBM is deterministic here (5 seeds, bit-identical). | keep — the NEAR definition is load-bearing for every comparison |
| [b0-sanity](b0-sanity/) | First flat FPTM on LAMDA, IID F1 95.11. Both pre-registered interpretability predictions **wrong**: tolerance ratio 14.5% not ~2%, and no pure blacklists. | merge into b1-gate — superseded by it |

## The method result

| | verdict | keep? |
|---|---|---|
| [b2-generality](b2-generality/) | **Count-symmetry + additivity is the whole requirement.** Exact closed-form Shapley verified for fuzzy clauses, conjunctions (⇒ classical TM as a corollary), m-of-n thresholds, and an *arbitrary* function of the miss count. Ordered rule lists provably outside the class (error 1.1e−1 vs 2e−13). | **core.** This is the paper's spine |

Implementation [`julia/shapley.jl`](../julia/shapley.jl), brute-force check
[`test/shapley_closed_form.jl`](../test/shapley_closed_form.jl) — 144 configurations, max disagreement
8.9e−16. Pre-registration in [`docs/clause-attribution.md`](../docs/clause-attribution.md).

## The attribution result

| | verdict | keep? |
|---|---|---|
| [b0-noise-floor](b0-noise-floor/) | **LAMDA's reported explanation drift sits at its own estimator's noise floor.** Same month, two runs: Jaccard 0.926. Consecutive months: 0.958. Same model, explainer re-run only: 0.926 — so it is coalition sampling at `nsamples=100` against 4,561 features, not monthly refitting. | **core** |

## Detection — B1 and the cross-checks

| | verdict | keep? |
|---|---|---|
| [b1-gate](b1-gate/) | **PASS.** LAMDA, 10 seeds, 20 clauses/class: IID −1.82 vs LightGBM, **+4 to +12 on four of five drifted years** at 5–9 standard errors. | keep, but it must be read beside the cross-checks |
| [b5-apigraph](b5-apigraph/) | **DOES NOT REPLICATE.** Ahead on 1 of 5 later years; behind XGBoost on all six. The `s` control ruled itself out. | **core** — this is the honest centre of the detection story |
| [b5-androzoo](b5-androzoo/) | **NARROWS.** Fails the ±5 criterion on 2020 by 0.45. Dense regime (39% vs 1.75%). Found that **clause sign composition tracks input density** — 83.8% negated at 3% density vs 64.6% at 39%. | keep — the density/sign finding is mechanistic and independent of the verdict |

**Net:** superior drift robustness was a LAMDA result. What survives all three datasets is an
*efficiency* claim — within roughly five F1 of the best of two boosters at a few hundred literals per
class — not a performance one.

## Why is FPTM dominated? Nine hypotheses, all rejected.

These six directories are one investigation. Prune aggressively; the conclusion is one paragraph.

| | hypothesis | verdict |
|---|---|---|
| [b5-diagnosis](b5-diagnosis/) | clause budget binding; recall vs precision deficit | **no** — 10× clauses = +0.75 F1; at matched precision FPTM gives up 12.6 recall to XGBoost |
| [b5-threshold](b5-threshold/) | bad default operating point | **partly** — default costs **8.6 F1** — but dominated 6/6 anyway |
| [b5-clause-length](b5-clause-length/) | clauses too long/short; `T` mis-scaled; short×many | **no** — 10-cell (L, clauses, T) grid, none escapes; best cell is the original gate config |
| [b5-imbalance](b5-imbalance/) | class imbalance, untested regime for this whole line of work | **no** — balancing slides the operating point, not the frontier. Side result: **6k rows ≡ 30k** |
| [b5-resolution](b5-resolution/) | margin too coarse to rank | **no** — top bucket size 1, purity 100%, 96% precision @ 10% recall |
| [b5-diversity](b5-diversity/) | extra clauses are duplicates; feedback rule saturated | **no** — zero pairs >0.9 Jaccard; diversity *rises* with budget |

**Standing conclusion.** At every configuration tested, a 20–200 clause FPTM lies inside the
gradient-boosting precision/recall frontier on APIGraph. Nine explanations were eliminated, so the
deficit is the hypothesis class rather than a setting: a small number of large tolerant conjunctions
ranks worse in the mid-range than an additive ensemble of thousands of shallow splits, and no budget,
balance, length or threshold recovers it. The clauses are structurally diverse and predictively
redundant.

**The operating point that survives:** 96% precision at 10% recall — low alert volume, high purity. A
different product from a recall-tuned booster, and a defensible one.

## Not yet run

- Interpretability dataset control — χ² or frequency ranking of raw features vs the clause literals.
  **Blocks every readability claim.** The equivalent control retracted the sibling project's result.
- Booster probability sweep, for frontier-against-frontier rather than frontier-against-point.
- Label drift vs data drift, from LAMDA's per-year VirusTotal verdict changes (`vt_detections.csv`,
  already downloaded).
- Model footprint and labels-per-unit-detection.
