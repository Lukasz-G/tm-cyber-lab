# Experiment index

Every experiment, its verdict, and whether it earns its place. Written so that pruning is a decision
about which rows to delete and not an archaeology exercise.

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
| [ripper-exact](ripper-exact/) | **The closed form verified on a learner we did not write.** RIPPER-induced rules, weighted into an additive ensemble, checked against brute force over all 2^13 coalitions: **max error 1.29e-14**, and the Python side is an independent reimplementation so it tests the derivation not the port. 0.7s at full width. **Arm 4 replicates**: sampled vs exact 0.765, sampled vs itself 0.765 — variance-not-bias is not a TM fact. Raw disjunctive RIPPER stays outside the class. | **core** — it is what makes this a rule-ensemble paper in place of a TM paper |
| [lf-attribution](lf-attribution/) | **Tolerance is representational, and it saturates.** From the classical conjunction (LF=1) to LF=5, attribution support grows 260→498 features and the top-10 mass halves (54%→25%); beyond LF≈5 an eightfold further increase moves nothing. Clause size is flat throughout. Both T policies agree, and coincide to the digit at LF=10. Matches the sibling finding that a fuzzy vote buys ~3x, not LFx. | keep — the one question a classical TM cannot be asked |
| [b2-generality](b2-generality/) | **Count-symmetry + additivity is the whole requirement.** Exact closed-form Shapley verified for fuzzy clauses, conjunctions (⇒ classical TM as a corollary), m-of-n thresholds, and an *arbitrary* function of the miss count. Ordered rule lists provably outside the class (error 1.1e−1 vs 2e−13). | **core.** This is the paper's spine |

Implementation [`julia/shapley.jl`](../julia/shapley.jl), brute-force check
[`test/shapley_closed_form.jl`](../test/shapley_closed_form.jl) — 144 configurations, max disagreement
8.9e−16. Pre-registration in [`docs/clause-attribution.md`](../docs/clause-attribution.md).

## The attribution result

| | verdict | keep? |
|---|---|---|
| [b0-noise-floor](b0-noise-floor/) | **LAMDA's reported explanation drift sits at its own estimator's noise floor.** Same month, two runs: Jaccard 0.926. Consecutive months: 0.958. Same model, explainer re-run only: 0.926 — so it is coalition sampling at `nsamples=100` against 4,561 features, not monthly refitting. | **core** |
| [b2-drift](b2-drift/) | **The measurement the project exists to make, all five pre-registered arms, 88 months.** Pure data drift is **0.294** against a reported 0.958. Monthly refitting contributes **more than the data** (arm3 − arm2 = +0.367). Arm 4 — impossible for published work, since it needs exact values — shows the sampled estimator lands as far from the truth (0.851) as from *itself* (0.889), so the error is **variance not bias**, and 10× the budget nearly halves it (0.496), so **the budget is the cause**. | **core.** With b2-generality this is the paper |

Two honest limits are recorded in that README and not here.
Arm 1 (two seeds, same month, **0.547**) sits *above* arm 2, so at 20 clauses per class model variation exceeds month-to-month data drift.
And only **496 of 4,561** features have nonzero exact attribution, so its k=1000 rows are mostly ties among zeros and must not be read.

## Interpretability — the control that gates every readability claim

| | verdict | keep? |
|---|---|---|
| [interp-residual-retrain](interp-residual-retrain/) | **The pruning survives retraining, but splits by period.** Every arm retrained, 5 seeds, all 8 periods, features ZEROED not dropped so width / S / s and the dead-channel perturbation are identical across arms. Over **2019-2022 the gain is +5.28 F1** over a size-matched control (+3.96 over the unpruned model); over **2016-2018 it is -1.12** and hurts 2017 by 3.04. Four contiguous wins, three contiguous failures - so "5 of 7" satisfies the pre-registered count while hiding the structure. **Not an unconditional recommendation.** | **core** - it is the honest form of the pruning result |
| [residual-density-control](residual-density-control/) | **The pruning gain is identity, not density, and the better control makes it bigger.** The published control was matched on count only; zeroing is not sign-neutral and sign is set by presence rate, where the residual sits at 11.1% against a uniform draw's 2.6%. Matched feature-by-feature on presence rate, the 2019-2022 advantage goes **+5.28 -> +6.07**, and arms 1-3 reproduce [interp-residual-retrain](interp-residual-retrain/) to the decimal. Match is partial (60% of the density gap), so the verdict is the direction. The regime split **deepens** (-1.12 -> -2.05). | **core** - it is what lets the drift-forensics claim stand |
| [interp-residual](interp-residual/) | **The 38 features frequency misses are the model's overfitting.** Ablating the 62 overlap features destroys the classifier (−88.85 F1 on IID), so what carries the model is exactly what a frequency count finds. The residual is real — 18× random on IID — but **removing it IMPROVES drifted-year F1, by 10.32 on 2021**. Identified from training data alone, so it is a candidate intervention. Recasts attribution as drift forensics and not readability. | **core**, with caveats — needs the full year sweep and a retrain-without arm |
| [groundtruth-overlap](groundtruth-overlap/) | **The attributed features are indicators, and this is the one semantic advantage of the model over the corpus.** APIGraph, whose vocabulary is released: the attributed top-20 is **85.0% indicator-class against a 23.0% base rate**, with app-identity features down from 63.1% to **1.7%**. At k=100 attribution beats a frequency count by **+12.7pp** indicator and **-17.0pp** identity, which set-overlap could not see because the two rankings agree on the set and differ on the ORDER. Named top-10 is a coherent premium-SMS profile. **LAMDA ships `feat_0...feat_4560`, so this is impossible on the primary corpus.** | **core** - measurement 5 of 5, and it took the longest to run |
| [interp-dataset-control](interp-dataset-control/) | **RETRACTS every readability claim, by the criterion fixed before the run.** A plain document-frequency difference — no model at all — recovers **62 of the model's top-100** features; χ² recovers 54; chance gives 4. Also **confirms pre-registered prediction 1** in form but not in consequence: ~80% of included literals are *negated*, which reads as a blacklist until [blacklist-anatomy](blacklist-anatomy/) shows a random literal at this density is 94.8% negated. The reading that the clauses supply no indicator of compromise was withdrawn there. Blakely & Granmo's inclusion frequency overlaps exact Shapley by only 49/100, so it is not a proxy for it at this width. | **core** — a negative, and it decides what the paper may say |

## Label drift

| | verdict | keep? |
|---|---|---|
| [prune-vs-fragility](prune-vs-fragility/) | **Refutes, in sign, the obvious explanation for the pruning regime split.** Predicted a negative correlation between the pruning gain and label fragility; measured **+0.253 (p=0.030, n=70 months)** — pruning helps roughly *twice as much* where labels are *least* reliable. Month-size control clears (−0.147, p=0.22). The manuscript paragraph was deleted, not softened, as its own todo required. | keep — a clean refutation, and the pre-registration worked |
| [b3-label-drift](b3-label-drift/) | **LAMDA's labels carry drift of their own.** Share of malware a threshold of 10 would relabel ranges **40%→80%** by year; in 2017–18 over half of malware sits at 4–6 detections. Largely separable from feature drift (rho **−0.344**, p=0.0015, n=82) with the benign-only control holding at +0.748. The year-resolution analysis gave the **opposite sign** and its pre-registered rule was under-specified — it tested a correlation without requiring significance. | keep — the label-fragility series has no published peer; the sign flip is a methodological caution |

## Deployment cost

| | verdict | keep? |
|---|---|---|
| [b4-footprint-labels](b4-footprint-labels/) | **Footprint: 44.5 KB inference masks vs 144 MB for LightGBM (3,308×)** — but an *updatable* model carries automaton state too, 356 KB, 8× more, so the small-footprint and online-updating claims are not about the same artefact. **Label efficiency: the continual-learning claim fails.** Continual ≈ retrain-from-scratch within 1 F1 in all 20 cells. Two bigger findings fall out: **250 fresh labels beat 150,090 stale ones by 59 F1**, and **keeping the old data is worse than discarding it at every cell**. | keep — the footprint distinction and the recency-beats-volume result |

## Detection — B1 and the cross-checks

| | verdict | keep? |
|---|---|---|
| [b1-gate](b1-gate/) | **PASS.** LAMDA, 10 seeds, 20 clauses/class: IID −1.82 vs LightGBM, **+4 to +12 on four of five drifted years** at 5–9 standard errors. | keep, but it must be read beside the cross-checks |
| [b5-apigraph](b5-apigraph/) | **DOES NOT REPLICATE.** Ahead on 1 of 5 later years; behind XGBoost on all six. The `s` control ruled itself out. | **core** — this is the honest centre of the detection story |
| [threshold-bias](threshold-bias/) | **Elevates the threshold finding to a field-level result.** An oracle threshold inflates F1 by **0.03 in distribution** and by **7-19 points on drifted periods**, peaking at **33.38**. It hits all three models, **unequally** (7.07 vs 18.97 on the same corpus), so it can reorder a comparison. Both common defaults are wrong in opposite directions: the oracle inflates, the argmax deflates by 8.6. Rate matching closes it to 0.60-3.53 for everyone. | **core** — the paper's methodological contribution |
| [calibration-check](calibration-check/) | **Withdraws the threshold-transfer claim.** Platt and isotonic are monotone, so calibration cannot change a ranking — raw/Platt/isotonic agree **to the decimal** across 6 model-corpus combinations, which answers that objection structurally. But **rate matching** repairs the boosters (LightGBM on LAMDA 18.97 → **1.78** oracle gap) and they lead on mean F1 on both corpora. The advantage was the threshold rule, not the model. | **core** — it is the reason no detection advantage is claimed |
| [b1-threshold-transfer](b1-threshold-transfer/) | **The APIGraph threshold-transfer result REPLICATES on LAMDA**, at a 48% base rate instead of 10%, with a larger effect: +3.43 AP against us; honest thresholding costs FPTM **6.19 F1** against **16.60 / 15.31** for the boosters; FPTM leads the deployable-threshold F1 on **7 of 8** periods, losing only IID. | **core** — the only claim in the project that holds on two corpora |
| [b5-fairness](b5-fairness/) | **Fixes all three recorded unfairnesses at once** — both sides swept, threshold chosen on held-out data, dominance on every seed. Boosters keep +3.77 AP on 6/6; FPTM wins the deployable-threshold F1 on 5/6. Caught a split bug that would have selected a threshold on a slice containing **0% malware**. | **core** — it is the honest basis for every comparison |
| [b5-androzoo](b5-androzoo/) | **NARROWS.** Fails the ±5 criterion on 2020 by 0.45. Dense regime (39% vs 1.75%). Found that **clause sign composition tracks input density** — 83.8% negated at 3% density vs 64.6% at 39%. | keep — the density/sign finding is mechanistic and independent of the verdict |

**Net:** superior drift robustness was a LAMDA result. What survives all three datasets is an
*efficiency* claim — within roughly five F1 of the best of two boosters at a few hundred literals per
class — not a performance one.

## The dominance of the boosters: nine hypotheses, all rejected

These six directories are one investigation. Prune aggressively; the conclusion is one paragraph.

| | hypothesis | verdict |
|---|---|---|
| [b5-diagnosis](b5-diagnosis/) | clause budget binding; recall vs precision deficit | **no** — 10× clauses = +0.75 F1; at matched precision FPTM gives up 12.6 recall to XGBoost |
| [b5-threshold](b5-threshold/) | bad default operating point | **partly** — default costs **8.6 F1** — but dominated 6/6 anyway |
| [b5-clause-length](b5-clause-length/) | clauses too long/short; `T` mis-scaled; short×many | **no** — 10-cell (L, clauses, T) grid, none escapes; best cell is the original gate config |
| [b5-imbalance](b5-imbalance/) | class imbalance, untested regime for this whole line of work | **no** — balancing slides the operating point, not the frontier. Side result: **6k rows ≡ 30k** |
| [b5-resolution](b5-resolution/) | margin too coarse to rank | **no** — top bucket size 1, purity 100%, 96% precision @ 10% recall |
| [b5-diversity](b5-diversity/) | extra clauses are duplicates; feedback rule saturated | **no** — zero pairs >0.9 Jaccard; diversity *rises* with budget |

**Standing conclusion, revised again 2026-09-30 by [calibration-check](calibration-check/). There is now
no detection advantage claimed.**

- **Threshold-free, the boosters' frontier is better** — +3.77 average precision on APIGraph and +3.43 on
  LAMDA, ahead on every year, with both sides swept. The original conclusion survives on this metric.
- **The operating-point advantage does not survive a better threshold rule.** Under a max-F1-on-validation
  rule FPTM transfers far better (giving up 1.73 and 6.19 F1 against the boosters' 6–9 and 15–17). But
  **matching the predicted positive rate** instead — deployable, uses only unlabelled test features —
  repairs the boosters almost entirely (LightGBM on LAMDA: 18.97 → **1.78**) and they then lead on mean F1
  on **both** corpora. The advantage was a property of the threshold rule, not of the model.
- **Calibration proper is a no-op**, and provably so: Platt and isotonic are monotone, so they cannot
  change a ranking. Raw, Platt and isotonic arms agree *to the decimal* in all six model-corpus
  combinations.

What is left to say about detection: FPTM is competitive at 20 clauses per class, lies inside the
gradient-boosting frontier, and its margin is **more robust to a naive threshold rule** — a real,
twice-measured difference that is not a performance advantage. The deployable recommendation is to match
the predicted positive rate, which is advice about thresholding and improves every model including ours.
Every F1 elsewhere in this repo is an oracle value; the correction is measured.

## Not yet run

- Pre-registered **prediction 5** — data drift and label drift differently timed, from LAMDA's
  Appendix F strengthened/weakened/flipped verdict counts. **Blocked on the dataset authors:** that
  needs a detection count at two points in time per sample and the release carries one snapshot per
  `sha256`. An earlier version of this list claimed a `vt_detections.csv` was already downloaded; no
  such file exists in the release. See [b3-label-drift](b3-label-drift/) for what is runnable instead.
- FAR reconstruction on LAMDA, also blocked on the authors.
- **An explanation for the pruning regime split.** [interp-residual-retrain](interp-residual-retrain/)
  gains ~5 F1 on 2019–2022 and loses ~1 on 2016–2018, and nothing accounts for it. The label-fragility
  hypothesis was tested and the correlation came out with **the opposite sign**
  ([prune-vs-fragility](prune-vs-fragility/)), so pruning stays a candidate intervention with an
  uncharacterised precondition.
- ~~Bibliography metadata~~ — **done 2026-09-30.** Every entry verified against the publisher's record,
  and every arXiv entry additionally against the first page of the PDF. Three carried substantive errors,
  not merely gaps: `lamda2025` had the lab name in place of seven authors, the "Graph Tsetlin Machine"
  entry was titled after the model, not the paper, and the TESSERACT follow-up was retitled in its
  version 2. The 15 remaining bibtex warnings are structural — arXiv preprints have no volume or pages,
  and NDSS is unpaginated. PDFs are in the gitignored `papers/`.
