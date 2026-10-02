# Clause attribution for explanation-drift measurement

**Pre-registered 2026-09-19, before any LAMDA model exists.** Several attribution definitions are
defensible and they are not equivalent, so choosing one after seeing which produces the nicer
Jaccard curve would be choosing the headline. This file fixes the choice, the comparison protocol
and the falsification conditions in advance.

## 1. The quantity, and its comparator

LAMDA reports **explanation drift**: Jaccard and Kendall distances between consecutive months over
the top-ranked SHAP features. On LAMDA the Jaccard sits close to 0.9 for top-100 features — the
important-feature set almost entirely turns over month to month — while classification performance
is comparatively stable. On APIGraph the same measurement shows a gradual downward trend instead.

A Jaccard of 0.9 is consistent with real explanation drift *and* with the attribution estimator
being noisy. The question this work answers is which.

## 2. The reference measurement's procedure

Read from the released source (`code/section_4_concept_drift_analysis/4_5_shap_explanation_monthly_lamda.py`
and `4_5_shap_explanation_graphs.py`), not from the paper text, because three details matter and
only one of them is in the paper.

| | |
|---|---|
| model | an MLP, trained inside the per-month function |
| **retraining** | **a fresh model per (year, month, run)** — `train_chen_mlp(X_train, y_train)` is called inside `run_single_shap_experiment`, on that month's own `{year}-{month}_X_train.npz` |
| explainer | `shap.KernelExplainer` |
| background | `X_train[:100]` |
| explained | `X_test[:100]` |
| **coalition budget** | **`nsamples=100`** |
| repeats | `runs_per_year = 3` — three independent runs per month |
| pairing | consecutive months, within a run index; then mean ± SEM across the three runs |

Two observations follow, and they are the substance of this work.

**First, the protocol conflates three sources of change.** Because the model is re-fitted every
month, a change in the top-100 set can come from the data distribution moving, from a different
model being fitted to it, or from the explainer's sampling. "Explanation drift" as measured is the
sum of all three.

**Second, the coalition budget is about 1% of SHAP's own default.** `KernelExplainer`'s default is
`2·M + 2048`, which for M = 4,561 features is 11,170 coalitions. The published measurement uses
100 — fewer coalitions than there are features by a factor of 45. A top-100 ranking drawn from that
is expected to be substantially noise, and the paper reports no error bars on the attribution
itself.

**The control they had and did not run.** Three runs exist per month. Their aggregation computes
Jaccard *between consecutive months within a run*, then averages over runs. It never computes
Jaccard *between runs of the same month*. That quantity is the noise floor: if two independent
runs on identical data already disagree at Jaccard ≈ 0.9, then month-to-month churn of 0.9 measures
the estimator and not the malware. This is a cheap, decisive experiment and it is the first thing
to run.

## 3. The definition adopted

**Exact Shapley values, interventional value function, computed in closed form.**

For a coalition `S`, `v(S) = f(z_S)` where `z_S` takes the explained instance's value on features
in `S` and a background instance's value elsewhere; `f` is the class score, `positive − negative`
summed clause votes. Against a background *set*, `v` is the mean over backgrounds; Shapley is
linear in `v`, so the result is still exact.

This is the same quantity `KernelExplainer` estimates. That is the point: the comparison is
like-for-like, and the only difference is that one side samples and the other does not.

**It has a closed form.** A clause's vote is `max(0, ceiling − misses)`, and `misses` depends only
on *how many* of the clause's literals are flipped between instance and background, never on which.
So every feature outside that set has Shapley value exactly zero, the rest fall into two
equivalence classes, and the value of each collapses to a sum of `O((g+d)²)` terms with no
dependence on the feature count. Derivation and implementation:
[`julia/shapley.jl`](../julia/shapley.jl).

**Verified, not asserted.** [`test/shapley_closed_form.jl`](../test/shapley_closed_form.jl)
compares against brute-force enumeration of all 2¹⁴ coalitions on trained models, across 144
combinations of seed, `LF`, ceiling policy, instance, background and class. Maximum disagreement
8.9e-16. Efficiency is checked separately, because an early version satisfied efficiency while
splitting the total wrongly between features.

**Cost, measured at LAMDA's width (4,561 features, 3% density, single thread):** 6.7 s for one
month at the gate configuration of 20 clauses per class, using the same 100 background × 100
explained budget as the reference. The full ~135-month series is about 15 minutes on one core, and
scoring threads bit-identically across examples. At 200 clauses per class it is 28 s per month.
**The exact computation is cheaper than the sampled approximation it replaces**, which is worth
stating plainly instead of treating as a workaround for lacking a gradient.

### Rejected alternatives, and why

- **Firing-weighted clause membership** — sum over clauses of polarity × membership × the clause's
  firing rate on the month. Cheap and exact, but it is not what SHAP estimates, so it answers a
  different question and cannot be compared like-for-like.
- **Satisfied-mask credit** — credit a literal only when included *and* satisfied in an actual
  evaluation. Uses what FPTM uniquely exposes, and meaningful because most nonzero clause votes are
  partial matches. Kept as a secondary, reported alongside; not the primary, for the same
  comparability reason.
- **Leave-one-feature-out.** The obvious exact choice, and wrong here. LOFO is the single
  all-others-present marginal, whereas Shapley averages over all coalition orderings, and the two
  differ most precisely on **redundant** literals — which fuzzy clauses accumulate by design, since
  tolerance is what lets a clause degrade gracefully on noisy input. LOFO would systematically
  under-credit the literals FPTM has most of, and would make the ranking look more stable than it
  is. It was the plan until the closed form turned out to exist.
- **Automaton-state weighting** — ranking literals by TA confidence. Not usable: automata pile up
  exactly on the include threshold, so the confidence signal barely exists in a trained model.

## 3a. Prior art, checked 2026-09-19

Three pieces of work sit close to this and none of them makes the claim above.

**Chow et al., *Drift Forensics of Malware Classifiers* (AISec 2023)** — named in the project's
founding document as the closest prior work. It is closer in spirit than in substance. They pick
"points of interest" (months where an oracle classifier beats the deployed one), attribute the drop
to specific malware families, and then compare the top-k features of the deployed versus the oracle
classifier on the missed samples. The attribution is Gradient ⊙ Input on a **linear SVM**, which
they note equals Integrated Gradients in the linear case — so it is exact, but exact only because
the model is linear. There is no Jaccard, no Kendall, and no month-over-month set-churn measurement:
the comparison is model-versus-model at a chosen month, not month-versus-month. **They buy
exactness by giving up nonlinearity. We get exactness on a nonlinear model.** That is the sentence
that positions this work against them.

**Blakely & Granmo, *Closed-Form Expressions for Global and Local Interpretation of Tsetlin
Machines* (arXiv:2007.13885)** — the obvious collision, and it is not one. Their Global Feature
Strength is the **inclusion frequency** of a feature across positive-polarity clauses, normalised
by the clause count: a model-intrinsic heuristic, not a Shapley value. They compare it to SHAP on
Wisconsin Breast Cancer (30 features) and report that the top-10 sets largely agree — a
correspondence, explicitly not an identity. Classical Boolean-output TM, no fuzzy vote, no temporal
analysis. Two things follow. First, the exact-Shapley result above is a different and stronger
claim, and must be stated as *equal to* Shapley, not *like* SHAP. Second, their measure is
**data-independent**, so under a fixed model it shows zero explanation drift by construction; it is
worth reporting as an additional arm precisely because that makes the contrast with a
data-conditional measure legible.

**Kalný, Jureček & Stamp, *Detecting Concept Drift in Evolving Malware Families Using Rule-Based
Classifier Representations* (arXiv:2604.22629)** — the nearest live competitor, and it post-dates
the founding document. Decision-tree rulesets and RIPPER over temporal windows on **EMBER2024**
(Windows PE), family-vs-benign and family-vs-family, with rule-level drift metrics including
"Jaccard cover rules", prediction agreement and feature L1 distance. Different dataset, different
platform, different learner — but the same instinct, that rule representations are the right lens
on drift. It must be cited and distinguished, and it strengthens the choice of RIPPER as the
interpretable-model baseline. It does not address the estimator-variance question, because
decision-tree rules are exact and the question does not arise for them.

**Nobody has measured the noise floor of a sampled attribution in this setting**, and nobody has
done exact month-over-month attribution on a nonlinear malware model. That gap is arms 1 and 4.

## 4. Comparison protocol

Fixed in advance, and matched to the reference wherever matching is possible.

- **Metrics.** Jaccard distance over the top-k feature *sets*, Kendall distance over their
  *ranking*, exactly as the reference computes them: `1 − |∩|/|∪|`, and `(1 − τ)/2`.
- **k.** 100 and 1000, both reported.
- **Pairing.** Consecutive months.
- **Budget.** 100 background and 100 explained rows per month, matching the reference, so that any
  difference is attributable to the estimator, not to sample size. Exactness is free here,
  so a second arm at the full month is also reported — see arm 5.
- **Window.** 2013–2022. Later years are excluded for the reason given in the repository README:
  LAMDA's 2024 and 2025 malware counts are 794 and 23 against roughly 45,000 benign per year, which
  is antivirus label lag and not drift.

### The five arms

Reported together, because the comparison between them *is* the result.

1. **Noise floor.** Two independent runs, same month, same protocol. For the TM this is a re-trained
   model with a different seed. For the reference it is their own two runs. Jaccard between them.
2. **Fixed model, moving data.** One model trained on 2013–14, attributions recomputed per month.
   Isolates data drift.
3. **Refitted per month.** A fresh model per month, as the reference does. Arm 3 minus arm 2 is the
   contribution of model refitting, which the published number folds in silently.
4. **Sampled versus exact, same model.** KernelExplainer at `nsamples=100` against the closed form,
   on the identical TM. This is the direct measurement of estimator variance, with the model, the
   data and the attribution target all held fixed. No published work can run this arm, because it
   requires a model whose exact Shapley values are computable.
5. **Budget sensitivity.** Exact attribution over the full month and not 100 rows, to show how
   much of any residual churn is the 100-row sample in place of the coalition sampling.

Arms 1 and 4 are the contribution. Arms 2 and 3 separate what the published number combines.

## 5. What would falsify this

Stated now so the result cannot be reinterpreted later.

- **If arm 1's noise floor is near zero and arms 2–3 still show Jaccard ≈ 0.9**, the churn is real,
  the published finding stands, and our contribution is the decomposition in arms 2–3 plus the
  first exact measurement of it. That is a smaller result, and it would be reported as one.
- **If arm 4 shows sampled and exact attributions agreeing closely**, then `nsamples=100` was
  adequate despite appearances, and the estimator-variance hypothesis is wrong.
- **If the TM's own clause-level Jaccard is ≈ 0.9 with a near-zero noise floor**, explanation drift
  is a property of the problem and not of the estimator, which is a *stronger* result for the
  drift literature than the one this work set out to test, and must be reported as the headline, not buried.
- **If the TM fails the detection gate**, none of this is reportable as a malware result, because
  attribution drift in a model that does not work is not evidence about malware.

## 6. Reporting requirements

Every number carries: the parallel mode and thread count, the clause-vote ceiling policy, the
pinned Tsetlin-stack commit, the clause budget, and `LF` / included-literals. The first three
identify which model was run; the last is the diagnostic for whether any of it is human-readable,
which is a separate claim from whether the attribution is exact.
