# interp-dataset-control — the model's feature list, without the model

The control that gates every readability claim in this project. The equivalent control in the sibling
algorithm project **retracted** its one interpretability positive: readable words extracted from an IMDb
model turned out to come from the dataset, and a plain frequency ranking reproduced them.

Pre-registered criterion, written into the script header before the run:

> RETRACT every readability claim if chi-squared or frequency recovers **half or more** of the model's
> top-k. SURVIVES if they sit near the random baseline.

## Answer: a fired criterion, and a retracted readability claim

LAMDA 2013–14 train, 150,090 rows, flat FPTM at the B1 gate configuration, 3 seeds. Overlap between
the model's exact-Shapley top-k and each ranking's top-k.

| k | ∩ χ² | ∩ frequency | ∩ random | ∩ inclusion freq. |
|---|---|---|---|---|
| 20 | 10.3 / 20 | **11.7 / 20** | 0.0 / 20 | 8.3 / 20 |
| 100 | 53.7 / 100 | **62.0 / 100** | 4.0 / 100 | 49.3 / 100 |
| 500 | 221.3 / 500 | **265.3 / 500** | 49.7 / 500 | 348.7 / 500 |

**A document-frequency difference — `P(f|malware) − P(f|benign)`, no model involved at all — recovers
62% of the model's top-100.** χ² recovers 54%. Both clear the half threshold at k=20 and k=100, and
frequency clears it at k=500 too. The criterion was fixed in advance and it fires.

The random column is why the others can be read: chance alone gives 4 of 100. So the overlap is
emphatically not an artefact of the comparison — the model's ranking really is, substantially, the
dataset's ranking.

### Precise statement of what survives

About **38%** of the model's top-100 is *not* recovered by frequency and **46%** not by χ². That
residual is real and is what an honest claim can be built on — but the pre-registration is explicit
that it "licenses *the clauses select features a univariate ranking does not* only for the
non-overlapping part, and that part has to be shown, not asserted". It has not been shown. So:

- **Not claimable:** that this model's clauses give an analyst a readable account of what distinguishes
  malware. Most of what they point at, a one-line frequency count also points at.
- **Claimable, and unaffected:** everything in `b2-drift/` and `b2-generality/`. Exactness of the
  attribution is a property of the closed form, not of whether the features are interesting. The drift
  decomposition stands on its own.
- **Open, and cheap to settle later:** whether the ~38% residual carries predictive weight the
  univariate rankings miss. An ablation — drop the residual, keep the overlap, measure F1 — would
  answer it. Not run.

## Prediction 1 confirmed — blacklist-shaped clauses

Pre-registered prediction 1 said the clauses would encode *absence*, not presence, because
Drebin features are sparse so "none of these tokens present" is the cheapest clause to learn. Measured:

| bank | negated literals | total |
|---|---|---|
| positive polarity | **79.2 / 80.0 / 81.5 %** | ~690 |
| negative polarity | **88.6 / 87.4 / 86.5 %** | ~680 |

(three seeds). Roughly four in five included literals require a feature to be **absent**.

This is a security statement, not an aesthetic one. A detector whose clauses encode mostly absence
gives an analyst **no indicators of compromise** — "this APK is malware because it lacks these 550
tokens" is not something a responder can act on. It also compounds the retraction above: the part of
the ranking that *is* model-specific is largely made of absences.

It matches `b5-androzoo/`, which found sign composition tracks input density (83.8% negated at 3%
density, 64.6% at 39%). LAMDA sits at ~3% density, so ~80% negated is the predicted value, and this
confirms that relationship on a second dataset.

## Inclusion frequency as no proxy for Shapley here

Reported alongside as a fifth ranking and not as an arm: Blakely & Granmo's Global Feature
Strength, the inclusion frequency of a feature across positive-polarity clauses. It overlaps the exact
Shapley top-100 by **49.3 of 100** — about as far off as χ² (53.7) and *further* than plain frequency
(62.0).

Their paper reports that the measure *corresponds* with SHAP on Wisconsin Breast Cancer (30 features)
and is explicit that it is not an identity. At 4,561 features the correspondence is weak. Worth citing
precisely, and it is a reason to say "equal to Shapley" about the closed form and never "like SHAP".

Inclusion frequency is also **data-independent** given a fixed model, so it shows zero explanation
drift by construction — which is why it cannot substitute for the data-conditional measure in
`b2-drift/`.

## Arms, and why four

| arm | ranking | role |
|---|---|---|
| 1 | exact Shapley attribution | the thing under test |
| 2 | χ² of feature against label | the standard univariate selector; the one that retracted the sibling result |
| 3 | document-frequency difference | the crudest dataset ranking — if even this reproduces the list, the claim dies simply |
| 4 | **random at matched k** | without it "the overlap is low" is unreadable. Chance gives 4 of 100 |

Arm 4 is the one that would be easy to omit and would have made the result uninterpretable. Arm 3
turned out to be the strongest dataset ranking, which is worth noting: the sophisticated selector was
not the dangerous one.

## Configuration

Flat FPTM, `clauses_per_class` 20, `T` 10, `S` 100, `L` 64, `LF` 10, 30 epochs, `LiteralCapped()`
ceiling, `parallel = :none`, tm-lab **43dba5f**, attribution threaded 16 ways and bit-identical.
Attribution uses 100 background and 100 explained rows from the training period, matching the budget
in `b2-drift/`. Between 447 and 482 of 4,561 features have nonzero exact attribution across the three
seeds — sparsity is a consequence of the closed form and not a threshold.

χ² and the frequency difference share a single pass over the training rows, so the two dataset
rankings cannot disagree about the data.

## Reproduce

```
julia --project=. -t 16 research/interp-dataset-control/run.jl 3
```

About seven minutes: three 150k-row trainings dominate, the dataset rankings take 0.4 s.
