# b4-footprint-labels — the cost of deployment, and the price of a label

Two secondary claims, measured and not quoted. One of them contradicts a figure this project has
cited about itself; the other kills the continual-learning claim and replaces it with something more
useful.

## Part A — the true footprint claim, and its difference from ours

Counted from the trained model at the gate configuration, width 4,561, 40 clauses total:

| | size | |
|---|---|---|
| include masks — what **inference** needs | **44.5 KB** | this is the ~50 KB figure |
| automaton state — what **updating** needs | **356.3 KB** | **8.0×** the masks |
| LightGBM on the same task, pickled | **144 MB** | **3,308×** the masks |

**The 3,300× advantage over gradient boosting is real and larger than advertised.** Even the updatable
form is 400× smaller than the booster.

**But the small-footprint and online-updating claims cannot be made about the same artefact.** The 50 KB
model is inference-only: include masks and nothing else. A model that keeps learning must carry one
automaton counter per literal per clause, which is 8× more. This project has cited both figures and
should not cite them together without this distinction.

## Part B — continual updating buys nothing, and stale data actively harms

$F_1$ on each year's held-out test split. `frozen` is the 2013–14 model, spending no labels.
`retrain-recent` trains a fresh model on the budget alone. `continual` continues training the 2013–14
model on the budget. `retrain-all` trains a fresh model on 2013–14 **plus** the budget. 2 seeds.

| year | budget | frozen | retrain-recent | continual | retrain-all |
|---|---|---|---|---|---|
| 2018 | 250 | 26.59 | **85.29** | 84.92 | 41.04 |
| 2018 | 1000 | 26.59 | **89.05** | 88.38 | 63.00 |
| 2018 | 4000 | 26.59 | **91.19** | 90.96 | 74.13 |
| 2018 | 16000 | 26.59 | **92.43** | 92.39 | 87.54 |
| 2019 | 250 | 74.58 | **91.83** | 91.29 | 79.36 |
| 2019 | 16000 | 74.58 | **95.81** | 95.81 | 90.84 |
| 2020 | 250 | 71.88 | **94.01** | 93.05 | 82.42 |
| 2020 | 16000 | 71.88 | **96.66** | 96.56 | 92.72 |
| 2021 | 250 | 63.94 | 93.39 | **94.08** | 80.52 |
| 2021 | 16000 | 63.94 | **96.76** | 96.44 | 94.70 |
| 2022 | 250 | 71.40 | **95.81** | 95.21 | 81.00 |
| 2022 | 16000 | 71.40 | **98.17** | 98.05 | 93.75 |

(Full 20-cell table in `results.txt`.)

### Three findings, in order of how much they change the story

**1. 250 fresh labels beat 150,090 stale ones, by 59 $F_1$.** On 2018 the frozen model trained on all of
2013–14 scores 26.59; a model trained on 250 labelled samples from 2018 scores 85.29. Drift on this
benchmark is severe enough that a tiny current sample dominates a large historical one.

**2. Keeping the old data is worse than discarding it — at every one of the 20 cells.** `retrain-all`
trails `retrain-recent` everywhere, by 44 $F_1$ at the smallest budget on 2018 and still by 5 at the
largest. The stale rows are not merely uninformative, they outvote the fresh ones. The gap closes as the
budget grows, which is the fresh data gradually winning the argument.

**3. Continual updating is indistinguishable from starting over.** `continual` and `retrain-recent` sit
within about 1 $F_1$ of each other in all 20 cells, in both directions (2021 at 250 is the one place
continual leads, by 0.69). **The label-efficiency claim fails**: continuing from the existing model does
not reach a given $F_1$ at a smaller budget. What the 2013–14 model contributes is not negative here —
it is simply not worth anything once fresh labels exist.

So the defensible operational statement is **not** "update the model online". It is: **label a small
recent sample and retrain on that alone.** Which is cheaper to implement than continual learning and
does not carry its poisoning surface.

This is consistent with the project's standing position that "online learning solves drift" is the wrong
framing — applying a label is nearly free, producing one is not. The measurement now says the thing
online updating was supposed to win at, it does not win at.

### The real caveat bounding all three

The budget is drawn from the **same year** as the test split (LAMDA splits each year 80/20), so these
numbers measure **adaptation to a distribution already observed**, not forecasting. "250 labels recover
most of the performance" means *if you can label 250 samples from the period you are about to be
evaluated on*. That is a realistic operational setting — an analyst triaging current samples — but it is
not zero-shot generalisation to an unseen future, and the numbers must not be read as such.

The `frozen` column is the honest zero-shot number, and it is the one that matches the detection tables
elsewhere in this repository (26.6 on 2018 against 28.1 measured there at 10 seeds).

## Arms, and why four

| arm | role |
|---|---|
| **frozen** | what the labels buy. Without it every other column looks like an absolute score, not an improvement |
| **retrain-recent** | the naive baseline, and the one that turned out to win |
| **continual** | the claim under test |
| **retrain-all** | the upper bound if labels are cheap and compute is not. Without it, continual beating retrain-recent would have looked like a win when it might only have been "more data helps" |

Arm 4 is what revealed finding 2, which is the most surprising result here and was not the thing being
tested.

## Configuration

Flat FPTM, `clauses_per_class` 20, `T` 10, `S` 100, `L` 64, `LF` 10, 30 epochs, `LiteralCapped()`,
`parallel = :none`, tm-lab **43dba5f**, 2 seeds. Base pool 150,090 rows from 2013–14 train portions.
Budgets drawn at random from each test year's train portion with a per-seed RNG — **not** as a prefix,
because LAMDA's per-year files are ordered by class and a prefix would be single-label. Evaluation is
always the year's test portion, which no arm trains on.

Footprint counts assume `states_num = 256`, so one byte per automaton; the include-mask figure is two
bits per feature per clause (the literal and its negation).

## Reproduce

```
julia --project=. -t 16 research/b4-footprint-labels/run.jl 2
```

About an hour, almost all of it the `retrain-all` arm, which trains on 150k+ rows in each of 20 cells.
