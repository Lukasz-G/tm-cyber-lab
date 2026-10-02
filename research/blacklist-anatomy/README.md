# blacklist-anatomy — the "80% negated" figure as a property of input sparsity

`interp-dataset-control/` measured that about four in five included literals require a feature to be
**absent**, and this project concluded from it that the clauses "give an analyst no indicator of
compromise". **That conclusion is wrong**, and this experiment is what shows it.

## The numbers

LAMDA 2013–14 train, 150,090 rows, overall feature density **1.19%**. Three seeds; seed 1 shown, all
three agree to within a point.

| feature set | P(f \| malware) | P(f \| benign) | % of literals negated |
|---|---|---|---|
| exact-attribution top-100 | **0.299** | 0.188 | 61.8% |
| — overlap (62), *carries the model* | **0.419** | 0.237 | **53.4%** |
| — residual (38) | 0.112 | 0.112 | 79.5% |
| random control, from the support | 0.023 | 0.015 | 94.8% |
| **all 4,561 features** | 0.012 | 0.008 | **83.8%** |

Median presence ratio P(mal)/P(ben): **2.16** for the overlap set, **1.03** for the residual.

## Three readings, and the right one

**The model-wide 83.8% is the background rate, not a finding.** A *random* included literal is **94.8%**
negated. At 1.19% density a negated literal is satisfied for free on almost every row, so a clause
accumulates them at no cost. Quoting 83.8% as though it characterises the model is quoting the sparsity
of the data.

**The features the model actually relies on are far less negated** — 61.8% across the top-100 and
**53.4%** across the 62 that carry it, which is roughly balanced. And they are **malware-enriched**: present
in 42% of malware against 24% of goodware, a median ratio above two.

So the detector is **not** firing on absences, and it is **not** a "not-goodware" detector either — its
load-bearing features are present-in-malware indicators about half the time. **The claim that these
clauses supply no indicator of compromise is withdrawn.** Pre-registered prediction 1 is confirmed on raw
literal counts and the interpretation drawn from it was not.

## The residual's absence of class signal, and the pruning result

The 38 features a frequency ranking misses have `P(f | malware) = P(f | benign) = 0.112`, a ratio of
**1.03**. They carry **zero** discriminative information in the training period.

That is a far better account of `interp-residual-retrain/` than the label-fragility story, which was
refuted in sign. The model attributes weight to features that do not distinguish the classes; removing
them improves detection on later periods because they were never signal. It also fits the shape of the
earlier result — they cost a little in-distribution, where the model has fitted them, and help out of
distribution, where they do not transfer.

## A caveat this raises for the ablation experiments

`interp-residual/` and `interp-residual-retrain/` ablate features **by zeroing them**, with a random
control matched on **size**. The sign composition is not matched: residual 79.5% negated against the
control's 94.8%.

That matters because **zeroing is not sign-neutral**. Forcing a feature to zero removes evidence for a
*positive* literal but *satisfies* a **negated** one. A set that is 94.8% negated and a set that is 79.5%
negated therefore receive systematically different interventions, and the difference between them is not
purely "which features". The direction of the bias is not obvious and we do not claim it is small.

The right control would be matched on size **and** sign composition. That has not been run, and the
pruning result should be read with this caveat until it is.

## Figure

`paper/figures/blacklist-anatomy.py` draws the two partitions on class-conditional presence axes. The
overlap set sits below the diagonal, in the malware-enriched region; the residual sits on it. That is the
whole finding in one picture, and the figure is what makes "no class signal" concrete in place of a
number in a table.

## Arms

| arm | role |
|---|---|
| top-100 attributed | what the model relies on |
| overlap / residual, separately | the ablation experiments treat these as interchangeable apart from size; they are not |
| **random control from the nonzero-attribution support** | fixes what "rare everywhere" and "negated by default" look like on this data. Without it, 80% negated reads as a finding instead of as the background |

The random control is the arm that changes the conclusion, and it would have been easy to omit.

## Configuration

Flat FPTM, 20 clauses per class, `T` 10, `S` 100, `L` 64, `LF` 10, 30 epochs, `LiteralCapped()`,
`parallel = :none`, tm-lab **43dba5f**, 3 seeds. Attribution over 100 background and 100 explained rows
from the training pool, `k = 100`. Literal signs counted across both polarity banks of the malware class.

## Reproduce

```
julia --project=. -t 16 research/blacklist-anatomy/run.jl 3
python paper/figures/blacklist-anatomy.py
```

Writes `features.csv`: one row per top-100 feature with its class-conditional presence rates, signed
literal counts, exact attribution and partition.
