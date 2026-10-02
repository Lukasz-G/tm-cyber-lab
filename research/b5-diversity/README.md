# b5-diversity — the extra clauses and their non-duplication

If 10× the clause budget buys 0.75 F1, the obvious explanation is that the extra clauses are copies and
the feedback rule has saturated — which would make the ceiling a tm-lab problem, not a TM-Cyber
one.

## Result: rejected

APIGraph, 3 seeds. Pairwise Jaccard over clause literal sets, against size-matched random clauses.

| clauses | bank | n | median size | mean Jaccard | pairs >0.9 | random baseline |
|---|---|---|---|---|---|---|
| 20 | positive | 10 | 68 | 0.082 | **0%** | 0.014 |
| 20 | negative | 10 | 68 | 0.102 | **0%** | 0.014 |
| 200 | positive | 100 | 68 | 0.034 | **0%** | 0.013 |
| 200 | negative | 100 | 67 | 0.038 | **0%** | 0.013 |

Zero near-duplicate pairs at either budget. Diversity *increases* with budget (0.082 → 0.034). Trained
overlap is 2.5–7× the random baseline, so clauses do share real structure — they are learning
overlapping patterns, not noise — but nowhere near duplication.

**The random control is what makes this readable.** Sparse clauses over 1,159 features have low overlap
by chance alone, so "overlap is low" is meaningless without knowing what chance looks like at matched
clause sizes.

⇒ 100 structurally distinct patterns, +0.75 F1. The clauses are diverse and **predictively redundant**.
Not saturation by duplication.

## Reproduce

```
julia --project=. -t 16 research/b5-diversity/run.jl 3
```
