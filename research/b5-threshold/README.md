# b5-threshold — dominated, or just badly thresholded?

Every comparison in the project used one operating point per model. A TM's class score is an integer
margin, so it has a threshold to sweep, and nobody had swept it.

## Two results

**1. The default argmax threshold costs 8.6 F1 on average** (best swept minus default, APIGraph, all six
years, 5 seeds). Every F1 this project has reported — B1 on LAMDA included — is at a suboptimal point.

**2. FPTM is still dominated.** It cannot reach a booster's recall at that booster's precision, nor its
precision at that booster's recall:

| | dominated on |
|---|---|
| vs LightGBM | **6 of 6 years** |
| vs XGBoost | 5 of 6 |

Worst case, 2016: to match LightGBM's 94.1% precision, FPTM's recall falls to **11.8%** against
LightGBM's 46.1%.

| year | default F1 | best swept F1 | recall @ LGB precision | LGB recall |
|---|---|---|---|---|
| 2013 | 82.21 | 88.09 | 65.7 | 74.4 |
| 2014 | 65.18 | 74.85 | 30.7 | 49.4 |
| 2015 | 60.07 | 69.71 | 25.1 | 45.5 |
| 2016 | 58.55 | 67.46 | **11.8** | 46.1 |
| 2017 | 63.40 | 72.64 | 22.6 | 53.9 |
| 2018 | 63.09 | 71.31 | 30.8 | 46.0 |

⇒ the gap is representational, not a threshold artefact. The frontier itself is inside theirs.

## Caveats

- Best-F1 picks its threshold on the test set — an oracle upper bound, not an achievable operating
  point. A validation-chosen threshold would be lower.
- Boosters are at their default 0.5 cut, so F1 comparisons flatter us. The dominance test does not
  depend on either threshold and is the verdict. Both recorded in `docs/open-issues.md`.

## Reproduce

```
julia --project=. -t 16 research/b5-threshold/run.jl 5
```
