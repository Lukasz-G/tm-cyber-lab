# b5-clause-length — clause length and the (L, clauses, T) grid against the frontier

Boosters compose many shallow conjunctions; an FPTM clause here is one ~69-literal conjunction with
tolerance 10. `L` sets that length and had been carried over from LAMDA unquestioned.

## Result: no cell escapes domination

APIGraph, 3 seeds, judged on the swept frontier because the default threshold costs 8.6 F1.

| L | clauses | T | literals | best swept F1 | domLGB | domXGB |
|---|---|---|---|---|---|---|
| 8 | 20 | 10 | 12 | 74.94 | 6/6 | 6/6 |
| 16 | 20 | 10 | 19 | 75.76 | 5/6 | 5/6 |
| 32 | 20 | 10 | 36 | 75.66 | 6/6 | 6/6 |
| **64** | **20** | **10** | 68 | 74.64 | **5/6** | **4/6** |
| 8 | 200 | 10 | 13 | 70.72 | 6/6 | 6/6 |
| 16 | 200 | 10 | 19 | 72.55 | 6/6 | 6/6 |
| 64 | 200 | 10 | 68 | 71.79 | 6/6 | 6/6 |
| 8 | 200 | 32 | 13 | 74.69 | 6/6 | 6/6 |
| 16 | 200 | 32 | 20 | 72.37 | 6/6 | 6/6 |
| 64 | 200 | 32 | 70 | 72.94 | 6/6 | 6/6 |

**`L` controls length precisely** — 8/16/32/64 give 12/19/36/68 literals, so clauses run 1.1–1.5x over
it, consistent with `L` being a growth gate and not a cap.

**The booster-shaped cell (8, 200) is the worst of the ten.** Many shallow conjunctions is the wrong
prescription here.

**`T` was a real confound on F1 and irrelevant to the frontier.** At 200 clauses the published relation
gives T ≈ 32, not 10. Rescaling recovers +3.97 F1 at L=8 and +1.15 at L=64 — and leaves dominance at
6/6 in every cell.

**Best of ten is the original gate config** (64, 20, T=10). Nothing beats what we started with.

## Caveats

- Dominance from one seed per cell in the first pass; best-F1 averages three. The 4/6-vs-6/6 ordering
  among cells is likely noise. The conclusion "no cell escapes" is safe because none does on any seed
  run. See `docs/open-issues.md`.
- Boosters unswept, as above. Dominance is the verdict; F1 columns are shape only.

## Reproduce

```
julia --project=. -t 16 research/b5-clause-length/run.jl 3
```

`results_grid.txt` is the ten-cell grid; `results.txt` the earlier four-cell `L`-only pass.
