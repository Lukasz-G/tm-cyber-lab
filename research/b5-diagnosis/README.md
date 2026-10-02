# b5-diagnosis — precision/recall decomposition against the clause budget

First step of the "why is FPTM dominated" line. B5 recorded only F1, which cannot separate *misses
malware* from *cries wolf*.

## Result

APIGraph, 5 seeds, default threshold. Gate config vs 10x the clauses, with and without `T` rescaled.

| arm | mean gap to best booster | worst year |
|---|---|---|
| 20 clauses, T=10 (gate) | −6.77 | −9.85 |
| 200 clauses, T=10 | −6.02 | −8.72 |
| 200 clauses, T=32 | −6.06 | −9.73 |

Median literals/clause: 69, 68, 70. Unchanged by budget.

**Clause budget is not binding.** 10× clauses buys 0.75 F1. `T` rescaling: nothing.

## The decomposition, 2016

| model | precision | recall |
|---|---|---|
| FPTM 20 clauses | 82.76 | 45.62 |
| FPTM 200 clauses | 75.17 | 50.83 |
| LightGBM | 94.13 | 46.09 |
| XGBoost | **82.79** | **58.26** |

vs LightGBM: FPTM matches recall (45.6 vs 46.1), loses 11 precision.
vs XGBoost: at **identical precision** (82.76 vs 82.79), FPTM gives up **12.6 recall**.

Raising the clause count moves FPTM *along* the trade-off — precision −7.6, recall +5.2 — not outward.
This is the signature every subsequent arm reproduced.

## Reproduce

```
julia --project=. -t 16 research/b5-diagnosis/run.jl 5
```

Does not revisit the B5 verdict: that replication ran at the pre-registered configuration and stands.
