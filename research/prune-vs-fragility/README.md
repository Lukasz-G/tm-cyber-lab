# prune-vs-fragility — does pruning help only where the labels are stable? No. The opposite.

`interp-residual-retrain/` found the pruning gain splits by period: +5.28 F1 over a size-matched control
across 2019–2022, −1.12 across 2016–2018. `b3-label-drift/` independently found 2017 and 2018 are exactly
where LAMDA's label boundary is most fragile. Two measurements made for unrelated reasons lined up, and
the obvious story was that pruning what fails to generalise needs labels stable enough for
"generalise" to mean something.

**The story is wrong.** Not unsupported — wrong in sign.

## Result

70 months (≥150 rows, ≥30 malware), 5 seeds, per-month evaluation. Label fragility is the share of that
month's malware a threshold of 10 detections would relabel benign; it ranges 11.0% to 95.7%.

| correlation across months | rho | p | n |
|---|---|---|---|
| pruning delta vs **label fragility** *(the test)* | **+0.253** | **0.0298** | 70 |
| pruning delta vs malware count *(control)* | −0.147 | 0.219 | 70 |
| pruning delta vs baseline F1 *(context)* | +0.134 | 0.264 | 70 |

| | mean pruning delta |
|---|---|
| months with **low** fragility (≤ median 80.8%) | +1.80 F1 |
| months with **high** fragility (> median) | **+3.34 F1** |

The hypothesis predicted a **negative** correlation — pruning helping more where labels are cleaner. The
measured correlation is **positive and significant**: pruning helps *more* where the labels are *more*
fragile, by roughly double.

## Verdict: rejected, and the paragraph is deleted rather than softened

The pre-registered criterion, fixed in the script header before the run:

> SUPPORTED if the delta correlates NEGATIVELY with fragility at p < 0.05 over the monthly series AND the
> control correlation with malware count is materially weaker. REJECTED if the fragility correlation is
> not significant, or if the count correlation is comparable, in which case the label-fragility paragraph
> is deleted from the manuscript.

The correlation is significant but of the opposite sign, which the criterion did not anticipate and the
script's branch logic reported as "AMBIGUOUS". That label is too generous: a hypothesis that predicts a
negative association and meets a significant positive one is refuted, not unresolved. **The
label-fragility paragraph has been removed from the manuscript**, per the `\todo` that accompanied it,
which said to delete rather than soften if the check failed.

The control behaved: the association is not explained by month size (−0.147, p = 0.22), so the positive
result is not an artifact of small noisy months. It is a real relationship pointing the other way, and we
have no account of it. We do not offer one.

## What this does not disturb

The regime split itself is unaffected — it was measured in `interp-residual-retrain/` and stands: pruning
gains ~5 F1 over a matched control on 2019–2022 and loses ~1 on 2016–2018. **What is now missing is any
explanation for it**, and pruning therefore remains a candidate intervention with an uncharacterised
precondition, not a recommendation.

## One discrepancy to be aware of

The monthly deltas here are **positive on average in both fragility halves** (+1.80 and +3.34), while the
year-resolution analysis found 2016–2018 negative. The two are computed over different sets: this
experiment keeps only months with ≥150 rows and ≥30 malware, 70 of them, so the thin early months that
drag the yearly figures down are partly excluded. The per-year verdict in `interp-residual-retrain/` is
the one to quote for the regime split, because it does not filter; this one is for the correlation, which
needs months with enough rows to give a stable F1.

## Design notes

Three arms — pruned, size-matched random control, and unpruned baseline — with features **zeroed rather
than dropped**, so width, `S` and `s = width/S` are identical across arms and both pruned arms carry the
same dead-channel perturbation. The delta is pruned minus random, never pruned minus baseline.

The **month-size control correlation is not optional**. Label fragility and sample size could easily move
together, and a correlation with fragility alone would not distinguish a real dependence on label quality
from fragility proxying thin months.

Month resolution is likewise not a refinement. At year resolution there are seven points, where a
correlation of 0.5 cannot be told from zero — the mistake `b3-label-drift/` made, whose year-level answer
had the opposite sign to its monthly one. Repeating it here would have been inexcusable, and the
significance requirement that rule lacked is written into this one.

## Reproduce

```
julia --project=. -t 8 research/prune-vs-fragility/run.jl 5
```

About 35 minutes; 15 trainings on 150k rows. Writes `months.csv` with the per-month deltas, fragility,
malware count and baseline F1.
