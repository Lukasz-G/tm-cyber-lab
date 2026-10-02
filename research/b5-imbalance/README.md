# b5-imbalance — class imbalance excluded as the cause

Strongest hypothesis on the cross-dataset evidence: width varies 15x and density 22x across our three
datasets and neither tracks the outcome, but base rate does perfectly — LAMDA at 37% malware is where
FPTM did well, both 10% datasets are where it is dominated. FPTM has no class weighting, so at 90%
benign the benign side receives nine times the feedback. And **every dataset this line of work has ever
used is balanced** (MNIST, Fashion-MNIST, CIFAR-10, IMDb), so imbalance was an untested regime.

## Result: rejected

APIGraph, 20 clauses, 5 seeds. Evaluation always on the unchanged ~10% malware test distribution.

| train scheme | rows | mal% | best swept F1 | domLGB | domXGB | default P / R |
|---|---|---|---|---|---|---|
| as shipped | 30,533 | 10.0 | 74.69 | 6.0/6 | 5.2/6 | 85.2 / 55.7 |
| balanced, undersampled | 6,122 | 50.0 | 74.09 | 6.0/6 | 5.6/6 | 70.0 / 75.5 |
| balanced, oversampled | 54,944 | 50.0 | 75.06 | 6.0/6 | **6.0/6** | 72.6 / 75.7 |
| **volume control** | 6,122 | 10.0 | 75.04 | 5.8/6 | 5.0/6 | 86.0 / 58.5 |

Balancing moves the frontier by nothing (74.69 → 74.09/75.06, within noise) and makes dominance against
XGBoost *worse*. What it does is slide the operating point: precision −14, recall +20 — the same
along-the-frontier trade that clause count produces.

**Why the volume control was necessary:** undersampling changes balance *and* data volume together.
Without arm 4 the result would have been uninterpretable. With it, both are answered at once.

## Side result, more informative than the hypothesis

**6,122 rows ≡ 30,533 rows** (75.04 vs 74.69). Four-fifths of the training data is inert. Combined with
10× clauses buying 0.75 F1, the model converges on a small fraction of the available data and capacity.

## Reproduce

```
julia --project=. -t 16 research/b5-imbalance/run.jl 5
```

Oversampling is by replication, which for a Tsetlin machine is equivalent to class-weighted feedback —
the same example drives updates nine times.
