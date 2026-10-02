# b5-resolution — margin resolution, adequate for ranking

Last standing hypothesis after six knobs were eliminated. Precision at a high threshold needs
fine-grained ranking; an FPTM margin is an integer sum over ~10 firing clauses whose votes sit near the
bottom of [1, `LF`], against a booster's continuous score from 5000 trees. Recall collapsing to 11.8%
when 94% precision is demanded is what a large tied top bucket would look like.

## Result: rejected

APIGraph 2016 (56,036 rows, 9.5% malware), 3 seeds.

| clauses | distinct margin values | top bucket size | top bucket purity | precision @ 10% recall |
|---|---|---|---|---|
| 20 | 118 | **1** | **100%** | 96.0% |
| 200 | 283 | **1** | **100%** | 94.8% |

Top bucket is a single example at 100% purity. The ranking is fine-grained exactly where a
high-precision operating point needs it. 118 distinct levels over 56k examples is coarse in aggregate
but not at the head, and the head is what matters.

Consistent with the sweep instead of contradicting it: FPTM reaches 94% precision at ~12% recall,
LightGBM reaches 94% precision at 46% recall. **Same precision, 4x the recall.** The limit is the
*shape* of the PR curve — precision decays faster with recall — not its granularity.

For reference a gradient booster emits a distinct float per example: cardinality = n, top bucket = 1.

## Operationally relevant side note

**96% precision at 10% recall** is a real deployment point: low alert volume, high purity. Different
product from a booster tuned for recall, not strictly a worse one.

## Reproduce

```
julia --project=. -t 16 research/b5-resolution/run.jl 3
```
