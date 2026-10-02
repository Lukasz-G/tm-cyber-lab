# b1-threshold-transfer — replication of threshold transfer on LAMDA

`b5-fairness/` found on APIGraph that gradient boosting holds the better precision/recall frontier while
its *operating point* does not survive drift. That was written up as an advantage which had survived a
cross-dataset check. **It had not** — `b5-fairness` runs on one corpus. This is the missing half, and it
was run because this project's own record is unkind to single-corpus findings: the LAMDA drift-robustness
advantage looked just as convincing and did not replicate.

There was also a mechanism that could have broken it. LAMDA's pool is **48% malware** against APIGraph's
**10%**, and threshold transfer is a claim about where a decision boundary sits.

## Answer: replication, with a larger effect here

### (a) Threshold-free — the boosters keep the better frontier

Average precision, 3 seeds. Positive gap is against us.

| eval | FPTM | LightGBM | XGBoost | gap |
|---|---|---|---|---|
| 2019 | 91.84 ± 1.48 | 92.58 | 94.71 | +2.86 |
| 2020 | 93.34 ± 1.17 | 91.38 | 94.78 | +1.43 |
| 2021 | 91.66 ± 1.68 | 87.71 | 92.59 | +0.93 |
| 2022 | 94.12 ± 1.25 | 94.70 | 96.58 | +2.46 |

**Mean gap +3.43 AP.** APIGraph gave +3.77. The frontier finding is the same on both corpora.

### (b) At a threshold chosen without seeing the future — FPTM leads 7 of 8

| eval | FPTM | LightGBM | XGBoost |
|---|---|---|---|
| IID | 95.65 | **97.44** | 97.15 |
| 2016 | **85.69** | 85.58 | 85.27 |
| 2017 | **38.85** | 34.46 | 28.67 |
| 2018 | **33.59** | 28.24 | 25.82 |
| 2019 | **81.13** | 72.57 | 75.78 |
| 2020 | **83.05** | 67.87 | 73.34 |
| 2021 | **78.49** | 52.97 | 64.54 |
| 2022 | **81.09** | 65.68 | 68.45 |

The only loss is IID, by 1.79, which matches the in-distribution deficit measured in `b1-gate/`. On 2021
the margin over LightGBM is **25.5 F1**.

### The mechanism, stronger on LAMDA

| | LAMDA | APIGraph |
|---|---|---|
| FPTM | **+6.19** | **+1.73** |
| LightGBM | +16.60 | +8.96 |
| XGBoost | +15.31 | +6.14 |

F1 given up by choosing the threshold on held-out training data instead of on the test year. FPTM gives
up **2.5–2.7× less** than the boosters on both corpora. The absolute costs are larger on LAMDA, which is
consistent with its drift being more severe — 2018 collapses for every model.

## Verdict

**Threshold transfer is a two-corpus result.** Both datasets agree on all three components:

1. Gradient boosting has the better frontier (+3.43 and +3.77 AP).
2. Honest thresholding costs the rule ensemble far less than it costs the boosters.
3. The rule ensemble consequently leads the deployable-threshold F1 on most periods (7/8 and 5/6).

This makes it the **only** claim in the project that holds on both corpora, and it replaces the
drift-robustness claim that did not. Both halves must still travel together: the boosters' frontier is
genuinely better, and quoting only the F1 table would overstate the result in exactly the way the earlier
comparisons did.

## The limits of the base-rate explanation

Threshold transfer could plausibly have been an artefact of APIGraph's 10% base rate — a boundary placed
in a sparse-positive regime might transfer for reasons that do not apply near balance. It survives at
48%, and with a larger effect, so the mechanism is not base-rate dependent. That was the specific reason
to run this and not assume it.

## Configuration and alignment

FPTM: 20 clauses per class, `T` 10, `S` 100, `L` 64, `LF` 10, 30 epochs, `LiteralCapped()`,
`parallel = :none`, tm-lab **43dba5f**, 3 seeds. Boosters: 5000 estimators, learning rate 0.02, 256
leaves — the same configuration used on APIGraph, so the two corpora are treated identically. Pool
150,090 rows → 120,072 train / 30,018 validation via an interleaved split (every 5th row), 48.1% and
47.8% malware.

Features are read from the `.tmx` matrices on both sides, not from the parquet release, because
the Julia export indexes those files: reading the same files makes "identical rows" provable and not assumed. **Both sides assert their label vectors match row for row before anything is computed**, and
that passed on the validation slice and all seven test years. Test years are subsampled to 25,000 rows
with a fixed seed.

Per-year only; no FAR mean, because LAMDA's 2024–25 malware counts are antivirus label lag, not drift and a mean across them would report the lag as a result.

## Reproduce

```
julia --project=. -t 16 research/b1-threshold-transfer/export_margins.jl 3
python research/b1-threshold-transfer/run.py
```

About three minutes for the export; the boosters on 120k × 4,561 dominate the Python side.
