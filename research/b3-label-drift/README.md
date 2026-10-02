# b3-label-drift — LAMDA's "concept drift" against movement in the label boundary

Every drift result on this dataset treats the labels as fixed ground truth. They are not: malware means
`vt_detection >= 4`, benign means exactly 0, and `[1,3]` is discarded. That is a threshold on a
continuous, vendor-generated quantity, and if the detection-count distribution moves then samples cross
the boundary for reasons that have nothing to do with the APK.

## First, the exclusions

Pre-registered **prediction 5** concerns LAMDA's Appendix F — samples whose VirusTotal verdicts
*strengthened, weakened or flipped* between two scans (10,289 weakened in 2017, and so on). That needs
a detection count at **two points in time** per sample.

**The public release does not contain it.** `metadata.csv` carries exactly one `vt_detection` per
`sha256`. There is no `vt_detections.csv` anywhere in the release, contrary to an earlier note in this
repository's experiment index, which was **wrong and has been corrected**. So **prediction 5 is untested
and blocked on asking the authors** — it is not reinterpreted, withdrawn or quietly satisfied by what
follows. What follows is a weaker, runnable question about the same worry.

## Answer: real movement in the label boundary, largely separate from feature drift

### Arm 1 — instability in the label

Share of each year's malware that a threshold of 10 detections would relabel benign:

| year | malware | median vt | 4–6 detections | flip at 5 | **flip at 10** |
|---|---|---|---|---|---|
| 2013 | 44,383 | 9.0 | 27.7% | 6.8% | 53.4% |
| 2014 | 45,756 | 11.0 | 27.0% | 10.1% | 44.6% |
| 2016 | 45,134 | 12.0 | 21.9% | 8.1% | **40.1%** |
| 2017 | 21,359 | 6.0 | **51.1%** | 18.5% | 77.8% |
| 2018 | 39,350 | 6.0 | **57.8%** | 25.4% | 76.3% |
| 2019 | 41,585 | 10.0 | 26.0% | 7.7% | 49.1% |
| 2020 | 46,355 | 7.0 | 40.7% | 12.2% | 67.3% |
| 2021 | 35,627 | 7.0 | 45.3% | 14.4% | **79.7%** |
| 2022 | 41,648 | 10.0 | 21.1% | 7.7% | 45.1% |

**Label fragility ranges from 40% to 80% and the median detection count from 6 to 12.** In 2017 and
2018 more than half of all malware sits at 4–6 detections — within two votes of being discarded as
ambiguous. This is not a constant, and no drift analysis on this dataset currently accounts for it.

Worth noting independently of prediction 5: **2017 is one of the two worst years here**, and 2017 is
also the year LAMDA's Appendix F reports its spike in weakened verdicts. Two different measurements on
two different quantities pointing at the same year is suggestive, not evidence — but it is the reason
to keep asking the authors for the Appendix F series.

### Arms 2 and 3 — feature drift, and its readability control

L1 distance between consecutive years' per-feature presence rates:

| transition | all samples | benign only | difference |
|---|---|---|---|
| 2013→2014 | 18.758 | 14.130 | +4.627 |
| 2014→2016 | 18.147 | 18.536 | −0.389 |
| 2016→2017 | 30.379 | 22.321 | +8.058 |
| 2017→2018 | 20.080 | 18.892 | +1.188 |
| 2018→2019 | **41.570** | 23.848 | **+17.722** |
| 2019→2020 | 19.487 | 19.073 | +0.414 |
| 2020→2021 | 34.094 | 29.835 | +4.259 |
| 2021→2022 | 30.827 | 24.431 | +6.396 |

Benign samples are `vt_detection == 0` exactly — the one label no threshold choice can disturb — so
drift measured on them is pure data drift with the label question removed.

### The comparison, at month resolution

| | rho | p | n |
|---|---|---|---|
| label fragility vs feature drift | **−0.344** | 0.0015 | 82 |
| feature drift, all vs benign-only *(control)* | **+0.748** | <0.0001 | 82 |

85 months with ≥200 rows.

**The control holds.** All-sample feature drift tracks benign-only feature drift at +0.748, so the
feature series is not an artefact of the labels reshaping which samples get averaged.

**The two signals are weakly and *negatively* related.** Months with more feature drift tend to have
*less* fragile labels. Real (p = 0.0015) but small, and it is not the confounding that would sink the
separability claim — if label fragility and feature drift were the same phenomenon seen twice, the
correlation would be strongly positive. It is not.

So: **largely separable, with a weak negative coupling that belongs in any write-up as a caveat and not as a finding.**

## The wrong sign at year resolution, as the methodological point

The rule was pre-registered in the script header: *separable* if |rho| < 0.5 with the control tracking,
*confounded* if |rho| ≥ 0.5. At year resolution it returned **rho +0.548, p = 0.160** and therefore
fired **CONFOUNDED**.

That verdict is wrong, and the rule that produced it was badly specified: it tested a correlation
threshold **without requiring significance**, which at eight transitions it cannot have. At month
resolution the same quantity is **−0.344** — the opposite sign — with p = 0.0015.

Both are kept in `results.txt`. A rule fixed in advance is not worth much if it is fixed at a
resolution that cannot answer the question, and the honest response is to say so, not to quote
whichever number reads better.

## Arms

| arm | what it measures | why it is needed |
|---|---|---|
| 1 | label fragility per period | the thing nobody measures |
| 2 | feature drift, all samples | the thing everybody measures |
| 3 | **feature drift, benign only** | the control. `vt_detection == 0` is threshold-independent, so drift here cannot be a labelling artefact. Without it, a correlation between 1 and 2 could not be told apart from the labels reshaping arm 2's population |

## Claims licensed

- That **LAMDA's labels carry drift of their own**, with fragility varying 40–80% by year, and that a
  paper treating them as fixed ground truth is making an unstated assumption.
- That **feature drift on this dataset is genuine**, since it survives restriction to samples whose
  label no threshold can move.
- It does **not** license prediction 5's claim about differently-timed data and label drift. That needs
  the two-timepoint verdict data and is blocked.
- It does **not** establish that label fragility harms detection. Appendix E reports that varying the
  threshold 4→13 barely changes baseline performance, which is a different and compatible fact:
  performance can be insensitive to a boundary that is nevertheless moving.

## Reproduce

```
python research/b3-label-drift/run.py
```

Reads `metadata.csv` (1,008,381 rows) and each year's `.tmx`. A few minutes, dominated by reading the
matrices. Memory: one year's dense matrix at a time, freed between years.
