# b0-composition — our copy of LAMDA against the published table

**Calibration, not a finding.** Every later comparison against the published 97.49 / 59.48 / 47.24 F1
is meaningless if we hold a different release than those numbers came from. This checks that we do.

## Answer: yes, exactly.

| quantity | ours | published | delta |
|---|---|---|---|
| APKs | 1,008,381 | 1,008,381 | 0 |
| malware | 369,906 | 369,906 | 0 |
| benign | 638,475 | 638,475 | 0 |
| families | 1,380 | 1,380 | 0 |
| singletons | 150,604 | 150,604 | 0 |

2015 absent as expected, 4,561 feature columns ⇒ 9,122 literals, 120 months present.

Per-year malware counts also reproduce the figures the paper cites for the late years — 2023 = 7,892,
**2024 = 794, 2025 = 23** against roughly 45,000 benign each. That collapse is antivirus label lag, not drift, and it is the reason nothing here is ever reported as a single FAR mean across
2018–2025.

## The one thing worth knowing: how to count families

The paper does not spell this out, and counting naively gives the wrong answer by two orders of
magnitude.

AVClass2 names every **unclustered** sample `singleton:<sha256>`, so those strings are per-sample
placeholders, not families. There is also a literal `unknown` bucket holding 2,985 samples. The
published figures correspond to:

- **families = distinct family names, excluding `singleton:*` and excluding `unknown`** → 1,380
- **singletons = number of `singleton:*` strings** → 150,604

Counting distinct family strings instead gives **151,985**, and counting families of size one gives
**151,075**. The gap between 151,075 and 150,604 is real and worth noting: **471 named families hold
exactly one sample**. So "singleton" in the paper's sense means *unclustered by AVClass2*, not
*family of size one* — they are different sets, and the 471 are in one but not the other.

Benign rows all carry the family string `benign`, so the malware/benign partition and the family
partition are consistent.

## Per-year composition

| year | split | rows | malware | benign | malware % |
|---|---|---|---|---|---|
| 2013 | train | 86,431 | 44,383 | 42,048 | 51.4% |
| 2014 | train | 101,183 | 45,756 | 55,427 | 45.2% |
| 2016 | near | 109,193 | 45,134 | 64,059 | 41.3% |
| 2017 | near | 99,144 | 21,359 | 77,785 | 21.5% |
| 2018 | far | 104,292 | 39,350 | 64,942 | 37.7% |
| 2019 | far | 91,050 | 41,585 | 49,465 | 45.7% |
| 2020 | far | 102,073 | 46,355 | 55,718 | 45.4% |
| 2021 | far | 81,155 | 35,627 | 45,528 | 43.9% |
| 2022 | far | 86,416 | 41,648 | 44,768 | 48.2% |
| 2023 | label lag | 54,354 | 7,892 | 46,462 | 14.5% |
| 2024 | label lag | 48,427 | 794 | 47,633 | 1.6% |
| 2025 | label lag | 44,663 | 23 | 44,640 | 0.1% |

Note that the "deliberately 50:50" balance is a corpus-level statement, not a per-year one: 2017 sits
at 21.5% malware and 2013 at 51.4%. Any per-month or per-year operating point has to be read against
that year's own base rate.

## The month counts as a constraint on the attribution protocol

120 months present. Row counts run from **2** to 55,497, median 4,294. **Six months hold fewer than
200 rows** and therefore cannot supply the 100 background + 100 explained samples the reference
explanation-drift protocol uses:

| month | rows |
|---|---|
| 2016-11 | 129 |
| 2016-12 | 32 |
| 2017-05 | 34 |
| 2017-07 | 2 |
| 2017-08 | 8 |
| 2017-09 | 4 |

These need an explicit policy before any month-over-month series is computed — excluded, or reported
with a smaller sample and marked. Silently letting them through would put six near-empty months into
a Jaccard curve, and month-to-month distances computed from 2 or 4 samples are noise by construction.
That is an independent reason to suspect the published curve, and it is checkable.

## Reproduce

```
python research/b0-composition/run.py data/lamda/raw
```

Raw output in `results.txt`. The dataset is `IQSeC-Lab/LAMDA` on HuggingFace (DOI
10.57967/hf/5563), baseline variant only — per-year Parquet, 341 MB, each row carrying `hash`,
`label`, `family`, `vt_count`, `year_month` and `feat_0..feat_4560`.

`year_month` being present in the Parquet matters: the reference explanation-drift script reads
monthwise `.npz` files from a path local to the authors' machine, which is not part of the release,
so monthly splits here are derived by us from the released `year_month` and not taken from their
own monthly files.
