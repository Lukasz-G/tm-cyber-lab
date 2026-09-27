# b0-boundary — the Python/Julia matrix boundary at full scale

**Calibration.** The boundary was already checked bit-exact on small fixtures at 1, 64, 65 and 4,561
columns. What was untested is scale, per-month slicing, and whether a month slice is fast enough to
do ~120 of them.

## Answer: PASS, all 12 years

Julia's read is compared against a digest computed **from the original Parquet by a separate script**
(`digest.py`), so the two sides never share code — reading the `.tmx` back in Python would only prove
the writer agrees with itself.

Checked per year: row count, total popcount over all 4,561 columns, malware count, and that a
month-sliced read returns the same rows as slicing the full read. 1,008,381 rows total.

| | |
|---|---|
| full-year load | 0.08 – 0.39 s |
| month slice | 0.056 – 0.552 s (median 0.121 s) |
| packed size | 25.7 – 62.9 MB per year, 661 MB including sidecars |

Month slicing is comfortably fast enough for a ~120-month drift series.

## Month coverage is irregular, and it matters downstream

The conversion surfaced something the composition check only hinted at. Coverage is not twelve months
per year:

- **2013**: months 6, 9, 10, 11, 12 only
- **2014**: months 1–8 only
- **2016**: months 2–12 (no January)
- **2017**: months 1–5, 7–12 (no June)
- **2025**: month 1 only

Six months hold fewer than 200 rows (2017-07 has **2**, 2017-09 has **4**).

**Consequence for any month-over-month measurement:** "consecutive months" is not a uniform one-month
step. A 2013-06 → 2013-09 pair is a three-month gap, and a pair involving 2017-07 is computed from two
samples. Any Jaccard or Kendall series over consecutive months therefore has a non-uniform time axis
and several points that are noise by construction, unless gaps and thin months are handled explicitly.
That has to be stated wherever such a series is reported, and it is an independent reason to be
careful about reading trends off one.

## Reproduce

```
python research/b0-boundary/digest.py > research/b0-boundary/expected.txt
julia --project=. research/b0-boundary/run.jl
```

Raw output in `results.txt`, expected digests in `expected.txt`.
