# b5-apigraph — the drift advantage does not replicate

**This is a negative result and it is the most important experiment in the project so far.**

On LAMDA, flat FPTM at twenty clauses per class sat 1.8 F1 behind gradient boosting in-distribution and
**4 to 12 points ahead** of it on four of five drifted years. The obvious question is whether that is a
property of Tsetlin machines or a property of LAMDA. It is a property of LAMDA.

## The numbers

APIGraph, the dataset's own protocol: train on the 2012 pool (30,533 rows, 10.0% malware), test on each
year 2013–2018. Width 1,159. Ten seeds, 20 clauses per class, `T` = 10, `L` = 64, `LF` = 10,
`LiteralCapped`, 30 epochs, `parallel = :none`. tm-lab pin 43dba5f.

| year | FPTM (S=25) | FPTM (S=100) | LightGBM | XGBoost | vs LGB | vs XGB |
|---|---|---|---|---|---|---|
| 2013 | 82.95 ±2.40 | 84.24 ±2.01 | 84.32 | 83.29 | −1.37 | −0.34 |
| 2014 | 64.92 ±4.76 | 66.76 ±3.07 | 65.31 | 69.88 | −0.39 | −4.96 |
| 2015 | 60.28 ±3.81 | 61.25 ±2.24 | 61.75 | 66.13 | −1.47 | −5.85 |
| 2016 | 59.23 ±4.32 | 59.65 ±3.56 | 61.88 | 68.39 | −2.65 | −9.16 |
| 2017 | 63.52 ±6.63 | 64.55 ±4.56 | 67.76 | 72.13 | −4.24 | −8.61 |
| 2018 | 64.70 ±5.09 | 65.28 ±4.12 | 60.59 | 72.30 | **+4.11** | −7.60 |

Pre-registered criterion: within 5 F1 on the first test year *and* ahead on a majority of later years.
The first condition holds (−1.37). The second fails — **ahead on 1 of 5**, and against XGBoost behind
on all six.

## What the third arm rules out

The two FPTM columns exist because `s = width/S` has confounded a comparison in this project's sibling
before, once accounting for more of an apparent effect than the thing being measured. LAMDA ran width
4,561 at `S` = 100, so `s` = 45.6; holding that constant at width 1,159 requires `S` = 25.

The two arms differ by only 0.4–1.8 F1, and the **unmatched** arm is slightly *better*. So `s` is not
the explanation. Without this arm, "the result did not replicate" would have been indistinguishable
from "we changed `s` and it broke", and those have completely different consequences. The dataset is
the explanation.

## A detail that would have flattered us

On LAMDA, LightGBM and XGBoost were close (97.49 / 97.18 on IID, within 3 points on every FAR year).
On APIGraph they are **not**: XGBoost beats LightGBM by 4–12 points on every year after 2013, reaching
72.30 against 60.59 in 2018.

The single year where FPTM is ahead of LightGBM, 2018, is precisely the year where LightGBM is weakest,
and FPTM is 7.6 points *behind* XGBoost there. Had this experiment carried only LightGBM — the model
LAMDA's own paper leads with — the table would have read as a partial replication. It is not one.

## What survives and what does not

**Does not survive:** any claim that Fuzzy-Pattern Tsetlin machines are more robust to concept drift
than gradient boosting. That was a LAMDA result and it is now scoped to LAMDA. It should not appear in
a paper as a general claim, and the LAMDA table should carry this result beside it.

**Survives:** that a twenty-clause model is *competitive* — within 1.4 F1 of LightGBM on the first test
year of a second dataset at a realistic 10% base rate, and within 0.4 of XGBoost there — at a few
hundred literals per class. That is a real and useful claim, and a weaker one than we had yesterday.

**Untouched:** the method result (exact Shapley in closed form for additive count-symmetric rule
ensembles) and the noise-floor result (LAMDA's reported explanation drift sits at its own estimator's
floor). Neither depends on the detector winning anything. The project's spine was already the method
rather than the detector, which is the only reason this negative result costs a section rather than the
paper.

## Limitations, including one that cuts both ways

- **Hyperparameters were carried over unchanged**, only `S` rescaled. That is the right protocol for a
  replication — tuning per dataset would be fitting rather than testing — but it means "does not
  replicate" partly overlaps with "was not tuned for a 10% base rate", and `T` = 10 was derived for
  LAMDA's configuration. Whether tuning changes the conclusion is **untested**, and if it is tested it
  has to be pre-registered as its own question rather than used to rescue this one.
- **These are not the APIGraph authors' features.** APIGraph releases only MD5 hashes, so this uses the
  Drebin extraction published with Chen et al., *Continuous Learning for Android Malware Detection*
  (USENIX Security 2023) — almost certainly the same data LAMDA's own APIGraph comparison used, given
  the matching file layout, but a third party's extraction nonetheless.
- **Different protocol from LAMDA by design.** One 2012 training pool and per-year tests, which is what
  the dataset ships. Absolute F1 is not comparable across the two datasets; only the gap to gradient
  boosting on identical rows is.
- A third dataset is available and unused: the same archive carries Drebin features for AndroZoo
  2019–2021 at width 16,978 and **39% feature density**, against APIGraph's 1.75%. That is a much
  larger regime change than this one, and worth running before anything is written.

## Reproduce

```
python -m tmcyber.build_apigraph data/apigraph/data/gen_apigraph_drebin data/apigraph/tmx/apigraph
python research/b5-apigraph/baselines.py
julia --project=. -t 16 research/b5-apigraph/run.jl 10
```

Raw output in `results.txt` and `baselines.txt`; the baselines are cached in `baselines.json`.

One trap in the source format, recorded because it silently produces a plausible wrong answer:
`y_train` holds **multi-class family indices**, not a binary flag. Zero is benign and every positive
value is a family. Testing `y == 1` selects one arbitrary family — 179 samples of the 3,061 malware in
the 2012 pool — and yields a dataset that appears to be 0.0% malware. Binary detection is `y > 0`.
