# b5-androzoo — the dense regime, and the third dataset

APIGraph killed the drift-robustness claim. What was left was "a twenty-clause model is *competitive*
with gradient boosting", and this is the regime that should break that if anything does: **16,978
features at 39% density**, against APIGraph's 1,159 at 1.75% and LAMDA's 4,561 at 3%.

## The numbers

AndroZoo 2019–2021 as shipped with Chen et al. (USENIX Security 2023). Train on the 2019 pool (45,489
rows, 10.0% malware), test on 2020 and 2021. Ten seeds, 20 clauses per class, `T` = 10, `L` = 64,
`LF` = 10, `LiteralCapped`, 30 epochs, `parallel = :none`. tm-lab pin 43dba5f.

| year | FPTM (s-matched, S=372) | FPTM (S=100) | LightGBM | XGBoost | vs best booster |
|---|---|---|---|---|---|
| 2020 | 58.81 ±3.62 | 59.07 ±3.48 | 60.93 | **64.52** | **−5.45** |
| 2021 | 45.89 ±1.12 | 46.12 ±0.70 | 50.17 | **50.35** | −4.23 |

Pre-registered criterion: within 5 F1 of the **best** booster on both years. **Fails on 2020 by 0.45 of
a point**, passes on 2021.

That margin is small and the honest reading is "just outside", not "badly behind". But the criterion
was fixed in advance precisely so it could not be reinterpreted afterwards, and 5.45 > 5.

**Against LightGBM alone the gaps are −1.86 and −4.05 — inside the criterion on both years.** As on
APIGraph, XGBoost is the stronger booster and comparing against the weaker one would have produced an
apparent pass. This is the second dataset where that choice decides the verdict.

## `s` is again not the explanation

The two FPTM arms differ by 0.26 and 0.23 F1, despite `s = width/S` moving by a factor of 3.7 between
them (45.6 against 169.8). Across all three datasets now, matching `s` has changed results by under 2
points while the dataset has changed them by ten or more. Worth stating plainly: **`s` is a real
confound in encoder comparisons at differing widths, and it is not what is driving any of these
cross-dataset results.** The control has now earned its keep twice by ruling itself out.

## The finding that is actually new here

| | LAMDA | APIGraph | AndroZoo |
|---|---|---|---|
| feature density | 3% | 1.75% | **39%** |
| median literals per clause | 69 | 69 | 71 |
| `LF` / literals | 14.5% | 14.5% | 14.1% |
| **negated literals** | **83.8%** | — | **64.6%** |

Clause size and tolerance ratio are essentially invariant across a twentyfold change in input density —
69, 69, 71 literals and 14.5%, 14.5%, 14.1%. But the **negated-literal fraction drops from 84% to 65%**
exactly where density rises.

This is the mechanism behind pre-registered prediction 1 appearing directly. At 3% density a "feature
absent" literal is satisfied by 97% of rows, so it costs a clause almost nothing to include and clauses
hoard them; at 39% density that stops being true and the machine stops leaning on them. The prediction
itself was wrong — LAMDA produced no pure blacklists — but the reasoning underneath it was sound, and
this is the first evidence that **clause sign composition tracks input density rather than being a
property of the model**. That is a statement about how these machines work, not about malware, and it
did not require a win to establish.

## What the three datasets now say together

| dataset | density | malware | FPTM vs best booster |
|---|---|---|---|
| LAMDA | 3% | 37% | ahead 4–12 on four of five drifted years |
| APIGraph | 1.75% | 10% | behind 0.3–9.2 on all six years |
| AndroZoo | 39% | 10% | behind 4.2–5.5 on both years |

**Superior drift robustness was a LAMDA result.** What holds across all three is weaker: a twenty-clause
model lands within roughly five F1 of the best of two gradient boosters, at a few hundred literals per
class. That is a genuine efficiency claim and not a performance one, and it should be written that way.

## Limitations

- **Only two test years**, so this is a weaker test than APIGraph's six, and 2021 has 15,661 rows
  against 2020's 38,904, with individual months as thin as 381 rows. The 2021 column rests on less
  data than its narrow standard deviation suggests.
- **Hyperparameters carried over unchanged** except the `S` rescaling — correct for a replication, but
  `T` = 10 was derived for LAMDA and has never been tuned for a 10% base rate or a dense input. Whether
  tuning closes a 5.45-point gap is untested, and testing it must be pre-registered separately rather
  than used to convert this verdict.
- Gradient boosting here runs at 91–97% precision and 34–48% recall, a very different operating point
  from LAMDA's. Comparing single F1 values across such different precision/recall splits hides more
  than it shows; a threshold sweep would say more and has not been run.
- Third-party feature extraction, as with APIGraph.

## Reproduce

```
python -m tmcyber.build_apigraph data/apigraph/data/gen_androzoo_drebin data/apigraph/tmx/androzoo
python research/b5-androzoo/baselines.py
julia --project=. -t 16 research/b5-androzoo/run.jl 10
```

Raw output in `results.txt` and `baselines.txt`. An earlier run of `run.jl` printed `NARROWS` from a
missing `baselines.json`, which was a NaN artefact rather than a verdict; the script now warns when the
file is absent, and that warning should be treated as invalidating the run.
