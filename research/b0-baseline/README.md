# b0-baseline — reproducing the published gradient-boosting baseline

**Calibration, not a finding.** Its job is to make every later comparison mean something, and to fail
loudly if our pipeline is not the authors' pipeline.

## The short version

Our LightGBM reproduces the published **IID F1 to two decimal places — 97.49 against 97.49**. It then
came out **12.6 points above** the published NEAR figure, which looked alarming and turned out to be a
definition: the published NEAR is a **mean of per-year F1**, not pooled over rows. Read that way, we
get 59.09 against 59.48 published — a 0.4 match.

FAR is still not reconstructed and is reported as unresolved.

## Reproduction, pooled over rows

Training on the released 2013–14 train portions, 5000 estimators, learning rate 0.02, 256 leaves, as
the paper reports.

| | IID | NEAR | published IID | published NEAR |
|---|---|---|---|---|
| LightGBM | **97.49** | 72.12 | 97.49 | 59.48 |
| XGBoost | 97.18 | 71.46 | 97.05 | 55.84 |

IID matching to 0.00 while NEAR is out by 12.6 is the diagnostic signature of a correct model with a
wrong evaluation-set definition. If our features, splits or hyperparameters were wrong, IID would not
land.

## Recovering the definition

A single fixed LightGBM model, evaluated six ways (`splitdef.py`, raw output `splitdef.txt`):

| definition | NEAR | vs published | FAR | vs published |
|---|---|---|---|---|
| pooled rows, all portions | 72.12 | +12.64 | 59.59 | +12.35 |
| pooled rows, held-out portion | 71.99 | +12.51 | 59.67 | +12.43 |
| **mean of per-year F1, all portions** | **59.09** | **−0.39** | 57.42 | +10.18 |
| **mean of per-year F1, held-out** | **59.19** | **−0.29** | 57.48 | +10.24 |
| mean per-year F1, FAR 2018–2025, all | — | — | 42.87 | −4.37 |
| mean per-year F1, FAR 2018–2025, held-out | — | — | 43.02 | −4.22 |

**NEAR is the mean of per-year F1.** Whether the held-out portion or the whole year is used barely matters, 59.19 against 59.09, because the release's split is stratified within year.
Whether you pool rows or average years matters enormously, because it changes the weight given to a bad year.

Why the difference is so large: 2016 scores 84.83 and 2017 scores **33.36**. Pooling rows lets 2016's
109,193 samples dominate; averaging years gives the two equal weight. The published number is the
average, so it is roughly the midpoint of a good year and a collapsed one.

**FAR remains unresolved.** 2018–2022 averaged gives 57.42; extending to 2018–2025 gives 42.87;
published is 47.24, which lies between. We do not know their year range, and and not search for a
subset that happens to reproduce 47.24 — which would be fitting a definition to a target — it is
recorded as not reconstructible from the release.

**Consequence:** comparisons of our models against each other are made on identical row sets and are
unaffected. Comparisons against the *published* figures are stated with the definition attached, and
never for FAR.

## Per-year results, non-monotone in time

| year | LightGBM F1 | XGBoost F1 | rows | malware |
|---|---|---|---|---|
| 2016 | 84.83 | — | 109,193 | 45,134 |
| 2017 | 33.36 | — | 99,144 | 21,359 |
| 2018 | **28.32** | 25.31 | 104,292 | 39,350 |
| 2019 | 72.22 | 70.73 | 91,050 | 41,585 |
| 2020 | 67.77 | 65.89 | 102,073 | 46,355 |
| 2021 | 53.87 | 56.25 | 81,155 | 35,627 |
| 2022 | 64.90 | 62.22 | 86,416 | 41,648 |

**Degradation is not monotone in time.** 2018, the year immediately after training, is the worst year
in the whole series — worse than 2022, four years further out — and 2019 is two and a half times
better than 2018. The same holds for XGBoost, and (in `../b1-gate/`) for a Tsetlin machine, so it is a
property of the data and not of any model.

Precision holds at 82–99% throughout while recall collapses from 97% to 15–59%. The failure mode is
missed malware, not false alarms, which is the thing an operational paper should lead with.

A single mean over 2018–2022 is 57.42 — a number that describes neither 2018 nor 2019. Nothing in this
project reports one.

## Determinism of gradient boosting here

The B1 comparison puts a ten-seed Tsetlin mean against a single LightGBM run, so LightGBM's own seed
variance has to be established, not assumed. With `feature_fraction` and `bagging_fraction` at
their defaults of 1.0 there should be no stochasticity left in the tree building.

**It is deterministic.** Five seeds, `seeds.py`, raw output `seeds.txt`: all five produced
**bit-identical predictions** on all seven evaluation sets, giving a standard deviation of exactly
0.0000 everywhere (IID 97.49, NEAR 72.12, 2018 28.32, 2019 72.22, 2020 67.77, 2021 53.87, 2022 64.90).

So the single run *is* the mean, and comparing a multi-seed Tsetlin mean against one LightGBM number is
correct in place of a shortcut. The caveat dissolves instead of needing to be averaged away — which is
the cheaper outcome, and the reason the check was worth running instead of assuming either way.

Note what this does *not* say: LightGBM is deterministic **at this configuration**, because
`feature_fraction` and `bagging_fraction` sit at their defaults of 1.0 and there is no stochasticity
left in the tree building. Turn either below 1.0 and the seed matters again.

## Deviations from the paper's protocol

1. The paper describes holding out the last month of each training year; the release ships an 80/20
   stratified within-year train/test split. We use the released split, so "IID" here is the 2013–14
   test portions, not two held-out months. Given that IID reproduces to 0.00, this appears not
   to matter.
2. NEAR and FAR use whole years unless stated; the table above shows the held-out-only variant differs
   by under 0.3 F1.

## Reproduce

```
python research/b0-baseline/run.py        # the two baselines, pooled
python research/b0-baseline/splitdef.py   # which definition reproduces the published figures
python research/b0-baseline/seeds.py      # is LightGBM deterministic here
```

`run.py` takes about 22 minutes (LightGBM 4 min, XGBoost 18). `splitdef.py` caches the trained model
to `data/lamda/` so it need not retrain.
