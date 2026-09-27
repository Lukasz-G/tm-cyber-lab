# TM-Cyber

Tsetlin Machines on Android malware **concept drift**: flat Fuzzy-Pattern TM over static
Drebin-style features, evaluated on [LAMDA](https://arxiv.org/abs/2505.18551) under an
AnoShift-style temporal split.

## The claim

Not that online learning solves drift — it does not, and the reason is in
[What this is not](#what-this-is-not). The claim is:

> Exact, clause-level drift forensics at a clause budget small enough to read — and a test of
> whether reported explanation drift is real or an artifact of approximate attribution.

LAMDA measures SHAP explanation drift and finds a Jaccard distance close to 0.9 between consecutive
months over top-100 features: the important-feature set almost entirely turns over month to month,
while classification accuracy is comparatively stable. Those SHAP values come from a
`KernelExplainer` with 100 background and 100 test samples per month against an MLP, so a Jaccard of
0.9 is consistent both with real explanation drift and with estimator variance, and the measurement
cannot separate them.

A Tsetlin machine's clauses are not a sampled approximation of the model — they are the model. So
clause-level attribution has zero estimator variance, and the churn can be measured exactly. That
result does not depend on beating anyone's F1.

## Why this dataset

Drebin features are natively binary bag-of-tokens, so booleanization — normally the largest confound
in Tsetlin-machine work on security data, and in the sibling project worth roughly eight times any
algorithmic change — is eliminated by construction. What is measured is the machine, not the
encoder.

LAMDA's baseline variant is also structurally the setting where Fuzzy-Pattern TM is strongest:
4,561 sparse binary features, against the published IMDb result of 90.15% at one clause per class in
about 50 KB.

## Status

Design stage. No experimental results yet. The gate below can fail, and the project stops if it does.

| | |
|---|---|
| **B0** | pipeline calibration: LAMDA composition reproduces, LightGBM reproduces 97.49 / 59.48 / 47.24 F1, the Python/Julia boundary round-trips bit-exactly |
| **B1** | **the gate.** Flat FPTM matches the published LightGBM / XGBoost / MLP / SVM baselines at ~20 clauses per class, 2013–2022, binary detection. If it cannot, stop |
| **B2** | exact clause-level explanation drift, month over month, against the SHAP finding |
| **B3** | separate data drift from label drift, using LAMDA's per-year VirusTotal verdict changes |
| **B4** | model footprint, and labels consumed per unit of retained detection for continual vs periodic retraining at matched budget |
| **B5** | cross-check on APIGraph — single-dataset drift results are not credible |

## What this is not

- **Not a claim that online learning solves drift.** Online updating makes *applying* a label nearly
  free; it does not produce labels, and antivirus consensus is slow. LAMDA's own 2024–25 malware
  counts — 794 and 23 samples against ~45,000 benign per year — are that latency made visible. The
  XGBoost baseline retrains in minutes, and DroidEvolver already does lightweight online updates with
  pseudo-labels. Any win has to be label efficiency, on-device deployment or reaction latency, and
  has to be measured as such.
- **Not a claim of interpretability by construction.** On a high-dimensional sparse binary task in
  the sibling project, clause "rules" came out as near-strict conjunctions of thousands of literals,
  precise and useless to a human, and a separate readability result was retracted once its control
  showed the readable features came from the dataset rather than the model. Interpretability is
  measured here, never asserted.
- **Not flow-feature intrusion detection.** Saturated at 99.x%, dominated by dataset artifacts, and
  a setting where flat models already win.
- **Not a graph or message-passing result.** Graph Tsetlin Machine work lives in the sibling project
  and this track deliberately does not block on it.

## Layout

```
julia/bootstrap.jl        clones the tm-lab Tsetlin stack into vendor/ at a pinned commit
julia/tmx.jl              reads the packed feature matrix as TMCore inputs
python/tmcyber/           dataset handling, splits, baselines; writes the packed matrix
docs/matrix-format.md     the .tmx format spec -- header plus raw packed words
research/<name>/          experiments: run script, raw output, README with question and answer
tools/                    bootstrap and arm fan-out for a rented CPU box
```

The Python/Julia boundary is the binarized feature matrix. Feature extraction, dataset handling,
baselines and statistics are Python; Tsetlin training, clause inspection and drift measurement are
Julia.

## Setup

```
julia --project=. julia/bootstrap.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'

python -m venv .venv
pip install -r requirements.txt
```

Then check the boundary, which every result depends on being bit-exact:

```
python test/boundary.py write /tmp/fx
julia --project=. test/boundary.jl /tmp/fx
python test/boundary.py verify /tmp/fx
```

`bootstrap.jl` clones [tm-lab](https://github.com/Lukasz-G/tm-lab) at a pinned commit rather than
vendoring it, so experiments reproduce from a fresh checkout without carrying another project's
source. `Manifest.toml` is not committed; the root `Project.toml` records the package paths.

Datasets are not committed. LAMDA is `IQSeC-Lab/LAMDA` on HuggingFace (DOI 10.57967/hf/5563).

Heavier sweeps run on a rented CPU box — `tools/vast_setup.sh` installs a pinned Julia and the
pinned Tsetlin stack, `tools/fanout.sh` runs experiment arms in parallel and appends each result to
one small log as it finishes. Cores go to arms rather than to making one run faster: training can
thread across classes or across clauses, but at the clause budgets used here neither pays, and at
the smallest one the clause axis is slower than running serially. Scoring does thread across
examples, bit-identically, which is what the attribution work needs.

## Method notes worth knowing before reading any result

- **Never a FAR mean across 2018–2025.** LAMDA's later years have too little labelled malware for a
  mean to mean anything; everything is reported per year.
- **LAMDA is balanced 50:50 by design**, which the dataset's authors argue for as a benchmark. It
  does not reflect deployment, so any operational number is also reported at a realistic base rate.
- **Clause counts are stated per class across both polarities**, matching the reference
  implementation, so "20 clauses" is unambiguous.
- **`L` is a growth gate, not a cap.** It gates whether a clause may grow in a given round and does
  not bound clause size; clauses routinely run several times over it. No result here reports clause
  sizes as bounded by `L`.

## License

MIT — see [LICENSE](LICENSE). Uses the tm-lab Tsetlin stack, which derives from Tsetlin.jl and
FuzzyPatternTM; attribution obligations are in [NOTICE.md](NOTICE.md).
