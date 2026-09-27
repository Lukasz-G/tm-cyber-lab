# b0-sanity — first flat FPTM on LAMDA, and the two readability diagnostics

**Not the gate.** The detection number here is one serial run at one configuration on the IID split
only. B1 is the gate and it needs NEAR and per-year FAR against the published baselines.

Configuration: 20 clauses per class across both polarities, `T` = 10, `S` = 100, `L` = 64, `LF` = 10,
`LiteralCapped` ceiling, `parallel = :none`, 10 epochs, single thread, tm-lab pin 43dba5f. Train =
the released 2013–14 train portions (150,090 rows, 72,111 malware); test = the 2013–14 test portions
(37,524 rows, 18,028 malware). Width 4,561 ⇒ 9,122 literals.

## It trains, and the first number is encouraging

**IID F1 = 95.11** (final, not best — best was 95.53 at epoch 7, and quoting peaks is how an earlier
project talked itself into a result that reversed). Precision 96.51, recall 93.76, FNR 6.24,
FPR 3.14. About **7 s per epoch** serial.

For context, against the published baselines on IID: LightGBM 97.49, XGBoost 97.05, MLP 97.21,
SVM 94.98. So 20 clauses per class already beats the published SVM and sits **2.4 F1 points behind
LightGBM** — before any tuning, and at a model size where the whole thing is a few hundred literals.

Nothing follows from that yet. The gate is about NEAR and FAR, where every published model collapses
from ~97 to ~59 and ~47, and where a 20-clause model has no particular reason to behave the same way.

## Both pre-registered predictions were wrong

They were written down before any LAMDA model existed, which is the only reason this is worth
reporting rather than rationalising.

### Prediction 2 — readability ratio — **wrong, and in the helpful direction**

Predicted: LAMDA would land near the published IMDb configuration's 1.7% `LF`/included-literals,
meaning near-strict conjunctions of thousands of literals, precise and useless to a human.

Measured: **14.49%** — `LF` = 10 against a median of **69** included literals per clause. Minimum 1,
maximum 144. That is far closer to MNIST's 21%, where clauses decompose into a readable core plus a
tolerance tail, than to IMDb's 1.7%, where they do not.

So the interpretability route is **not** closed off the way the prediction assumed. A 69-literal
clause is not a one-line rule, but it is three orders of magnitude away from the ~3,800-literal cores
that made the IMDb extraction useless. The reason the prediction failed is worth noting: it reasoned
from the *input width* (9,122 literals, IMDb-like) when the governing quantity is the *clause size*,
which is set by `L`, `LF` and the data's sparsity rather than by the width.

Clauses run **2.2× over `L` = 64** (median 69, max 144), which independently reproduces the sibling
project's finding that `L` is a growth gate and not a cap.

### Prediction 1 — pure blacklists — **wrong in substance, right in tendency**

Predicted: near-pure blacklists — no clause requiring any feature to be *present*, only absent —
which would mean the model offers an analyst no indicators of compromise.

Measured: **83.4% of literals are negated**, so the tendency is real and strong. But **0 of 20
clauses are pure blacklists.** Every clause carries at least one positive literal; the per-clause
positive fraction runs from 0.049 to 1.000 with a median of 0.167, i.e. roughly 11–12 "feature
present" literals in a typical 69-literal clause.

That difference matters for a security claim. A pure blacklist would have been a real problem to
disclose. Blacklist-*dominated* clauses that still carry a dozen positive literals each do contain
candidate indicators of compromise — and whether those are meaningful is exactly what the dataset
control and the ground-truth-overlap measurement have to decide. Neither has been run.

## What this licenses, and what it does not

- It licenses **attempting** an interpretability claim, which prediction 2 had provisionally written
  off. It does not license making one: the dataset control (would a χ² or frequency ranking of the
  raw features produce the same list?) retracted the equivalent result in the sibling project and has
  not been run here.
- It does **not** say anything about drift. Everything above is IID.
- The 95.11 is one seed, one configuration, one split. Treat it as evidence the pipeline works, not
  as a result.

## Reproduce

```
julia --project=. research/b0-sanity/run.jl
```

Raw output in `results.txt`. Requires the packed matrices built by
`python -m tmcyber.build_tmx data/lamda/raw data/lamda/tmx`.

One bug worth recording because it silently produced a plausible-looking wrong number: `100f1` in
Julia is a **Float32 literal** equal to 1000.0, not `100 * f1`. The first run printed an F1 of
"1000.00" for every epoch. It was obvious here; in a table of many numbers it would not have been.
