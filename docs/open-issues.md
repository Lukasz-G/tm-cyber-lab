# Open issues — things known to be wrong or incomplete

Recorded so they are fixed deliberately rather than rediscovered. Each entry says what is wrong, what
it affects, and what fixing it costs.

## 1. Baselines are compared at their default threshold, ours at a swept one

**Status: known unfair, in our favour. Must be fixed before anything is published.**

`research/b5-threshold/` and `research/b5-clause-length/` report FPTM's *best swept* F1 alongside
LightGBM and XGBoost F1 taken at their default 0.5 probability cut. A swept model against an unswept one
is not a comparison, and the gap it shows flatters us by construction.

The verdicts in those experiments deliberately rest on the **threshold-free dominance test** — can
FPTM's curve reach a booster's precision at its recall, or vice versa — which is unaffected. But the F1
columns must not be quoted.

**Fix:** sweep the boosters' predicted probabilities the same way and compare frontier against frontier,
or compare area under the precision/recall curve. Both models are already trained; the boosters need
`predict_proba` instead of `predict`. Perhaps 20 minutes including a rerun.

**Also affected:** every earlier F1 number in the project is at FPTM's default argmax threshold, which
the sweep showed costs **8.6 F1 on average** on APIGraph. So B1 on LAMDA, and both B5 cross-checks,
understate the model. The *verdicts* stand — they were pre-registered on default-threshold comparisons
against default-threshold baselines, which is at least symmetric — but the absolute numbers are
pessimistic and should be restated once both sides are swept.

## 2. Dominance counts come from one seed

**Status: underpowered, direction unknown.**

In `research/b5-clause-length/` the best-F1 figures average five seeds but the dominance counts are
computed from seed 1 only, to save time. That is inconsistent, and the counts it produced (4/6 to 6/6
across four `L` arms) are close enough that the differences between arms are probably noise.

**Consequence:** the conclusion "shorter clauses do not move the frontier outward" is safe, because no
arm escapes domination on any seed tested. The apparent ordering among the arms is **not** safe and
should not be read as `L` = 64 being genuinely best.

**Fix:** compute dominance on every seed and report mean ± spread. A few minutes of compute.

## 3. The threshold sweep picks its threshold on the test set

The best-F1 figures are oracle values — the threshold is chosen on the same data it is scored on — so
they are an upper bound rather than an achievable operating point.

**Fix:** choose the threshold on a held-out slice of the training period and apply it unchanged to each
test year. That is the number a deployment would actually get, and it will be lower.

## 4. FAR is not reconstructible on LAMDA

Recorded in `research/b0-baseline/`. The published NEAR figure was recovered as a mean of per-year F1;
FAR was not. Nothing in this project reports a FAR mean, so this blocks only direct comparison against
the published FAR column.

**Fix:** ask the authors. Not resolvable from the release.

## 5. Interpretability has no dataset control yet

`LF`/included-literals and sign composition are measured, but the control that matters — whether a χ² or
frequency ranking of the raw features produces the same feature list — has not been run. In the sibling
algorithm project that control **retracted** the equivalent result. Until it runs, no claim of readable
rules may appear anywhere.

## 6. ~~`tools/vast_setup.sh` points at a repository that does not exist~~ — fixed 2026-09-29

Its default `TMCYBER_REPO` was `https://github.com/Lukasz-G/TM-Cyber.git`, a 404. The repository now
exists as `https://github.com/Lukasz-G/tm-cyber-lab.git` (public, default branch `master`) and the
script points at it. Verified to resolve. Kept here rather than deleted so the entry does not read as
still open.
