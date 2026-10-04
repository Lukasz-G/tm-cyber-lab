# Open issues — things known to be wrong or incomplete

Recorded so they are fixed deliberately, not rediscovered. Each entry says what is wrong, what
it affects, and what fixing it costs.

## 1. ~~Baselines compared at their default threshold, ours at a swept one~~ — fixed 2026-09-30

Closed by `research/b5-fairness/` (APIGraph) and `research/b1-threshold-transfer/` (LAMDA). Both sides
are now swept, and the comparison is reported three ways: average precision (threshold-free), F1 at a
threshold chosen on held-out training data, and the oracle value that the old numbers were.

**It changed a conclusion.** The boosters keep the better frontier on both corpora (+3.77 and +3.43
average precision). But choosing a threshold honestly costs FPTM 1.73 and 6.19 F1 against the boosters'
6–9 and 15–17, so FPTM leads the deployable-threshold F1 on 5 of 6 and 7 of 8 periods. Both halves must
be quoted together.

**The correction owed to older numbers is measured**, not estimated: every F1 elsewhere in this
repository is an oracle value, and the gap between oracle and achievable is 1.73 F1 on APIGraph and 6.19
on LAMDA for FPTM.

## 2. ~~Dominance counts come from one seed~~ — fixed 2026-09-30

Closed by `research/b5-fairness/`, which computes dominance on every seed. The result is 5/5 on all six
years, so the earlier one-seed counts were underpowered and not misleading.

Recorded because it is easy to over-read: "reaches outside" means the curve exceeds the booster's
precision at *some* recall. Since FPTM's average precision is lower on every year, the curves must
cross — neither frontier contains the other, and this is **not** a dominance claim.

## 3. ~~The threshold sweep picks its threshold on the test set~~ — fixed 2026-09-30

Closed by the same two experiments. Thresholds are chosen on a held-out slice of the training period and
applied unchanged to each test period.

A bug worth keeping: the first version split the pool 80/20 **by row order**, and both corpora order
their pools by class, so the validation slice came out with **0.0% malware**. It would not have crashed;
it would have selected a meaningless threshold in silence. The split is now interleaved, and the exact
indices are exported from Julia and not the rule re-derived in Python, with both sides asserting
their label vectors match row for row before anything is computed.

## 4. The unreconstructible FAR figure on LAMDA

Recorded in `research/b0-baseline/`. The published NEAR figure was recovered as a mean of per-year F1;
FAR was not. Nothing in this project reports a FAR mean, so this blocks only direct comparison against
the published FAR column.

**Fix:** ask the authors. Not resolvable from the release.

**Blocked with it:** pre-registered prediction 5, which needs Appendix F's strengthened/weakened/flipped
verdict counts — a detection count at two points in time per sample. `metadata.csv` carries one snapshot
per `sha256`, and there is no `vt_detections.csv` in the release despite an earlier note in this
repository claiming otherwise. `research/b3-label-drift/` measures what the release does support.

## 5. ~~The missing dataset control for interpretability~~ — fixed 2026-09-30, with a retraction attached

Closed by `research/interp-dataset-control/`, and the criterion fixed before the run fired: a document
frequency difference with no model in it recovers **62 of the model's top-100** attributed features, and
chi-squared recovers 54, against 4 for chance. **No readability claim may appear anywhere.**

Two follow-ups settled it further. `research/interp-residual/` and
`research/interp-residual-retrain/` show the 62 overlap features carry essentially the whole model.
The 38 the frequency count misses are where the model fails to generalise: removing them improves
drifted-period F1 by about 5 over a matched control on 2019–2022, though not on 2016–2018.

What replaces the readability claim is a drift-forensics one, and it is narrower: exact attribution
locates a prunable non-generalising subset from training data alone. It is not a recommendation, because
the effect reverses on three contiguous earlier periods.

## 6. ~~`tools/vast_setup.sh` and its nonexistent repository~~ — fixed 2026-09-29

Its default `TMCYBER_REPO` was `https://github.com/Lukasz-G/TM-Cyber.git`, a 404. The repository now
exists as `https://github.com/Lukasz-G/tm-cyber-lab.git` (public, default branch `master`) and the
script points at it. Verified to resolve. Kept here and not deleted so the entry does not read as
still open.

## 7. ~~The residual ablations' control, matched on size alone~~ — settled 2026-10-01

`research/interp-residual/` and `research/interp-residual-retrain/` ablate 38 features by forcing them to
zero, against a control of 38 features drawn uniformly from the nonzero-attribution support. The two sets
are matched on count and both carry the same dead-channel perturbation, which is why the verdict rests on
residual-against-random, not on residual-against-baseline.

They are not matched on the variable that governs how much the zeroing does. **Zeroing is not
sign-neutral**: it removes the evidence for a positive literal but *satisfies* a negated one. The residual
features sit at a training-period presence rate of 11.2% against roughly 1.5-2.3% for a uniformly drawn
set, and presence rate is what sets sign composition, since a feature that is almost always absent gets
included negated. Measured in `research/blacklist-anatomy/`: the residual set is 79.5% negated, the
uniform control 94.8%.

So the published `resid - random` column could in principle have been reporting *density*, not identity. **It was not.** `research/residual-density-control/` ran the separating arm: 38 features matched
feature by feature on presence rate, with sign composition reported as an outcome and not selected on.
Arms 1-3 reproduce the earlier run to the decimal on all eight periods, and closing most of the density gap
**raises** the residual's advantage over 2019-2022 from +5.28 to **+6.07**. The confound is real and it was
working against the claim, so the earlier figure was conservative.

Two things keep this from being a clean close. The match is **partial** - it closes about 60% of the
presence-rate gap and 52% of the sign gap, because the attribution support does not contain 38 unused
features at 11% presence - so the reading is the direction over the controlled range, not a measurement at
a complete match. And the **regime split deepens** under the better control, from -1.12 to -2.05 over
2016-2018, with 2017 at -4.24; on those periods the matched control beats the unpruned model, so pruning
almost any dense set helps there and pruning this set helps less. The split remains unexplained after
`prune-vs-fragility/` refuted the label-fragility account in sign. Pruning is therefore still **not**
offered as a recommendation, and that limitation is now about the regime split alone, not about the
control.

## 8. ~~The unmeasured ground-truth overlap~~ — run 2026-10-02, beyond the primary corpus's reach

Of the five interpretability measurements this project required before any claim about what clauses mean,
the fifth went unrun the longest: do the selected Drebin tokens correspond to documented indicators? It
was singled out at the design stage as the measurement that is *better* in this domain than in image or
text tasks, because the feature names carry meaning.

**On LAMDA they do not.** The released parquet columns are `feat_0` ... `feat_4560`: the Drebin vocabulary
is stripped. No semantic check is possible on the primary corpus with the public release, by us or by
anyone. The design-stage argument for this domain over raw-byte PE malware rested partly on feature names
meaning something, and for the primary corpus it does not hold.

**On APIGraph they do**, and our packed matrix already carried them: 1,159 entries matching the 2012
selected vocabulary exactly. `research/groundtruth-overlap/` runs the measurement there. The attributed
top-20 is **85.0% indicator-class against a 23.0% base rate**, with app-identity features suppressed from
63.1% to **1.7%**, and at k=100 exact attribution beats a frequency count by +12.7 points of indicator
share and -17.0 points of app-identity pollution.

Two things remain open and not fixed. Whether the result transfers to the primary corpus is
**untestable** with the public release. And no comparison against a curated external indicator list has
been made; the categories used are the feature set's own and are coarse. The paper states both.

The figure caption that read "they are indicators" on LAMDA evidence is corrected: it now claims the
median presence ratio above two, which is what that corpus can support, and points at the section that
answers the semantic question on the other one.
