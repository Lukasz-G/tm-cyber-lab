# Open issues — things known to be wrong or incomplete

Recorded so they are fixed deliberately rather than rediscovered. Each entry says what is wrong, what
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
years, so the earlier one-seed counts were underpowered rather than misleading.

Recorded because it is easy to over-read: "reaches outside" means the curve exceeds the booster's
precision at *some* recall. Since FPTM's average precision is lower on every year, the curves must
cross — neither frontier contains the other, and this is **not** a dominance claim.

## 3. ~~The threshold sweep picks its threshold on the test set~~ — fixed 2026-09-30

Closed by the same two experiments. Thresholds are chosen on a held-out slice of the training period and
applied unchanged to each test period.

A bug worth keeping: the first version split the pool 80/20 **by row order**, and both corpora order
their pools by class, so the validation slice came out with **0.0% malware**. It would not have crashed;
it would have selected a meaningless threshold in silence. The split is now interleaved, and the exact
indices are exported from Julia rather than the rule re-derived in Python, with both sides asserting
their label vectors match row for row before anything is computed.

## 4. FAR is not reconstructible on LAMDA

Recorded in `research/b0-baseline/`. The published NEAR figure was recovered as a mean of per-year F1;
FAR was not. Nothing in this project reports a FAR mean, so this blocks only direct comparison against
the published FAR column.

**Fix:** ask the authors. Not resolvable from the release.

**Blocked with it:** pre-registered prediction 5, which needs Appendix F's strengthened/weakened/flipped
verdict counts — a detection count at two points in time per sample. `metadata.csv` carries one snapshot
per `sha256`, and there is no `vt_detections.csv` in the release despite an earlier note in this
repository claiming otherwise. `research/b3-label-drift/` measures what the release does support.

## 5. ~~Interpretability has no dataset control~~ — fixed 2026-09-30, and it RETRACTED the claim

Closed by `research/interp-dataset-control/`, and the criterion fixed before the run fired: a document
frequency difference with no model in it recovers **62 of the model's top-100** attributed features, and
chi-squared recovers 54, against 4 for chance. **No readability claim may appear anywhere.**

Two follow-ups settled it further. `research/interp-residual/` and
`research/interp-residual-retrain/` show the 62 overlap features carry essentially the whole model, while
the 38 the frequency count misses are where the model fails to generalise — removing them improves
drifted-period F1 by about 5 over a matched control on 2019–2022, though not on 2016–2018.

What replaces the readability claim is a drift-forensics one, and it is narrower: exact attribution
locates a prunable non-generalising subset from training data alone. It is not a recommendation, because
the effect reverses on three contiguous earlier periods.

## 6. ~~`tools/vast_setup.sh` points at a repository that does not exist~~ — fixed 2026-09-29

Its default `TMCYBER_REPO` was `https://github.com/Lukasz-G/TM-Cyber.git`, a 404. The repository now
exists as `https://github.com/Lukasz-G/tm-cyber-lab.git` (public, default branch `master`) and the
script points at it. Verified to resolve. Kept here rather than deleted so the entry does not read as
still open.
