# groundtruth-overlap — what the attributed features actually are

This is the fifth of the five interpretability measurements this project required before any claim about
what clauses mean, and the one that went unrun longest. It was singled out at the design stage as better
in this domain than in image or text tasks, *because the feature names carry meaning*.

**On the primary corpus they do not.** LAMDA ships its Drebin vocabulary stripped: the released parquet
columns are `feat_0 … feat_4560`. No semantic check is possible there, by anyone, with the public release.

APIGraph ships the real token names, and our packed matrix already carries them — 1,159 entries matching
the 2012 selected vocabulary exactly. So the measurement is available on the secondary corpus and nowhere
else.

## Why the vocabulary makes this a sharp test and not a browse

| Drebin category | n | share | |
|---|---|---|---|
| `activitylist_` | 615 | 53.1% | app identity |
| `requestedpermissionlist_` | 118 | 10.2% | **indicator** |
| `intentfilterlist_` | 117 | 10.1% | |
| `restrictedapilist_` | 96 | 8.3% | **indicator** |
| `broadcastreceiverlist_` | 58 | 5.0% | app identity |
| `servicelist_` | 52 | 4.5% | app identity |
| `usedpermissionslist_` | 32 | 2.8% | **indicator** |
| `urldomainlist_` | 23 | 2.0% | |
| `suspiciousapilist_` | 21 | 1.8% | **indicator** |
| `hardwarecomponentslist_` | 21 | 1.8% | |
| `contentproviderlist_` | 6 | 0.5% | app identity |

More than half the vocabulary is **Java class names of an application's own screens**. Those identify an
app; they tell a responder nothing transferable. The four categories that constitute an indicator of
compromise — a capability requested or an API reached for — are **23.0%** of the vocabulary.

So chance puts 23.0% of any feature set in the indicator classes and 63.1% in app identity. That fixes
what "enriched" means before anything is measured.

## Answer: 85% indicator, 1.7% app identity

Three seeds. Exact Shapley over 100 background and 100 explained rows from the 2012 pool.

| ranking | k | indicator % | app-identity % |
|---|---|---|---|
| **exact Shapley** | 20 | **85.0%** | **1.7%** |
| frequency | 20 | 85.0% | 5.0% |
| **exact Shapley** | 100 | **69.7%** | **15.0%** |
| frequency | 100 | 57.0% | 32.0% |
| exact Shapley | 500 | 40.4% | 44.1% |
| frequency | 500 | 38.8% | 43.8% |
| *base rate* | *all 1,159* | *23.0%* | *63.1%* |

**The top-20 is 85% indicator against a 23% base rate, and app-identity features are all but gone at
1.7% against 63.1%.** The attribution is not picking out which app this is; it is picking out what the app
asks to do.

## The one semantic advantage of the model over the corpus found in this project

At `k = 100` the attribution beats a plain frequency count on both halves: **+12.7 points of indicator
share** (69.7% against 57.0%) and **−17.0 points of app-identity pollution** (15.0% against 32.0%).

That matters because of what it sits beside. `interp-dataset-control/` retracted the readability claim
precisely because a frequency count recovered most of the attributed *set*. This measures something the
overlap count could not see: of the features each ranking selects, the attribution's are more often the
ones a responder could act on. The model is not finding different features so much as it is **ordering
them better**, and the ordering is what an analyst reads.

At `k = 500` the advantage vanishes (40.4% against 38.8%), which is expected and not a surprise: the
attribution support is only 313–321 of 1,159 features, so `k = 500` is mostly ties among zeros. The same
sparsity bound limits every top-`k` in this project above the support size.

## The named profile, which is the part a reviewer will read

Ten features carrying the most exact attribution, seed 1:

| | category | token |
|---|---|---|
| 1 | requestedpermission | `android.permission.send_sms` |
| 2 | suspiciousapi | `telephonymanager.getdeviceid` |
| 3 | usedpermission | `android.permission.send_sms` |
| 4 | suspiciousapi | `smsmanager.sendtextmessage` |
| 5 | intentfilter | `android.intent.action.boot_completed` |
| 6 | requestedpermission | `android.permission.read_phone_state` |
| 7 | requestedpermission | `android.permission.receive_sms` |
| 8 | urldomain | `91.213.175.176` |
| 9 | suspiciousapi | `telephonymanager.getsubscriberid` |
| 10 | usedpermission | `android.permission.internet` |

Device-identifier harvesting (IMEI at 2, IMSI at 9), SMS send and receive (1, 3, 4, 7), boot persistence
(5), network egress (10), and a hardcoded IP address (8). That is a premium-SMS fraud profile, and it is
coherent without anyone having told the model what coherent looks like.

**The frequency count's top ten is all permissions and APIs and contains no network indicator.** The
hardcoded IP at rank 8 is the kind of concrete artefact a responder can pivot on, and only the
attribution surfaces it in the top ten. One example is an anecdote, so this is reported as the
qualitative difference it is and no claim is built on it.

## What this licenses, and what it does not

- **The attributed features on APIGraph are indicators**, at 85% of the top 20 against a 23% base rate,
  with app identity suppressed from 63.1% to 1.7%. This is the first claim in the project about what
  clauses *mean* that clears a pre-fixed criterion.
- **Exact attribution has a semantic advantage over the corpus baseline** at `k = 100`, which the earlier
  set-overlap control could not detect and which does not reinstate any readability claim.
- It does **not** transfer to the primary corpus. LAMDA's vocabulary is not released, so the claim there
  remains the statistical one: the load-bearing features are malware-enriched at a median presence ratio
  above two. The paper's figure caption is corrected to say that and no more.
- It is **not** a rule-readability result. Individual features are interpretable here; the clauses that
  combine them are still thousands of literals wide, and `interp-dataset-control/` stands.
- No comparison against a curated indicator list (MITRE ATT&CK mobile techniques, Snort rules) is made.
  The category taxonomy is Drebin's own and is coarse. Matching tokens to documented techniques is a
  further step and would need an external list we have not assembled.

## Configuration

APIGraph 2012 pool, 30,533 rows, 10.0% malware, width 1,159. Flat FPTM, 20 clauses per class, `T` 10,
`S` 100, `L` 64, `LF` 10, 30 epochs, `LiteralCapped()`, `parallel = :none`, tm-lab **43dba5f**, 3 seeds.
Attribution over 100 background and 100 explained rows from the training pool; no test period is touched.
Category assignment is by the token's own Drebin prefix, so it involves no judgement on our part.

## Reproduce

```
julia --project=. -t 16 research/groundtruth-overlap/run.jl 3
```

About four minutes: three trainings on 30k rows at width 1,159, plus attribution.
