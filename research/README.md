# research/

**[INDEX.md](INDEX.md) lists every experiment with its verdict** and flags which earn their place.
Start there.

Experiments. **Not a package** — scripts here may depend on anything, and are not held to the API
stability or test coverage expected of library code.

Each directory holds `run.jl` or `run.py`, its raw output in `results.txt`, and a `README.md` giving
the question and the answer. Negative results are kept, not deleted.

## Required script header

Every `run.*` starts with these four lines, filled in **before the run**. They are not
documentation; they are the thing that stops the experiment being worthless.

```
# QUESTION:  one sentence.
# SURPRISE:  what result here would surprise someone who has read the papers?
# ARMS:      every arm, including the control that isolates the mechanism.
# PASS/FAIL: the criterion, and what happens if it fails.
```

**If nothing in `SURPRISE` would surprise a paper-reader, do not run the experiment.** In the
sibling project roughly two thirds of one session's compute went to confirming published results,
including a ~2.5 CPU-hour sweep whose headline was a relation quoted in the header of the very
script that "discovered" it. That more clauses help, that `T` scales with clause count, that
booleanization matters, and that drift degrades malware detectors are all published. Calibration
needs one cheap point, not a swept grid with seeds.

**`ARMS` must name three, not two.** Every two-arm comparison in the sibling project turned out to
be confounded — three times — and the isolating third arm changed the conclusion each time, once
killing a premise and not a tuning.

Two specific controls, both learned the hard way:

- **Dead channel.** Any arm *wider* than its control needs an arm padded with always-zero bits at
  the same width. An always-zero literal is satisfied for free by its negation, so a clause includes
  it at no evaluation cost, which inflates its literal count and moves the `L` growth gate. On
  CIFAR-10, 32 dead bits were worth +0.0109 on their own. A random-content channel controls for
  content; only a zero channel controls for the budget effect.
- **`s = width/S`.** Comparisons at differing input widths must hold `s` constant by rescaling `S`.
  In one measurement it accounted for more of the apparent effect than the thing being measured.
  LAMDA's 925 / 4,561 / 25,460 feature variants are three different widths.

## Reporting rules

- **Quote final accuracy, not best.** One variant looked like −0.005 on peak and was −0.035 on
  final.
- **Per-year, never a FAR mean.** LAMDA's 2024 and 2025 malware counts are 794 and 23 against
  ~45,000 benign per year. That is AV label lag, not drift, and a mean across 2018–2025 reports the
  lag as a result.
- **State the scope inside the claim** — "on LAMDA 2013–2022 at 20 clauses", not "about FPTM". A
  single-dataset positive is provisional, and says so in the write-up, not afterwards.
- **Record the tm-lab commit** next to any number that depends on the machine's behaviour. The
  pinned commit is in `julia/bootstrap.jl`; a run against a local working copy via `TM_LAB_PATH` is
  not reproducible from this repository and must say so.
- **Say which clause-vote ceiling policy was used.** The FPTM paper, the reference implementation
  and the optimised fork disagree, and `TMCore` can express all of them. Two results reporting
  "FPTM" are not necessarily reporting the same model.

## Data

Datasets live in gitignored `/data/`. Derived `.tmx` matrices and their Arrow sidecars are written
there by the Python pipeline and read by Julia — see `docs/matrix-format.md`. A matrix is never
rewritten in place; deriving a subset writes a new file, so a result always names the exact input
that produced it.
