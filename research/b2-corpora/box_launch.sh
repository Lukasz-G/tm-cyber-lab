#!/usr/bin/env bash
# The full b2-corpora run on a rented CPU box, end to end with no human step:
#   phase 1  run.jl per corpus (4 corpora, both configurations, B0/U/O, arms 2, 2r, 3, 1, 3n, 1n, 5, P)
#   phase 2  arms.py per corpus, arm and condition (2s, 4, 0)
#   phase 3  decide.py per corpus
#
#   setsid nohup bash research/b2-corpora/box_launch.sh >/dev/null 2>&1 </dev/null &
#
# Results log: $WORK/results-b2.log. Phases 1 and 2 log to their own files and are copied into it, with
# their FANOUT_ lines renamed PHASE1_ and PHASE2_; phase 3 appends to it, so its last line is FANOUT_DONE
# exactly once, after decide.py has run for every corpus.
#
# Cores: CORES (default 88) is the budget. Phase 1 runs the 4 corpora at once with CORES/4 Julia threads
# each; phase 2 runs CORES/JOBS arms.py processes at once with JOBS workers each, one thread per worker.
# Memory: every arm is named name@MB, so tools/fanout.sh starts it only when MemAvailable covers 1.25 x MB
# plus the MB of arms started in the last minute; run.jl and arms.py check memory again before every job.
# MB is the smoke's measured peak per process (Julia) or per worker x JOBS (Python), rounded up.
# Resumable: fanout.sh skips an arm already recorded as ARM_DONE in its phase log, and run.jl and
# arms.py skip every unit whose outputs and .sha256 sidecars verify. Re-running this script resumes.
# It makes no vast.ai call and kills no process.
set -uo pipefail

WORK="${WORK:-/workspace}"
REPO="${REPO:-$WORK/tm-cyber}"
CORES="${CORES:-88}"
JOBS="${JOBS:-8}"                                   # arms.py workers per process in phase 2
export PATH="$WORK/julia-1.11.9/bin:$PATH"
# shellcheck disable=SC1091
. "$WORK/venv/bin/activate"
cd "$REPO" || { echo "no repository at $REPO" >&2; exit 1; }

LOG="$WORK/results-b2.log"
LOG1="$WORK/results-b2-julia.log"
LOG2="$WORK/results-b2-arms.log"
THREADS_JL=$(( CORES / 4 ))
P2=$(( CORES / JOBS ))
PY="python -X utf8"
B=research/b2-corpora

# --- phase 1: Julia, the 4 corpora at once ----------------------------------------------------------
# peak per process: smoke 0.8-1.2 GB at 3 months; the full series holds every month's vectors, so x3
J=()
for c in lamda androzoo apigraph hypercube; do
    J+=("b2-julia-$c@4000|julia --project=. -t $THREADS_JL $B/run.jl $c")
done

# --- phase 2: arms.py, one process per corpus, arm and condition; the longest first ----------------
# Seconds per unit and peak per worker from the smoke (2026-11-13, contended laptop): AndroZoo 4c 620 s
# at 3.9 GB, LAMDA arm 0 455 s at 1.3 GB; AndroZoo arm 0 was not measured (about 1.5 GB from its arrays,
# gated at 2 GB). AndroZoo arm 4 runs its three conditions in one process, so its 3.9 GB workers stay at
# JOBS in number; arms.py gates every unit on memory as well.
mb() { echo $(( $1 * JOBS + 500 )); }               # JOBS workers at a per-worker peak, plus the parent
A=()
add() {                                             # corpus arm conds per-worker-MB
    A+=("b2-arms-$1-$2-${3//,/}@$(mb "$4")|$PY $B/arms.py $1 --arms $2 --conds $3 --jobs $JOBS")
}
add androzoo 4 B0,U,O 3900
for k in B0 U O; do add androzoo 0 $k 2000; done
for k in B0 U O; do add lamda 0 $k 1300; done
for k in B0 U O; do add lamda 4 $k 1200; done
for k in B0 U O; do add hypercube 4 $k 1100; done
for k in B0 U O; do add apigraph 4 $k 500; done
for k in B0 U O; do add hypercube 0 $k 1500; done
for k in B0 U O; do add apigraph 0 $k 900; done
for k in B0 U O; do add androzoo 2s $k 900; done
for k in B0 U O; do add lamda 2s $k 400; done
for k in B0 U O; do add hypercube 2s $k 400; done
for k in B0 U O; do add apigraph 2s $k 300; done

# --- phase 3: decide.py per corpus ------------------------------------------------------------------
D=()
for c in lamda apigraph androzoo hypercube; do
    D+=("b2-decide-$c@2000|$PY $B/decide.py $c")
done

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "phase 1: ${#J[@]} arms, parallelism 4, $THREADS_JL Julia threads each"; printf '  %s\n' "${J[@]}"
    echo "phase 2: ${#A[@]} arms, parallelism $P2, $JOBS workers each"; printf '  %s\n' "${A[@]}"
    echo "phase 3: ${#D[@]} arms"; printf '  %s\n' "${D[@]}"
    exit 0
fi

echo "b2-corpora box run: phase 1 $(date -u +%FT%TZ)"
THREADS=$THREADS_JL bash tools/fanout.sh "$LOG1" 4 "${J[@]}"
echo "b2-corpora box run: phase 2 $(date -u +%FT%TZ)"
THREADS=1 bash tools/fanout.sh "$LOG2" "$P2" "${A[@]}"
echo "b2-corpora box run: phase 3 $(date -u +%FT%TZ)"
{
    sed 's/^FANOUT_/PHASE1_/' "$LOG1"
    sed 's/^FANOUT_/PHASE2_/' "$LOG2"
} > "$LOG"
THREADS=1 bash tools/fanout.sh "$LOG" 4 "${D[@]}"
