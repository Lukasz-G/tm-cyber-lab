#!/usr/bin/env bash
# Launcher for research/threshold-corpora on a rented Linux box.
#
# Runs every unit (corpus, condition, model, seed). run.jl and run.py skip a unit whose output and matching
# .sha256 sidecar exist, so this launcher can be rerun after any interruption and recomputes only what is
# missing. Before each job it waits until MemAvailable (/proc/meminfo) covers 1.25 x the measured peak RAM
# of that job class (peak_ram.json), re-checking every 60 s; it fails after 30 min of waiting, or at once if
# the need exceeds MemTotal. It never stops or destroys the machine.
#
#   research/threshold-corpora/launch.sh [corpus ...]          default: lamda apigraph androzoo hypercube
#   JOBS=5 STAGGER=30 THREADS=60 PYTHON=python3 research/threshold-corpora/launch.sh lamda
#
# JOBS     FPTM seed processes run in parallel (seeds 2..5; seed 1 runs first because it writes common.json)
# STAGGER  seconds between starting two seed processes, so each one has loaded its data before the next
#          memory check
# THREADS  booster n_jobs for run.py (default: all cores, since one Python job runs at a time); recorded in
#          every results file
set -u
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python3}
JOBS=${JOBS:-5}
THREADS=${THREADS:-$(nproc)}        # booster n_jobs: the cores allotted to the one Python job running at a time
export BOOSTER_THREADS=$THREADS
STAGGER=${STAGGER:-30}
CORPORA=${*:-lamda apigraph androzoo hypercube}
LOG=data/threshold-corpora/logs
mkdir -p "$LOG"

need_mb() {   # 1.25 x the measured peak of (language, corpus, condition), in MB
    "$PY" -c "import json, math, sys
P = json.load(open('research/threshold-corpora/peak_ram.json'))
lang, corpus, cond = sys.argv[1:]
print(math.ceil(1.25 * P.get(lang, {}).get(corpus, {}).get(cond, P['default'][lang])))" "$1" "$2" "$3"
}

avail_mb() { awk '/^MemAvailable:/ {printf "%d", $2 / 1024}' /proc/meminfo; }

total_mb() { awk '/^MemTotal:/ {printf "%d", $2 / 1024}' /proc/meminfo; }

wait_mem() {  # wait_mem <need MB> <job label>; fails after 30 min, or at once if the need exceeds MemTotal
    local need=$1 job=$2 a t0
    if [ "$need" -gt "$(total_mb)" ]; then
        echo "memory check $job: need $need MB exceeds total memory $(total_mb) MB"; exit 1
    fi
    t0=$(date +%s)
    while :; do
        if [ $(( $(date +%s) - t0 )) -gt 1800 ]; then
            echo "memory check $job: need $need MB not available after 30 min"; exit 1
        fi
        a=$(avail_mb)
        if [ "$a" -ge "$need" ]; then
            echo "$(date '+%F %T') memory check $job: need $need MB, available $a MB"
            return
        fi
        echo "$(date '+%F %T') memory check $job: need $need MB, available $a MB; waiting 60s"
        sleep 60
    done
}

for c in $CORPORA; do
    for cond in B0 U O; do
        nj=$(need_mb julia "$c" "$cond")
        wait_mem "$nj" "julia $c $cond seed 1"
        julia --project=. research/threshold-corpora/run.jl "$c" 1 1 --condition="$cond" >> "$LOG/julia-$c-$cond-seed1.log" 2>&1 \
            || { echo "julia $c $cond seed 1 failed, see $LOG"; exit 1; }
        running=0
        for s in 2 3 4 5; do
            wait_mem "$nj" "julia $c $cond seed $s"
            julia --project=. research/threshold-corpora/run.jl "$c" "$s" "$s" --condition="$cond" >> "$LOG/julia-$c-$cond-seed$s.log" 2>&1 &
            running=$((running + 1))
            if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
            sleep "$STAGGER"
        done
        wait
        np=$(need_mb python "$c" "$cond")
        wait_mem "$np" "python $c $cond"
        "$PY" -u -X utf8 research/threshold-corpora/run.py "$c" --condition="$cond" >> "$LOG/python-$c-$cond.log" 2>&1 \
            || { echo "python $c $cond failed, see $LOG"; exit 1; }
    done
    "$PY" -u -X utf8 research/threshold-corpora/run.py "$c" --evaluate >> "$LOG/evaluate-$c.log" 2>&1 \
        || { echo "evaluate $c failed, see $LOG"; exit 1; }
done
echo "$(date '+%F %T') all units done for: $CORPORA"
