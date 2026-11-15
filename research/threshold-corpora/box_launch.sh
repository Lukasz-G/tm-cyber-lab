#!/usr/bin/env bash
# Full threshold-corpora run on the rented box, end to end, with no human step.
#
#   setsid nohup research/threshold-corpora/box_launch.sh > /workspace/box_launch.out 2>&1 &
#
# Phases, each ordered longest job first:
#   1. Julia: every FPTM unit (4 corpora x B0/U/O x seeds 1..5 = 60 single-threaded processes).
#   2. Python: run.py per corpus x condition (boosters: once under B0 and O, one fit per draw under U).
#   3. run.py --evaluate per corpus.
# Cores: at most CORES (default 40) in use: up to CORES Julia processes at one thread each, then PY_SLOTS
# Python jobs at CORES / PY_SLOTS booster threads each, then up to PY_SLOTS single-threaded evaluations.
# Memory: a job starts only when MemAvailable, minus the growth still to come in this script's running jobs
# (each job's need minus its current RSS), covers 1.25 x the measured peak of its job class (peak_ram.json).
# A job whose need exceeds MemTotal, or that waits 30 min with none of this script's jobs running, is
# recorded as failed and skipped. run.jl and run.py also gate themselves and skip verified units, so the
# script can be started again after an interruption and recomputes only what is missing.
# Log: one RESULT line per unit in /workspace/results-threshold.log, then FANOUT_DONE.
# It makes no vastai calls, never stops or destroys the machine, and never kills a process.
set -u
cd /workspace/tm-cyber
export PATH=/workspace/julia-1.11.9/bin:$PATH
# shellcheck disable=SC1091
source /workspace/venv/bin/activate
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CORES=${CORES:-40}
PY_SLOTS=${PY_SLOTS:-3}            # 3 x 13 threads: shortest bound on the Python phase (AndroZoo O vs total work)
THREADS=$((CORES / PY_SLOTS))
TREES=5000
LOG=/workspace/results-threshold.log
OUT=data/threshold-corpora/logs
mkdir -p "$OUT"
FAILURES=0

stamp() { date '+%F %T'; }
result() { echo "RESULT $* time=$(date '+%FT%T')" >> "$LOG"; }
need_mb() {
    python -c "import json, math, sys
P = json.load(open('research/threshold-corpora/peak_ram.json'))
lang, corpus, cond = sys.argv[1:]
print(math.ceil(1.25 * P.get(lang, {}).get(corpus, {}).get(cond, P['default'][lang])))" "$1" "$2" "$3"
}
avail_mb() { awk '/^MemAvailable:/ {printf "%d", $2 / 1024}' /proc/meminfo; }
total_mb() { awk '/^MemTotal:/ {printf "%d", $2 / 1024}' /proc/meminfo; }
verified() {  # verified <file>: the file and its .sha256 sidecar exist and agree
    [ -f "$1" ] && [ -f "$1.sha256" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$(tr -d ' \n' < "$1.sha256")" ]
}

declare -A RUN_NEED=() RUN_DESC=() RUN_START=()   # "=()" makes them set; bash 5.1 under set -u rejects ${#A[@]} otherwise
headroom_mb() {  # growth still to come in this script's running jobs
    local h=0 pid rss d
    for pid in "${!RUN_NEED[@]}"; do
        rss=$(awk '/^VmRSS:/ {printf "%d", $2 / 1024}' "/proc/$pid/status" 2>/dev/null || echo 0)
        d=$(( RUN_NEED[$pid] - rss ))
        [ "$d" -gt 0 ] && h=$((h + d))
    done
    echo "$h"
}

# run_pool <max parallel> <finish hook> <job> ...   where a job is "lang|corpus|cond|seed|command"
run_pool() {
    local max=$1 hook=$2; shift 2
    local queue=("$@") job lang c k s cmd need t_wait pid rc
    while [ "${#queue[@]}" -gt 0 ] || [ "${#RUN_NEED[@]}" -gt 0 ]; do
        for pid in "${!RUN_NEED[@]}"; do          # collect finished jobs
            if [ ! -d "/proc/$pid" ] || grep -q '^State:.*Z' "/proc/$pid/status" 2>/dev/null; then
                wait "$pid"; rc=$?
                "$hook" "${RUN_DESC[$pid]}" "$rc" $(( $(date +%s) - RUN_START[$pid] ))
                unset "RUN_NEED[$pid]" "RUN_DESC[$pid]" "RUN_START[$pid]"
            fi
        done
        if [ "${#queue[@]}" -gt 0 ] && [ "${#RUN_NEED[@]}" -lt "$max" ]; then
            job=${queue[0]}
            IFS='|' read -r lang c k s cmd <<< "$job"
            need=$(need_mb "$lang" "$c" "$k")
            if [ "$need" -gt "$(total_mb)" ]; then
                echo "$(stamp) $lang $c $k $s: need $need MB exceeds MemTotal"; "$hook" "$job" 99 0
                queue=("${queue[@]:1}"); t_wait=""; continue
            fi
            if [ $(( $(avail_mb) - $(headroom_mb) )) -ge "$need" ]; then
                eval "$cmd" > "$OUT/$lang-$c-$k-$s.log" 2>&1 &
                pid=$!
                RUN_NEED[$pid]=$need; RUN_DESC[$pid]=$job; RUN_START[$pid]=$(date +%s)
                echo "$(stamp) started $lang $c $k $s (need $need MB, pid $pid)"
                queue=("${queue[@]:1}"); t_wait=""
                sleep 5
                continue
            fi
            t_wait=${t_wait:-$(date +%s)}
            if [ "${#RUN_NEED[@]}" -eq 0 ] && [ $(( $(date +%s) - t_wait )) -gt 1800 ]; then
                echo "$(stamp) $lang $c $k $s: need $need MB not available after 30 min"; "$hook" "$job" 98 0
                queue=("${queue[@]:1}"); t_wait=""
            fi
        fi
        sleep 10
    done
}

julia_done() {  # one RESULT line per FPTM unit
    local lang c k s cmd rc=$2 secs=$3 ok=no
    IFS='|' read -r lang c k s cmd <<< "$1"
    verified "data/threshold-corpora/$c-$k-te0.10/fptm-seed$s.json" && ok=yes
    [ "$rc" -eq 0 ] && [ "$ok" = yes ] || FAILURES=$((FAILURES + 1))
    result "unit corpus=$c condition=$k model=fptm seed=$s exit=$rc verified=$ok seconds=$secs"
}

python_done() {  # one RESULT line per booster unit of the job, then one for the job
    local lang c k s cmd rc=$2 secs=$3 m seed ok seeds
    IFS='|' read -r lang c k s cmd <<< "$1"
    if [ "$k" = U ]; then seeds="1 2 3 4 5"; else seeds="once"; fi
    for m in lightgbm xgboost; do
        for seed in $seeds; do
            ok=no
            verified "data/threshold-corpora/$c-$k-te0.10/$m-seed$seed-t$TREES.npz" && ok=yes
            [ "$ok" = yes ] || FAILURES=$((FAILURES + 1))
            result "unit corpus=$c condition=$k model=$m seed=$seed trees=$TREES threads=$THREADS verified=$ok"
        done
    done
    [ "$rc" -eq 0 ] || FAILURES=$((FAILURES + 1))
    result "job run.py corpus=$c condition=$k exit=$rc seconds=$secs"
}

evaluate_done() {
    local lang c k s cmd rc=$2 secs=$3 ok=no
    IFS='|' read -r lang c k s cmd <<< "$1"
    verified "research/threshold-corpora/evaluation-$c-te0.10.txt" && ok=yes
    [ "$rc" -eq 0 ] && [ "$ok" = yes ] || FAILURES=$((FAILURES + 1))
    result "evaluate corpus=$c exit=$rc verified=$ok seconds=$secs"
}

echo "$(stamp) threshold-corpora full run: CORES=$CORES PY_SLOTS=$PY_SLOTS THREADS=$THREADS" | tee -a "$LOG"

# ---- phase 1: Julia units, longest first (measured load + one seed, 2026-11-13 timing runs)
JL_ORDER="androzoo:O lamda:O androzoo:B0 lamda:B0 hypercube:O lamda:U androzoo:U hypercube:U hypercube:B0 apigraph:O apigraph:B0 apigraph:U"
jobs=()
for ck in $JL_ORDER; do
    c=${ck%%:*}; k=${ck##*:}
    for s in 1 2 3 4 5; do
        jobs+=("julia|$c|$k|$s|julia --project=. research/threshold-corpora/run.jl $c $s $s --condition=$k")
    done
done
run_pool "$CORES" julia_done "${jobs[@]}"
echo "$(stamp) phase 1 (Julia) done"

# ---- phase 2: Python per corpus x condition, longest first, THREADS booster threads each
PY_ORDER="androzoo:O androzoo:U androzoo:B0 lamda:U lamda:B0 hypercube:U lamda:O hypercube:O hypercube:B0 apigraph:U apigraph:B0 apigraph:O"
jobs=()
for ck in $PY_ORDER; do
    c=${ck%%:*}; k=${ck##*:}
    jobs+=("python|$c|$k|all|BOOSTER_THREADS=$THREADS OMP_NUM_THREADS=$THREADS python -u -X utf8 research/threshold-corpora/run.py $c --condition=$k --threads=$THREADS")
done
run_pool "$PY_SLOTS" python_done "${jobs[@]}"
echo "$(stamp) phase 2 (Python) done"

# ---- phase 3: evaluation per corpus (single-threaded numpy)
jobs=()
for c in lamda androzoo hypercube apigraph; do
    jobs+=("python|$c|B0|evaluate|OMP_NUM_THREADS=1 python -u -X utf8 research/threshold-corpora/run.py $c --evaluate")
done
run_pool "$PY_SLOTS" evaluate_done "${jobs[@]}"
echo "$(stamp) phase 3 (evaluate) done"

echo "FANOUT_DONE failures=$FAILURES time=$(date '+%FT%T')" >> "$LOG"
echo "$(stamp) FANOUT_DONE failures=$FAILURES"
