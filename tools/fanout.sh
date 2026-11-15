#!/usr/bin/env bash
# Run many independent experiment arms at once on a CPU-rich box, writing every result to one
# small log as it lands.
#
#   setsid nohup bash tools/fanout.sh /workspace/b1.log 48 \
#       "lf10_s100_seed1|julia --project=. research/b1-gate/run.jl --LF 10 --S 100 --seed 1" \
#       "lf10_s100_seed2|julia --project=. research/b1-gate/run.jl --LF 10 --S 100 --seed 2" \
#       >/dev/null 2>&1 </dev/null &
#
# PARALLELISM is the number of arms in flight. THREADS (optional, via the THREADS env var,
# default 1) is how many Julia threads each arm gets; keep PARALLELISM * THREADS at or below the
# core count.
#
# Why this shape. TMCore threads inside a run too, but not usefully at the sizes this project
# trains at. `train!(parallel=:classes)` reached 1.8x on binary and `parallel=:clauses` 2.3x, both
# only at large clause counts -- at the twenty-clause gate config `:clauses` measured FIVE TIMES
# SLOWER than serial. Batch `predict` is the exception at ~6x, bit-identically. So the cores go to
# arms, not to runs: seeds, hyperparameter points, per-month attribution, per-year retraining
# schedules. Raise THREADS only when fewer independent arms remain than there are cores -- the tail
# of a sweep, or one large attribution job. Since predict-threading is bit-identical, that choice
# moves the wall clock and cannot move a result.
#
# What it guarantees:
#   * one line per finished arm in RESULTS, appended the moment it finishes, so a box that dies
#     costs only the arms in flight -- the numbers are kilobytes and the checkpoints are not
#   * an arm already recorded in RESULTS is skipped, so a re-run resumes
#   * a failed arm is recorded as ARM_FAILED and does not stop the others
#   * FANOUT_DONE is the last line, so a watcher can block on it and wake a human once
#   * an arm named "name@MB" starts only when MemAvailable (/proc/meminfo) covers 1.25 x MB plus the
#     MB of every arm started in the last 60 s, which may not have reached its peak yet; otherwise the
#     launcher waits and re-checks every 30 s. MB is the measured peak resident memory of that job class.
#     Where /usr/bin/time exists, each arm's own peak is recorded in its ARM_DONE line as maxrss_mb.
#
# Reading the log: a retried arm leaves its old ARM_FAILED line in place above its ARM_DONE, on
# purpose -- the history of what failed is worth keeping. Take the LAST record for each arm name.
#
# What it deliberately does NOT do: destroy the instance. Automation pulls and reports; a human
# destroys. An idle box costs cents an hour; a destroyed one costs the whole run.
set -uo pipefail

RESULTS="${1:?usage: fanout.sh RESULTS_LOG PARALLELISM "name|command" ...}"
P="${2:?usage: fanout.sh RESULTS_LOG PARALLELISM "name|command" ...}"
shift 2
[ "$#" -gt 0 ] || { echo "no arms given" >&2; exit 2; }

LOGDIR="$(dirname "$RESULTS")/arm-logs"
mkdir -p "$LOGDIR"
touch "$RESULTS"

# Pin the thread count explicitly. `predict` threads unconditionally, so without this every Julia
# process grabs the whole machine and P of them trample each other.
export JULIA_NUM_THREADS="${THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

if command -v nproc >/dev/null 2>&1; then
    cores=$(nproc)
    if [ $((P * JULIA_NUM_THREADS)) -gt "$cores" ]; then
        echo "warning: parallelism $P x threads $JULIA_NUM_THREADS exceeds $cores cores" >&2
    fi
fi

say() {
    # Concurrent appends of one short line are atomic enough on Linux, but flock costs nothing
    # and this log is the only thing guaranteed to survive the box.
    if command -v flock >/dev/null 2>&1; then
        # util-linux flock -c takes exactly one string, so pass the line through sh's own arguments
        flock "$RESULTS" sh -c 'printf "%s\n" "$1" >> "$2"' _ "$1" "$RESULTS"
    else
        printf '%s\n' "$1" >> "$RESULTS"
    fi
}

run_arm() {
    local name="$1" cmd="$2" log="$LOGDIR/$1.log" t0 rc rss=""
    t0=$(date +%s)
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -f 'FANOUT_MAXRSS_KB %M' -o "$log.rss" bash -c "$cmd" > "$log" 2>&1
        rc=$?
        rss=$(awk '/^FANOUT_MAXRSS_KB/ {printf "maxrss_mb=%d", $2 / 1024}' "$log.rss" 2>/dev/null)
    elif bash -c "$cmd" > "$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    local secs=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ]; then
        # The arm owns its own reporting: any line it prints as "RESULT ..." is carried into the
        # results log. Everything else stays in the per-arm log, which is expendable.
        local payload
        payload=$(grep -h '^RESULT' "$log" | tr '\n' ';' | sed 's/;$//')
        say "ARM_DONE $name ${secs}s ${rss} ${payload:-<no RESULT line>}"
    else
        say "ARM_FAILED $name ${secs}s rc=$rc ${rss} (see $log)"
    fi
}

mem_available_mb() {
    # MemAvailable where the kernel reports it, MemFree otherwise, -1 when neither can be read
    if [ -r /proc/meminfo ]; then
        awk '/^MemAvailable:/ {a = $2} /^MemFree:/ {f = $2}
             END {if (a) printf "%d", a / 1024; else if (f) printf "%d", f / 1024; else printf "-1"}' /proc/meminfo
    else
        printf -- "-1"
    fi
}

STARTS=()   # start time and peak MB of every arm launched, for the reservation of recent starts
wait_for_memory() {
    local name="$1" peak="$2" need avail recent i now
    [ "$peak" -gt 0 ] 2>/dev/null || return 0
    need=$(( (peak * 125 + 99) / 100 ))
    while :; do
        now=$(date +%s)
        recent=0
        for i in "${STARTS[@]}"; do
            [ $(( now - ${i%% *} )) -lt 60 ] && recent=$(( recent + ${i##* } ))
        done
        avail=$(mem_available_mb)
        if [ "$avail" -lt 0 ]; then
            echo "memory: cannot read available memory; starting $name unchecked" >&2
            return 0
        fi
        [ "$avail" -ge $(( need + recent )) ] && return 0
        echo "memory: $name needs ${need} MB + ${recent} MB reserved, ${avail} MB available; waiting" >&2
        sleep 30
    done
}

say "FANOUT_START $(date -u +%FT%TZ) arms=$# parallelism=$P threads=$JULIA_NUM_THREADS host=$(hostname) cores=$(nproc)"

running=0
queued=0
for spec in "$@"; do
    name="${spec%%|*}"
    cmd="${spec#*|}"
    if [ "$name" = "$cmd" ]; then
        say "ARM_FAILED $name 0s rc=2 (spec has no | separator)"
        continue
    fi
    peak=0
    if [ "${name#*@}" != "$name" ]; then
        peak="${name##*@}"
        name="${name%@*}"
    fi
    if grep -q "^ARM_DONE $name " "$RESULTS" 2>/dev/null; then
        echo "skip $name (already done)"
        continue
    fi
    wait_for_memory "$name" "$peak"
    STARTS+=("$(date +%s) $peak")
    run_arm "$name" "$cmd" &
    queued=$((queued + 1))
    running=$((running + 1))
    if [ "$running" -ge "$P" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait

done_n=$(grep -c '^ARM_DONE ' "$RESULTS" 2>/dev/null || true)
fail_n=$(grep -c '^ARM_FAILED ' "$RESULTS" 2>/dev/null || true)
say "FANOUT_DONE $(date -u +%FT%TZ) launched=$queued done=${done_n:-0} failed=${fail_n:-0}"
