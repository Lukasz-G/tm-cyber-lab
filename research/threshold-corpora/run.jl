# QUESTION:  how much does the decision-threshold rule change reported F1 on every corpus, when one code
#            path, one split and one set of rules is used on all of them, and does the training class
#            balance change the answer?
# SURPRISE:  predicted before the run, on every corpus and for all three models:
#            (1) the oracle gap (oracle F1 minus F1 at the validation threshold) is small on the
#                in-distribution split and larger on later years, and on AndroZoo it grows from 2020 to 2021;
#                it differs between FPTM and the boosters as on LAMDA in research/threshold-bias;
#            (2) the training condition moves the default operating point and not the best F1 or AP. Prior
#                evidence is research/b5-imbalance (FPTM only, APIGraph only): balancing moved default recall
#                from 55.7 to 75.5 while best swept F1 stayed at 74.69 against 74.09. Predicted here for all
#                three models: oracle F1 and AP do not change between B0, U and O; default minus validation
#                changes; oracle minus validation does not change.
#            Predictions (1) and (2) and both PASS/FAIL readings were reworded on 2026-11-13 for the B0/U/O
#            design, after a code review and after AndroZoo results from the superseded B0/B1/B2/W design had
#            been read. No threshold of the earlier wording was changed. The two clauses of (1) about AndroZoo's
#            growth from 2020 to 2021 and about FPTM against the boosters were in the first version of this
#            header (2026-11-13 00:16 UTC), before any AndroZoo result was read (02:46 UTC); readings (1b) and
#            (1c), which test them, were added on 2026-11-13 with their numbers fixed before any full result
#            is read.
# CONDITIONS (the same text in run.jl, run.py and research/b2-corpora): identical for FPTM, LightGBM and
#            XGBoost; no class weights, sample weights or scale_pos_weight in any model.
#              B0  the training split cut to exactly 1:9 on every corpus, LAMDA included: k = min(M, G div 9)
#                  malware rows and 9k goodware rows, drawn at random without replacement (the rule of
#                  research/b2-corpora). No other resampling. Every set meant to be at 10% (the B0 validation
#                  set, validation_te, the in-distribution split and every test year) is cut by the same rule,
#                  with a cap folded in as k = min(M, G div 9, cap div 10).
#              U   B0 with the majority class undersampled at random, without replacement, to the minority
#                  count: 50/50, size 2 x minority.
#              O   B0 with the minority class oversampled to the majority count: each minority row repeated
#                  floor(R) times (R = majority/minority), plus round((R - floor R) x minority) further
#                  minority rows drawn without replacement: 50/50, size 2 x majority. At exactly 10% that is
#                  x9.
#            Majority and minority are decided per training set, not assumed to be goodware and malware.
#            Seed s draws the U and O resample s and trains model s, for FPTM, LightGBM and XGBoost alike,
#            5 seeds. B0 has no resampling, and O on an exact 1:9 set draws no extra rows (R = 9), so under
#            both the training rows are identical for every seed: their boosters are trained once (they are
#            deterministic) and recorded as such. U boosters get 5 seeds = 5 draws. FPTM keeps 5 seeds in
#            every condition.
# ARMS:      the three conditions B0, U and O, test share 10% throughout. U against B0 changes the class
#            share by removing majority rows; O against B0 changes it by repeating minority rows; U against O
#            separates the two routes to the same 50/50 share. Each condition runs FPTM (margins exported
#            here), LightGBM and XGBoost (run.py). Each model is read through six threshold rules: oracle
#            (F1-maximising on the scored period's labels), default (the model's own decision), and two
#            validation-based rules computed on each of two validation sets. The first validation set is built
#            like the condition's training set; the second is the validation region resampled to the test
#            share. The rules are validation and validation_te (the F1-maximising threshold on each set,
#            applied unchanged), and rate and rate_te (the threshold that gives the predicted-malware share
#            that the corresponding validation threshold gives on its set). Both are reported; neither is
#            preferred.
# PASS/FAIL: not a gate; per corpus and model, evaluated under each condition and under each validation rule
#            (validation and validation_te), every combination reported. A difference COUNTS when it exceeds
#            1.0 F1 point and its 95% paired-bootstrap interval excludes zero (the deciding criterion, below).
#            Reading (1) SURVIVES for a combination if the oracle gap on the in-distribution split is at most
#            2.0 F1 points and its mean over the later years exceeds that in-distribution gap by a difference
#            that counts. It is NOT SUPPORTED if the difference does not count (roughly constant), and
#            UNRESOLVED if the in-distribution gap exceeds 2.0.
#            Reading (1b), AndroZoo only: the oracle gap in 2021 exceeds the oracle gap in 2020 by a difference
#            that counts; it SURVIVES for a condition and validation rule if this holds for all three models.
#            Reading (1c), every corpus: the mean later-year oracle gap of each booster exceeds FPTM's by a
#            difference that counts, the direction research/threshold-bias found on LAMDA (FPTM 7.07 against
#            18.97 and 17.50); it SURVIVES for a condition and validation rule if this holds for both boosters.
#            Reading (2) SURVIVES for a corpus and model if, on every period, the differences between B0, U
#            and O in oracle F1, in AP and in oracle minus validation do not count, while the difference in
#            default minus validation counts on at least one period. It is NOT SUPPORTED otherwise.
#            Two noise criteria are computed for every difference and printed side by side. Stated: the
#            difference exceeds 1.0 F1 point and the range across the 5 seeds or draws of each side (a booster
#            trained once under B0 or O has no range). Bootstrap: the difference exceeds 1.0 F1 point and its
#            95% interval from a paired bootstrap over the shared test rows (2,000 resamples; scores averaged
#            over seeds or draws for every model; thresholds fixed on the validation sets) excludes zero. The
#            bootstrap criterion decides every reading and both controls, for every model (decided 2026-11-13
#            by Lukasz before any full result was read); the stated criterion is printed beside it as
#            secondary.
#            Every F1 is reported with its rule and with AP beside it. No mean across years is reported except
#            as a labelled summary column.
# CONTROLS:  negative: on the in-distribution split at 1:9 the oracle gap is near zero (reading (1): at most
#            2.0 F1 points; under the bootstrap criterion, its upper 95% bound at most 2.0), with a verdict
#            line for every condition, rule and model. Positive, a planted prior shift with no covariate
#            drift: the in-distribution 1:9
#            set cut to exactly 1:1 from the same rows (all its malware rows and as many of its goodware rows,
#            drawn without replacement), scored from the margins and probabilities already computed, with no
#            extra training. Expected, fixed before the run: with the threshold chosen on the 1:9 validation
#            set (validation_te), the oracle gap on the 1:1 set exceeds the oracle gap on the 1:9
#            in-distribution set by a difference that counts under the bootstrap criterion (PASS/FAIL),
#            for every model and condition. The same comparison with the condition's own validation set (validation) is
#            reported beside it; under U and O that set is already 1:1, so there it is not a planted shift. A
#            control that fails is reported as a failure, and nothing is tuned to make it pass.
# EXPLORATORY (no pass/fail): the EM prior correction of Saerens, Latinne, Decaestecker, Neural
#            Computation 14(1), 2002, computed in --evaluate from the saved scores, with no retraining. Per
#            test period, starting from pi = pi_train (the condition's training share: 0.1 under B0, 0.5 under
#            U and O), every probability p is rescaled by the prior ratio, p' = (pi/pi_train) p /
#            [(pi/pi_train) p + ((1-pi)/(1-pi_train)) (1-p)], and pi is set to mean(p'), until |delta pi| <
#            1e-6 or 1,000 iterations. Reported per period: the estimated pi, and F1 of p' at 0.5 and at t_val,
#            the F1-optimal threshold chosen on validation in p-space at pi_train and applied unchanged to p'
#            (the label-free recalibration). This second rule replaces, before any full result, a vacuous one
#            that mapped the threshold through the same prior ratio as p and so reproduced the validation rule
#            exactly. The
#            boosters use their probabilities (the mean over draws under U); FPTM's seed-mean margin is first
#            Platt-scaled, fitted on the validation set built like the condition's training set.
# RESUMABLE: every unit (corpus, condition, model, seed) writes its output to a temporary file, renames it,
#            then writes a .sha256 sidecar. A unit whose output and matching sidecar exist, whose common hash
#            matches the current common content, and whose training rows, validation rows and settings hash
#            to the recorded values, is skipped on start. run.py reads only verified files.
# MEMORY:    before each job starts, MemAvailable (/proc/meminfo on Linux, the operating system's free-memory
#            figure elsewhere) must cover 1.25 x the measured peak RAM of that job class (peak_ram.json, from
#            the full-size timing runs); otherwise the job waits and re-checks every 60 s. It fails with an
#            error after 30 min of waiting, and at once if the need exceeds the machine's total memory.
#
# Construction, identical on every corpus:
#   pool         the corpus's pool period (table CORPORA below), concatenated in file order, at its released
#                share; every 10% set is cut from its region by the exact 1:9 rule (CONDITIONS);
#   split        by pool position: i mod 20 in {5,12,19} -> validation region, {2,9,16} -> in-distribution
#                region, the other 14 -> training split (70 / 15 / 15); robust to pools ordered by class;
#   test         the in-distribution region and every later year cut to exactly 1:9 (--test-share stays at
#                its only supported value, 0.10), years capped at 25,000 rows (k <= 2,500);
#   FPTM         20 clauses per class, T 10, L 64, LF 10, 30 epochs, LiteralCapped(), training parallel mode
#                :none, tm-lab 43dba5f; S per corpus so that TMCore's s = round(width/S) is 46 on every corpus
#                (LAMDA 100, APIGraph 25, AndroZoo 372, Hypercube 89); width, S and s are written into every
#                output; margin = score(malware) - score(goodware), default decision margin > 0 (ties go to
#                goodware, as in TMCore.predict).
# Every seed file carries the hash of the common content (positions, labels, configuration) and run.py
# refuses a seed file whose hash differs from common.json, or whose sidecar does not match.
#
#   julia --project=. research/threshold-corpora/run.jl <corpus> <seed_from> <seed_to> --condition=B0|U|O [--test-share=0.10] [--smoke]
#   research/threshold-corpora/launch.sh [corpus ...]   (the launcher used on the rented box)
#   python -X utf8 research/threshold-corpora/run.py <corpus> --condition=.. [--test-share=..] [--smoke]

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
using .Tmx: read_inputs, read_meta, read_features
using TMCore
using Random, Printf, Statistics, JSON, SHA

const CLAUSES, T, L, LF = 20, 10, 64, 10
const EPOCHS = 30
const TM_LAB_COMMIT = "43dba5f"
const CEILING = "LiteralCapped"
const PARALLEL = :none
const MAX_EVAL = 25_000
const SMOKE_POOL, SMOKE_EVAL = 6_000, 3_000
const SHARE_SEED = 31337
const VAL_RES, IID_RES = (5, 12, 19), (2, 9, 16)
const CONDITIONS = ("B0", "U", "O")
const PEAK_FILE = joinpath(@__DIR__, "peak_ram.json")
const PEAK = JSON.parsefile(PEAK_FILE)          # measured peak RAM per job class, bound at load
const MEM_FACTOR, MEM_POLL_S, MEM_MAX_WAIT_S = 1.25, 60, 1800

"MemAvailable in MB: /proc/meminfo on Linux, the operating system's free-memory figure elsewhere."
function mem_available_mb()
    if Sys.islinux() && isfile("/proc/meminfo")
        for ln in eachline("/proc/meminfo")
            startswith(ln, "MemAvailable:") && return parse(Int, split(ln)[2]) / 1024
        end
    end
    return Sys.free_memory() / 2^20
end

"MemTotal in MB: /proc/meminfo on Linux, the operating system's figure elsewhere."
function mem_total_mb()
    if Sys.islinux() && isfile("/proc/meminfo")
        for ln in eachline("/proc/meminfo")
            startswith(ln, "MemTotal:") && return parse(Int, split(ln)[2]) / 1024
        end
    end
    return Sys.total_memory() / 2^20
end

"Block until MemAvailable covers 1.25 x the measured peak of this job class; fail after 30 min or if it never can."
function wait_for_memory(job, lang, corpus, cond)
    peak = get(get(get(PEAK, lang, Dict()), corpus, Dict()), cond, PEAK["default"][lang])
    need = MEM_FACTOR * peak
    need <= mem_total_mb() || error("memory check $job: need $(round(Int, need)) MB exceeds total memory $(round(Int, mem_total_mb())) MB")
    t_wait = time()
    while true
        time() - t_wait > MEM_MAX_WAIT_S && error("memory check $job: need $(round(Int, need)) MB not available after 30 min")
        avail = mem_available_mb()
        @printf("memory check %s: need %.0f MB (1.25 x %.0f), available %.0f MB%s\n", job, need, peak, avail,
                avail >= need ? "" : "; waiting $(MEM_POLL_S)s")
        flush(stdout)
        avail >= need && return
        sleep(MEM_POLL_S)
    end
end

"Write `content` to a temporary file, rename it into place, then write the .sha256 sidecar."
function write_verified(path, content::AbstractString)
    tmp = path * ".tmp"
    write(tmp, content)
    mv(tmp, path; force=true)
    write(path * ".sha256", bytes2hex(sha256(read(path))))
end

"The vendored tm-lab commit actually loaded (the git HEAD of the tree TMCore is loaded from); refuses a mismatch."
function check_tm_lab()
    root = normpath(joinpath(dirname(pathof(TMCore)), "..", "..", ".."))
    head = try
        strip(read(`git -C $root rev-parse HEAD`, String))
    catch e
        error("cannot read the tm-lab commit at $root: $e")
    end
    startswith(head, TM_LAB_COMMIT) || error("tm-lab at $root is at $head, expected $TM_LAB_COMMIT")
    return String(head)
end

"Hash of a seed's training and validation rows and of the training settings that the common hash does not cover."
rows_hash(tr, va, seed) = bytes2hex(sha256(string(tr, "|", va, "|epochs=", EPOCHS, "|rng=MersenneTwister(1000*", seed,
                                                  "+epoch)|parallel=", PARALLEL)))

"True when `path` and its sidecar exist and the sidecar matches the file's SHA-256."
verified(path) = isfile(path) && isfile(path * ".sha256") &&
                 strip(read(path * ".sha256", String)) == bytes2hex(sha256(read(path)))

# One entry per corpus; adding a corpus means adding a line here and nothing else (run.py reads the
# directory and the stems from the exported common.json). Each directory holds <stem>.tmx,
# <stem>.meta.arrow (with a `label` column) and features.arrow. `monthly` corpora store one file per
# month as YYYY-MM; the others store one file per year. S is chosen so that round(width/S) = 46.
const CORPORA = Dict(
    "lamda"     => (dir = joinpath("data", "lamda", "tmx"), pool = ["2013", "2014"], years = 2016:2022, monthly = false, S = 100),
    "apigraph"  => (dir = joinpath("data", "apigraph", "tmx", "apigraph"), pool = ["train"], years = 2013:2018, monthly = true, S = 25),
    "androzoo"  => (dir = joinpath("data", "apigraph", "tmx", "androzoo"), pool = ["train"], years = 2020:2021, monthly = true, S = 372),
    "hypercube" => (dir = joinpath("data", "hypercube", "tmx"), pool = ["train"], years = 2022:2023, monthly = true, S = 89),
)

"File stems that make up one year, in the order run.py concatenates them."
year_stems(c, y) = c.monthly ?
    [s for s in (@sprintf("%d-%02d", y, m) for m in 1:12) if isfile(joinpath(c.dir, "$s.tmx"))] :
    [string(y)]

"Concatenate the stems: inputs and labels in file order, row order within each file."
function load_stems(c, stems)
    X = TMInput[]; Y = Bool[]
    for s in stems
        xs, _ = read_inputs(joinpath(c.dir, "$s.tmx"))
        meta = read_meta(joinpath(c.dir, "$s.meta.arrow"))
        append!(X, xs); append!(Y, Bool.(collect(meta.label)))
    end
    return X, Y
end

"""
    exact19(Y, idx; cap, seed) -> positions

The exact 1:9 cut of research/b2-corpora: k = min(M, G div 9) malware rows and 9k goodware rows of the
positions `idx` (into `Y`), drawn at random without replacement; with a cap, k = min(M, G div 9, cap
div 10). Deterministic in `seed`. Returns sorted positions and asserts the 1:9 ratio.
"""
function exact19(Y::Vector{Bool}, idx::Vector{Int}; cap=nothing, seed::Int)
    mal = [i for i in idx if Y[i]]; good = [i for i in idx if !Y[i]]
    M, G = length(mal), length(good)
    k = cap === nothing ? min(M, G ÷ 9) : min(M, G ÷ 9, cap ÷ 10)
    k > 0 || error("cannot cut to 1:9: $M malware, $G goodware")
    rng = MersenneTwister(seed)
    out = sort(vcat(mal[randperm(rng, M)[1:k]], good[randperm(rng, G)[1:(9k)]]))
    @assert count(Y[out]) == k && count(.!Y[out]) == 9k
    return out
end

class_counts(Y, pos) = (count(Y[pos]), count(.!Y[pos]))

"Positions of the minority and majority class within `pos` (ties: malware is the minority)."
function minority_majority(Y, pos)
    mal = [i for i in pos if Y[i]]; good = [i for i in pos if !Y[i]]
    return length(mal) <= length(good) ? (mal, good) : (good, mal)
end

"U: the majority class undersampled at random, without replacement, to the minority count."
function undersample(Y, pos, rng)
    mi, ma = minority_majority(Y, pos)
    return sort(vcat(mi, ma[randperm(rng, length(ma))[1:length(mi)]]))
end

"""
O: the minority class oversampled to the majority count. Each minority row is repeated floor(R) times
(R = majority/minority), plus round((R - floor R) x minority) further minority rows drawn without
replacement.
"""
function oversample(Y, pos, rng)
    mi, ma = minority_majority(Y, pos)
    R = length(ma) / length(mi)
    k = floor(Int, R)
    extra = round(Int, (R - k) * length(mi))
    return sort(vcat(ma, repeat(mi, k), mi[randperm(rng, length(mi))[1:extra]]))
end

"The training and validation positions of one seed under a condition."
function condition_rows(cond, Y, tr_b0, val_b0, seed)
    cond == "B0" && return tr_b0, val_b0
    f = cond == "U" ? undersample : oversample
    return f(Y, tr_b0, MersenneTwister(SHARE_SEED + 1000 + seed)), f(Y, val_b0, MersenneTwister(SHARE_SEED + 2000 + seed))
end

margins(m, X) = Float64[score(m, 2, x) - score(m, 1, x) for x in X]

flag(name, default) = (a = filter(s -> startswith(s, "--$name="), ARGS); isempty(a) ? default : String(split(a[1], "=")[2]))

"Directory tag shared with run.py."
run_tag(cname, cond, te, smoke) = "$cname-$cond-te$te" * (smoke ? "-smoke" : "")

function main()
    pos_args = filter(s -> !startswith(s, "--"), ARGS)
    length(pos_args) >= 3 || error("usage: run.jl <corpus> <seed_from> <seed_to> --condition=B0|U|O [--test-share=0.10] [--smoke]")
    cname = pos_args[1]
    s_from, s_to = parse(Int, pos_args[2]), parse(Int, pos_args[3])
    smoke = "--smoke" in ARGS
    cond = flag("condition", "")
    cond in CONDITIONS || error("--condition must be one of $(CONDITIONS)")
    te_arg = flag("test-share", "0.10")
    te_arg == "0.10" || error("--test-share: only 0.10 is supported (every 10% set is cut by the exact 1:9 rule)")
    c = CORPORA[cname]
    isdir(c.dir) || error("corpus directory $(c.dir) does not exist")
    out_dir = joinpath("data", "threshold-corpora", run_tag(cname, cond, te_arg, smoke))
    mkpath(out_dir)
    wait_for_memory("julia $cname $cond", "julia", cname, cond)
    tm_head = check_tm_lab()
    width = length(read_features(joinpath(c.dir, "features.arrow")))
    S = c.S
    s_machine = round(Int, width / S)

    # --- pool: concatenation of the pool stems at the released share; the smoke run takes a 1:9 subsample
    Xp, Yp = load_stems(c, c.pool)
    all_pos = collect(1:length(Yp))
    base = smoke ? exact19(Yp, all_pos; cap=SMOKE_POOL, seed=SHARE_SEED + 7) : all_pos

    # --- split by position within `base`, then cut every 10% set to exactly 1:9
    region(res) = [base[k] for k in eachindex(base) if (k % 20) in res]
    val_reg, iid_reg = region(VAL_RES), region(IID_RES)
    tr_reg = [base[k] for k in eachindex(base) if !((k % 20) in VAL_RES) && !((k % 20) in IID_RES)]
    tr_b0 = exact19(Yp, tr_reg; seed=SHARE_SEED + 1)
    val_b0 = exact19(Yp, val_reg; seed=SHARE_SEED + 4)
    val_te = val_b0      # the validation region at 1:9 is B0's validation set; U and O resample from it
    iid_pos = exact19(Yp, iid_reg; seed=SHARE_SEED + 3)
    # positive control: the 1:9 iid set cut to exactly 1:1 from the same rows (indices into iid_pos)
    iid_mal = [j for j in eachindex(iid_pos) if Yp[iid_pos[j]]]
    iid_good = [j for j in eachindex(iid_pos) if !Yp[iid_pos[j]]]
    k11 = min(length(iid_mal), length(iid_good))
    rng11 = MersenneTwister(SHARE_SEED + 5)
    iid11_idx = sort(vcat(iid_mal[randperm(rng11, length(iid_mal))[1:k11]],
                          iid_good[randperm(rng11, length(iid_good))[1:k11]]))

    @printf("%s condition=%s test-share=%s  width %d  S %d  s %d  pool %d rows (%.2f%% malware)\n",
            cname, cond, te_arg, width, S, s_machine, length(Yp), 100mean(Yp))
    for (nm, pos) in (("training region", tr_reg), ("B0 train", tr_b0), ("validation region", val_reg),
                      ("B0 val = val_te", val_b0), ("iid region", iid_reg), ("iid", iid_pos),
                      ("iid 1:1 control", iid_pos[iid11_idx]))
        m_, g_ = class_counts(Yp, pos)
        @printf("  %-18s %7d malware %8d goodware\n", nm, m_, g_)
    end
    @printf("clauses=%d T=%d S=%d L=%d LF=%d epochs=%d  ceiling=%s  parallel=%s  tm-lab %s%s\n",
            CLAUSES, T, S, L, LF, EPOCHS, CEILING, PARALLEL, TM_LAB_COMMIT, smoke ? "  SMOKE" : "")
    flush(stdout)

    # --- shared evaluation sets: val_te, iid, then each later year
    evals = Dict{String,Tuple{Vector{TMInput},Vector{Bool}}}()
    evals["val_te"] = (Xp[val_te], Yp[val_te])
    evals["iid"] = (Xp[iid_pos], Yp[iid_pos])
    year_pos = Dict{String,Vector{Int}}()
    for y in c.years
        Xy, Yy = load_stems(c, year_stems(c, y))
        kp = exact19(Yy, collect(1:length(Yy)); cap=smoke ? SMOKE_EVAL : MAX_EVAL, seed=SHARE_SEED + y)
        evals[string(y)] = (Xy[kp], Yy[kp])
        year_pos[string(y)] = kp
        @printf("  %-18s %7d malware %8d goodware   (of %d rows, %d malware)\n", string(y), count(Yy[kp]),
                count(.!Yy[kp]), length(Yy), count(Yy))
    end
    flush(stdout)

    # --- common content and its hash; every process computes it, the owner of seed 1 writes it
    config = Dict("clauses" => CLAUSES, "T" => T, "S" => S, "s" => s_machine, "width" => width, "L" => L,
                  "LF" => LF, "epochs" => EPOCHS, "ceiling" => CEILING, "parallel" => string(PARALLEL),
                  "tm_lab" => TM_LAB_COMMIT, "tm_lab_head" => tm_head, "max_eval" => smoke ? SMOKE_EVAL : MAX_EVAL,
                  "share_seed" => SHARE_SEED)
    ykeys = sort(collect(keys(year_pos)))
    canon = string(cname, "|", cond, "|", te_arg, "|", smoke, "|", base, "|", tr_b0, "|", val_b0, "|", val_te, "|",
                   iid_pos, "|", iid11_idx, "|", [year_pos[k] for k in ykeys], "|", Int.(Yp[base]), "|",
                   sort(collect(config), by=first))
    chash = bytes2hex(sha256(canon))
    common_path = joinpath(out_dir, "common.json")
    common_ok = verified(common_path) && JSON.parsefile(common_path)["common_hash"] == chash
    if s_from == 1 && !common_ok   # only the owner of seed 1 writes it, so parallel processes never race
        write_verified(common_path, JSON.json(Dict(
                "corpus" => cname, "dir" => replace(c.dir, "\\" => "/"), "condition" => cond,
                "test_share" => te_arg, "smoke" => smoke, "common_hash" => chash, "config" => config,
                "pool_stems" => c.pool, "years" => collect(c.years),
                "year_stems" => Dict(string(y) => year_stems(c, y) for y in c.years),
                "pool_rows" => length(Yp), "pool_share" => mean(Yp), "base_rows" => length(base), "base_share" => mean(Yp[base]),
                "train_region_counts" => collect(class_counts(Yp, tr_reg)), "val_region_counts" => collect(class_counts(Yp, val_reg)),
                "iid_region_counts" => collect(class_counts(Yp, iid_reg)),
                "b0_train_pos" => tr_b0, "b0_val_pos" => val_b0, "val_te_pos" => val_te, "iid_pos" => iid_pos,
                "iid11_idx" => iid11_idx,
                "year_pos" => year_pos,
                "labels" => merge(Dict(k => Int.(v[2]) for (k, v) in evals),
                                  Dict("b0_train" => Int.(Yp[tr_b0]), "b0_val" => Int.(Yp[val_b0]),
                                       "iid11" => Int.(Yp[iid_pos[iid11_idx]]))),
            )))
        @printf("  wrote common.json (hash %s)\n", chash[1:12])
    elseif common_ok
        @printf("  common.json verified (hash %s)\n", chash[1:12])
    end

    for seed in s_from:s_to
        seed_path = joinpath(out_dir, "fptm-seed$seed.json")
        tr_pos, val_pos = condition_rows(cond, Yp, tr_b0, val_b0, seed)
        rhash = rows_hash(tr_pos, val_pos, seed)
        if verified(seed_path)
            sf = JSON.parsefile(seed_path)
            if sf["common_hash"] == chash && get(sf, "rows_hash", "") == rhash
                @printf("  seed %d  skipped: verified output exists\n", seed)
                flush(stdout)
                continue
            end
        end
        t0 = time()
        Xtr, Ytr = Xp[tr_pos], Yp[tr_pos]
        m = TMClassifier([false, true], width; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
        for e in 1:EPOCHS
            train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=PARALLEL)
        end
        ttrain = time() - t0
        mg = Dict(k => margins(m, v[1]) for (k, v) in evals)
        mg["val"] = margins(m, Xp[val_pos])
        write_verified(seed_path, JSON.json(Dict("seed" => seed, "common_hash" => chash, "rows_hash" => rhash, "margins" => mg,
                                "train_seconds" => ttrain, "train_pos" => tr_pos, "val_pos" => val_pos,
                                "train_labels" => Int.(Ytr), "val_labels" => Int.(Yp[val_pos]),
                                "train_unique" => length(unique(tr_pos)), "maxrss_bytes" => Sys.maxrss())))
        @printf("  seed %d  train rows %d (%.1f%% malware, %d unique)  val rows %d  train %.1fs  total %.1fs  maxrss %.0f MB\n",
                seed, length(tr_pos), 100mean(Ytr), length(unique(tr_pos)), length(val_pos), ttrain, time() - t0,
                Sys.maxrss() / 2^20)
        flush(stdout)
    end
end

main()
