# QUESTION:  does flat FPTM's drift advantage over gradient boosting replicate on a second dataset?
# SURPRISE:  yes, either way, and this is the experiment the whole project's credibility rests on. In
#            the sibling algorithm project THREE OF FOUR comparable positives died on their second
#            dataset, so the prior is genuinely against replication. Two things change at once here:
#            the dataset (APIGraph, width 1,159 against LAMDA's 4,561) and the operating point (9.6%
#            malware against ~37%), the latter being closer to deployment than LAMDA's deliberate
#            balance.
# ARMS:      four, and the third exists to remove a confound that has bitten this project's sibling:
#              1. FPTM at S = 25   -- s = width/S held constant against the LAMDA run (4561/100 = 45.6)
#              2. FPTM at S = 100  -- same S as the LAMDA run, so s differs
#              3. LightGBM         -- identical rows (research/b5-apigraph/baselines.py)
#              4. XGBoost          -- identical rows
#            Arms 1 and 2 together separate "the dataset behaves differently" from "s changed". With
#            only one FPTM arm, any difference from the LAMDA result would be uninterpretable, because
#            s and the dataset would have moved together.
# PASS/FAIL: REPLICATES if FPTM is within 5 F1 of gradient boosting on the earliest test year and
#            ahead of it on a majority of the later years -- the qualitative pattern found on LAMDA.
#            FAILS TO REPLICATE otherwise, in which case the LAMDA drift-robustness result is reported
#            as a LAMDA result and the paper's claim narrows accordingly. Do not retune to rescue it.
#
#   julia --project=. -t 16 research/b5-apigraph/run.jl [nseeds]
#
# Protocol is the dataset's own: train on the 2012 pool, test on each later year. Absolute F1 is NOT
# comparable to the LAMDA numbers -- different width, different base rate, different protocol. Only
# the gap to gradient boosting on identical rows is.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: clause_literals
using TMCore
using Random, Printf, Statistics, JSON

const DIR = joinpath("data", "apigraph", "tmx", "apigraph")
const YEARS = 2013:2018
const CLAUSES, T, L, LF = 20, 10, 64, 10
const EPOCHS = 30
const PARALLEL = :none
const TM_LAB_COMMIT = "43dba5f"

# s = width/S. The LAMDA run used width 4,561 at S = 100, so s = 45.61. Holding s constant at width
# 1,159 requires S = 1159/45.61 = 25.4 -> 25. The S = 100 arm keeps S instead and lets s move, which
# is what makes the pair interpretable.
const S_MATCHED = 25
const S_SAME = 100

function load(stem)
    inputs, h = read_inputs(joinpath(DIR, "$stem.tmx"))
    meta = read_meta(joinpath(DIR, "$stem.meta.arrow"))
    return inputs, collect(Bool.(meta.label)), h
end

function load_year(year)
    X = TMInput[]; Y = Bool[]
    for m in 1:12
        stem = @sprintf("%d-%02d", year, m)
        isfile(joinpath(DIR, "$stem.tmx")) || continue
        xs, ys, _ = load(stem)
        append!(X, xs); append!(Y, ys)
    end
    return X, Y
end

function f1_of(pred, y)
    tp = count(pred .& y); fp = count(pred .& .!y); fn = count(.!pred .& y)
    prec = tp + fp == 0 ? 0.0 : tp / (tp + fp)
    rec = tp + fn == 0 ? 0.0 : tp / (tp + fn)
    return prec + rec == 0 ? 0.0 : 100 * 2prec * rec / (prec + rec)
end

function main()
    nseeds = isempty(ARGS) ? 10 : parse(Int, ARGS[1])
    feats = read_features(joinpath(DIR, "features.arrow"))
    width = length(feats)
    Xtr, Ytr, _ = load("train")
    @printf("tm-lab pin %s  threads %d  parallel %s\n", TM_LAB_COMMIT, Threads.nthreads(), PARALLEL)
    @printf("APIGraph: width %d (%d literals)  train %d rows, %d malware (%.1f%%)\n",
            width, 2width, length(Ytr), count(Ytr), 100count(Ytr) / length(Ytr))
    @printf("s = width/S:  S=%d -> s=%.1f (matched to the LAMDA run)   S=%d -> s=%.1f\n\n",
            S_MATCHED, width / S_MATCHED, S_SAME, width / S_SAME)

    evals = [string(y) => load_year(y) for y in YEARS]
    for (k, (X, Y)) in evals
        @printf("  %s: %d rows, %d malware (%.1f%%)\n", k, length(Y), count(Y), 100count(Y) / length(Y))
    end

    base = isfile("research/b5-apigraph/baselines.json") ?
           JSON.parsefile("research/b5-apigraph/baselines.json") : Dict()
    isempty(base) && println("\nWARNING: baselines.json missing — run baselines.py first")

    results = Dict{Int,Dict{String,Vector{Float64}}}()
    diag = Dict{Int,Float64}()
    for S in (S_MATCHED, S_SAME)
        per = Dict{String,Vector{Float64}}(k => Float64[] for (k, _) in evals)
        meds = Float64[]
        for seed in 1:nseeds
            m = TMClassifier([false, true], width;
                             clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
            for e in 1:EPOCHS
                train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=PARALLEL)
            end
            for (k, (X, Y)) in evals
                push!(per[k], f1_of(predict(m, X), Y))
            end
            counts = Int[]
            for bank in (m.positive[1], m.negative[1]), j in 1:bank.nclauses
                l = length(clause_literals(bank, j)); l > 0 && push!(counts, l)
            end
            push!(meds, isempty(counts) ? 0.0 : sort(counts)[cld(length(counts), 2)])
        end
        results[S] = per
        diag[S] = median(meds)
        @printf("\nS=%d (s=%.1f)  %s\n", S, width / S,
                join([@sprintf("%s %.2f±%.2f", k, mean(per[k]), std(per[k])) for (k, _) in evals], "  "))
    end

    println("\n=== FPTM vs gradient boosting on identical rows ===")
    @printf("%-6s %16s %16s %10s %10s %9s\n", "year", "FPTM S=25", "FPTM S=100", "LightGBM", "XGBoost", "vs LGB")
    wins = 0; total = 0; first_gap = NaN
    for (k, _) in evals
        a = results[S_MATCHED][k]; b = results[S_SAME][k]
        lgb = haskey(base, "lightgbm") ? get(base["lightgbm"], k, NaN) : NaN
        xgb = haskey(base, "xgboost") ? get(base["xgboost"], k, NaN) : NaN
        gap = mean(a) - lgb
        @printf("%-6s %8.2f±%-7.2f %8.2f±%-7.2f %10.2f %10.2f %+9.2f\n",
                k, mean(a), std(a), mean(b), std(b), lgb, xgb, gap)
        if !isnan(gap)
            total += 1
            k == string(first(YEARS)) ? (first_gap = gap) : (gap > 0 && (wins += 1))
        end
    end

    @printf("\nfirst test year gap to LightGBM: %+.2f  (criterion: within 5)\n", first_gap)
    @printf("later years where FPTM is ahead: %d of %d  (criterion: majority)\n", wins, total - 1)
    @printf("median literals/clause: S=25 %.0f  S=100 %.0f  (LF/literals %.1f%% and %.1f%%)\n",
            diag[S_MATCHED], diag[S_SAME], 100LF / diag[S_MATCHED], 100LF / diag[S_SAME])

    replicates = !isnan(first_gap) && abs(first_gap) <= 5.0 && wins > (total - 1) / 2
    for (k, _) in evals
        @printf("RESULT fptm_s25_%s=%.2f fptm_s100_%s=%.2f\n",
                k, mean(results[S_MATCHED][k]), k, mean(results[S_SAME][k]))
    end
    println("RESULT b5_apigraph=", replicates ? "REPLICATES" : "DOES_NOT_REPLICATE")
end

main()
