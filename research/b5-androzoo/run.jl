# QUESTION:  is a twenty-clause flat FPTM still competitive with gradient boosting when the feature
#            regime changes completely — 16,978 features at 39% density, against APIGraph's 1,159 at
#            1.75% and LAMDA's 4,561 at 3%?
# SURPRISE:  yes. The drift-robustness claim already died on APIGraph, so the surviving claim is
#            "competitive at a few hundred literals", and THIS is the regime that should break it if
#            anything does. At 3% density a negated literal is satisfied by 97% of rows, so it is
#            nearly free and clauses accumulate them cheaply; at 39% density that stops being true,
#            and the sibling project's whole account of how fuzzy clauses behave — tolerance tails
#            built from cheap negated literals — assumes the sparse case. This is the first test of
#            that account outside it.
# ARMS:      four. FPTM at S = 372 (s = width/S held at LAMDA's 45.6) and at S = 100 (s left to move),
#            against LightGBM and XGBoost on identical rows. The two FPTM arms are not optional: on
#            APIGraph they were what ruled out s as the explanation for a negative result, and here the
#            width change is 15x larger so s moves 15x further.
# PASS/FAIL: COMPETITIVE if FPTM is within 5 F1 of the BEST booster on both test years. Not "within 5
#            of LightGBM" — on APIGraph the two boosters diverged by up to 12 points and comparing
#            against the weaker one would have turned a failure into an apparent partial success.
#            NARROWS otherwise, and then the competitive claim is scoped to sparse inputs.
#
#   julia --project=. -t 16 research/b5-androzoo/run.jl [nseeds]
#
# Only two test years exist (2020, 2021), so this is a weaker test than APIGraph's six. Say so.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: clause_literals
using TMCore
using Random, Printf, Statistics, JSON

const DIR = joinpath("data", "apigraph", "tmx", "androzoo")
const YEARS = 2020:2021
const CLAUSES, T, L, LF = 20, 10, 64, 10
const EPOCHS = 30
const PARALLEL = :none
const TM_LAB_COMMIT = "43dba5f"

# LAMDA: width 4,561 at S = 100, so s = 45.61. At width 16,978, holding s constant needs S = 372.
const S_MATCHED = 372
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

function density(X::Vector{TMInput})
    n = min(500, length(X))
    tot = 0
    for i in 1:n
        for c in X[i].chunks
            tot += count_ones(c)
        end
    end
    return tot / (n * length(X[1]))
end

function main()
    nseeds = isempty(ARGS) ? 10 : parse(Int, ARGS[1])
    feats = read_features(joinpath(DIR, "features.arrow"))
    width = length(feats)
    Xtr, Ytr, _ = load("train")
    @printf("tm-lab pin %s  threads %d  parallel %s\n", TM_LAB_COMMIT, Threads.nthreads(), PARALLEL)
    @printf("AndroZoo: width %d (%d literals)  train %d rows, %d malware (%.1f%%)  density %.3f\n",
            width, 2width, length(Ytr), count(Ytr), 100count(Ytr) / length(Ytr), density(Xtr))
    @printf("s = width/S:  S=%d -> s=%.1f (matched to LAMDA)   S=%d -> s=%.1f\n\n",
            S_MATCHED, width / S_MATCHED, S_SAME, width / S_SAME)

    evals = [string(y) => load_year(y) for y in YEARS]
    for (k, (X, Y)) in evals
        @printf("  %s: %d rows, %d malware (%.1f%%)\n", k, length(Y), count(Y), 100count(Y) / length(Y))
    end

    base = isfile("research/b5-androzoo/baselines.json") ?
           JSON.parsefile("research/b5-androzoo/baselines.json") : Dict()
    isempty(base) && println("\nWARNING: baselines.json missing — run baselines.py first")

    results = Dict{Int,Dict{String,Vector{Float64}}}()
    diag = Dict{Int,Tuple{Float64,Float64}}()
    for S in (S_MATCHED, S_SAME)
        per = Dict{String,Vector{Float64}}(k => Float64[] for (k, _) in evals)
        meds = Float64[]; negs = Float64[]
        for seed in 1:nseeds
            m = TMClassifier([false, true], width;
                             clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
            for e in 1:EPOCHS
                train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=PARALLEL)
            end
            for (k, (X, Y)) in evals
                push!(per[k], f1_of(predict(m, X), Y))
            end
            counts = Int[]; pos = 0; neg = 0
            for bank in (m.positive[1], m.negative[1]), j in 1:bank.nclauses
                lits = clause_literals(bank, j)
                isempty(lits) && continue
                push!(counts, length(lits))
                p = count(l -> !l[2], lits); pos += p; neg += length(lits) - p
            end
            push!(meds, isempty(counts) ? 0.0 : sort(counts)[cld(length(counts), 2)])
            push!(negs, neg / max(1, pos + neg))
        end
        results[S] = per
        diag[S] = (median(meds), median(negs))
        @printf("\nS=%d (s=%.1f)  %s\n", S, width / S,
                join([@sprintf("%s %.2f±%.2f", k, mean(per[k]), std(per[k])) for (k, _) in evals], "  "))
    end

    println("\n=== FPTM vs gradient boosting on identical rows ===")
    @printf("%-6s %16s %16s %10s %10s %12s\n",
            "year", "FPTM s-matched", "FPTM S=100", "LightGBM", "XGBoost", "vs best")
    gaps = Float64[]
    for (k, _) in evals
        a = results[S_MATCHED][k]; b = results[S_SAME][k]
        lgb = haskey(base, "lightgbm") ? get(base["lightgbm"], k, NaN) : NaN
        xgb = haskey(base, "xgboost") ? get(base["xgboost"], k, NaN) : NaN
        best = max(lgb, xgb)
        gap = max(mean(a), mean(b)) - best      # kindest FPTM arm against the strongest booster
        push!(gaps, gap)
        @printf("%-6s %8.2f±%-7.2f %8.2f±%-7.2f %10.2f %10.2f %+12.2f\n",
                k, mean(a), std(a), mean(b), std(b), lgb, xgb, gap)
    end

    @printf("\ngap to the best booster: %s  (criterion: within 5 on both years)\n",
            join([@sprintf("%+.2f", g) for g in gaps], ", "))
    for S in (S_MATCHED, S_SAME)
        md, ng = diag[S]
        @printf("S=%-4d median literals/clause %.0f  (LF/literals %.1f%%)  negated %.1f%%\n",
                S, md, 100LF / md, 100ng)
    end
    println("  for contrast: LAMDA at 3% density gave 69 literals, 14.5%, 83.8% negated")

    competitive = !any(isnan, gaps) && all(g -> g >= -5.0, gaps)
    for (k, _) in evals
        @printf("RESULT fptm_smatched_%s=%.2f fptm_s100_%s=%.2f\n",
                k, mean(results[S_MATCHED][k]), k, mean(results[S_SAME][k]))
    end
    println("RESULT b5_androzoo=", competitive ? "COMPETITIVE" : "NARROWS")
end

main()
