# QUESTION:  does flat FPTM at ~20 clauses per class match gradient boosting on LAMDA, on IID, NEAR
#            and per-year FAR 2018-2022?
# SURPRISE:  yes. Nobody has run a Tsetlin machine on LAMDA. And the interesting outcome is not "does
#            it win" but the SHAPE over time: our LightGBM is non-monotone in year (F1 28.3 in 2018
#            against 72.2 in 2019), so whether a 20-clause model tracks that shape or a different one
#            is informative either way. A model that degrades differently is a finding; a model that
#            degrades identically says the drift is in the data rather than the hypothesis class.
# ARMS:      flat FPTM at 3 seeds, against LightGBM and XGBoost trained and evaluated by us on the
#            IDENTICAL row sets (research/b0-baseline). The comparison is deliberately NOT against the
#            published figures: our LightGBM reproduces their IID to 0.00 but exceeds their NEAR by
#            12.6, so their NEAR/FAR set definition is not unambiguously reconstructible from the
#            release. Comparing like-for-like on sets we define removes that confound entirely, and
#            the relationship between our LightGBM and theirs is reported separately.
# PASS/FAIL: PASS if FPTM is within 3 F1 points of our LightGBM on IID *and* its mean absolute gap
#            across NEAR and the five FAR years is within 10 points. FAIL → stop and reconsider the
#            domain, per the project's standing gate. Do NOT raise the clause budget to pass.
#
#   julia --project=. -t 16 research/b1-gate/run.jl
#
# Threads affect `predict` only, which is bit-identical across thread counts; training is
# parallel = :none because :classes caps at the class count (two) and :clauses is slower than serial
# at this clause count on this width.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: clause_literals
using TMCore
using Random, Printf, Statistics

const TMX_DIR = joinpath("data", "lamda", "tmx")
const TRAIN_YEARS = (2013, 2014)
const NEAR_YEARS = (2016, 2017)
const FAR_YEARS = (2018, 2019, 2020, 2021, 2022)

const CLAUSES, T, S, L, LF = 20, 10, 100, 64, 10
const EPOCHS = 30
const SEEDS = Tuple(1:(isempty(ARGS) ? 3 : parse(Int, ARGS[1])))
const PARALLEL = :none
const TM_LAB_COMMIT = "43dba5f"

# Our own LightGBM / XGBoost on these exact row sets (research/b0-baseline/results.txt), pooled over
# train+test portions of each year. Published figures for reference: IID 97.49, NEAR 59.48, FAR 47.24.
const OURS_LGB = Dict("IID" => 97.49, "NEAR" => 72.12,
                      "2018" => 28.32, "2019" => 72.22, "2020" => 67.77,
                      "2021" => 53.87, "2022" => 64.90)
const OURS_XGB = Dict("IID" => 97.18, "NEAR" => 71.46,
                      "2018" => 25.31, "2019" => 70.73, "2020" => 65.89,
                      "2021" => 56.25, "2022" => 62.22)

function portion(year, which)
    meta = read_meta(joinpath(TMX_DIR, "$year.meta.arrow"))
    idx = which === :all ? collect(eachindex(meta.file_split)) :
          [i for i in eachindex(meta.file_split) if meta.file_split[i] == String(which)]
    inputs, _ = read_inputs(joinpath(TMX_DIR, "$year.tmx"); rows=idx)
    return inputs, Bool[meta.label[i] for i in idx]
end

function concat(pairs)
    X = TMInput[]; Y = Bool[]
    for (xs, ys) in pairs
        append!(X, xs); append!(Y, ys)
    end
    return X, Y
end

function f1_of(pred, y)
    tp = count(pred .& y); fp = count(pred .& .!y); fn = count(.!pred .& y)
    tn = count(.!pred .& .!y)
    prec = tp + fp == 0 ? 0.0 : tp / (tp + fp)
    rec = tp + fn == 0 ? 0.0 : tp / (tp + fn)
    f1 = prec + rec == 0 ? 0.0 : 2prec * rec / (prec + rec)
    return (f1=100 * f1, precision=100 * prec, recall=100 * rec,
            fnr=100 * (1 - rec), fpr=100 * (fp / max(1, fp + tn)))
end

function main()
    @printf("tm-lab pin %s   threads %d   parallel %s   ceiling LiteralCapped\n",
            TM_LAB_COMMIT, Threads.nthreads(), PARALLEL)
    @printf("CLAUSES %d (per class, both polarities)  T %d  S %d  L %d  LF %d  epochs %d  seeds %s\n\n",
            CLAUSES, T, S, L, LF, EPOCHS, string(SEEDS))

    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    Xtr, Ytr = concat([portion(y, :train) for y in TRAIN_YEARS])
    evals = Pair{String,Tuple{Vector{TMInput},Vector{Bool}}}[]
    push!(evals, "IID" => concat([portion(y, :test) for y in TRAIN_YEARS]))
    push!(evals, "NEAR" => concat([portion(y, :all) for y in NEAR_YEARS]))
    for y in FAR_YEARS
        push!(evals, string(y) => portion(y, :all))
    end
    @printf("train %d rows   eval sets: %s\n\n", length(Ytr),
            join([@sprintf("%s(%d)", k, length(v[2])) for (k, v) in evals], " "))

    per_seed = Dict{String,Vector{Float64}}(k => Float64[] for (k, _) in evals)
    diag = NamedTuple[]

    for seed in SEEDS
        m = TMClassifier([false, true], length(feats);
                         clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
        t = @elapsed for e in 1:EPOCHS
            train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=PARALLEL)
        end
        line = @sprintf("seed %d  trained %.0fs  ", seed, t)
        for (k, (X, Y)) in evals
            s = f1_of(predict(m, X), Y)
            push!(per_seed[k], s.f1)
            line *= @sprintf("%s %.2f  ", k, s.f1)
        end
        println(line)

        counts = Int[]; pos = 0; neg = 0
        for bank in (m.positive[1], m.negative[1]), j in 1:bank.nclauses
            lits = clause_literals(bank, j)
            isempty(lits) && continue
            push!(counts, length(lits))
            p = count(l -> !l[2], lits); pos += p; neg += length(lits) - p
        end
        med = sort(counts)[cld(length(counts), 2)]
        push!(diag, (median_literals=med, lf_ratio=LF / med, neg_frac=neg / (pos + neg)))
    end

    println("\n=== flat FPTM vs our own gradient boosting, identical row sets ===")
    @printf("%-6s %14s %10s %10s %10s %10s\n", "set", "FPTM (mean±sd)", "LightGBM", "XGBoost", "vs LGB", "vs XGB")
    gaps = Float64[]
    for (k, _) in evals
        v = per_seed[k]
        mu, sd = mean(v), length(v) > 1 ? std(v) : 0.0
        lgb, xgb = OURS_LGB[k], OURS_XGB[k]
        @printf("%-6s %8.2f±%-5.2f %10.2f %10.2f %+10.2f %+10.2f\n", k, mu, sd, lgb, xgb, mu - lgb, mu - xgb)
        k == "IID" || push!(gaps, abs(mu - lgb))
    end

    iid_gap = abs(mean(per_seed["IID"]) - OURS_LGB["IID"])
    drift_gap = mean(gaps)
    @printf("\nIID gap to our LightGBM: %.2f  (criterion: <= 3)\n", iid_gap)
    @printf("mean |gap| across NEAR + 5 FAR years: %.2f  (criterion: <= 10)\n", drift_gap)

    med = Int(median([d.median_literals for d in diag]))
    @printf("\ndiagnostics: median literals/clause %d  LF/literals %.2f%%  negated %.1f%%\n",
            med, 100 * median([d.lf_ratio for d in diag]), 100 * median([d.neg_frac for d in diag]))

    pass = iid_gap <= 3.0 && drift_gap <= 10.0
    for (k, _) in evals
        @printf("RESULT fptm_%s=%.2f\n", lowercase(k), mean(per_seed[k]))
    end
    @printf("RESULT iid_gap=%.2f drift_gap=%.2f median_literals=%d\n", iid_gap, drift_gap, med)
    println("RESULT b1_gate=", pass ? "PASS" : "FAIL")
    return pass
end

main()
