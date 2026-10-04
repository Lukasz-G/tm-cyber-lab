# QUESTION:  where does the gap to gradient boosting on APIGraph come from — is FPTM losing recall,
#            losing precision, or is the twenty-clause budget simply binding?
# SURPRISE:  yes on the decomposition. The B5 runs recorded only F1, which cannot distinguish "misses
#            malware" from "cries wolf", and those have completely different causes and fixes. The
#            boosters on these datasets run at 91-97% precision and 34-48% recall, i.e. they buy F1 by
#            being extremely conservative; whether FPTM sits at a different point on that trade-off is
#            unmeasured and is the first thing to know.
# ARMS:      the gate budget (20 clauses) against a larger one (200), at fixed T and at T rescaled for
#            the larger budget. Three arms because clause count and T are coupled -- the published
#            relation is T ~ sqrt(CLAUSES/2 * LF) -- so raising clauses alone changes two things at
#            once, and a two-arm version could not tell which mattered.
# PASS/FAIL: not a gate. This is DIAGNOSIS, and it explicitly does NOT revisit the B5 verdict: that
#            replication was run at the pre-registered configuration and stands as run. Whether a
#            tuned model closes the gap is a different question from whether the LAMDA result
#            replicated at matched settings, and conflating them would be retrofitting.
#
#   julia --project=. -t 16 research/b5-diagnosis/run.jl [nseeds]
#
# APIGraph only: it is the cheapest of the three datasets (30k train rows, width 1,159) and the one
# where the replication failed by the largest margin, so it is where diagnosis is most informative per second.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: clause_literals
using TMCore
using Random, Printf, Statistics, JSON

const DIR = joinpath("data", "apigraph", "tmx", "apigraph")
const YEARS = 2013:2018
const L, LF, S = 64, 10, 25
const EPOCHS = 30

# T couples to the clause count through T ~ sqrt(CLAUSES/2 * LF), so the arms separate the two.
const ARMS = [("20 clauses, T=10  [the gate]", 20, 10),
              ("200 clauses, T=10 [clauses only]", 200, 10),
              ("200 clauses, T=32 [T rescaled]", 200, 32)]

function load(stem)
    inputs, _ = read_inputs(joinpath(DIR, "$stem.tmx"))
    meta = read_meta(joinpath(DIR, "$stem.meta.arrow"))
    return inputs, collect(Bool.(meta.label))
end

function load_year(year)
    X = TMInput[]; Y = Bool[]
    for m in 1:12
        stem = @sprintf("%d-%02d", year, m)
        isfile(joinpath(DIR, "$stem.tmx")) || continue
        xs, ys = load(stem); append!(X, xs); append!(Y, ys)
    end
    return X, Y
end

function prf(pred, y)
    tp = count(pred .& y); fp = count(pred .& .!y)
    fn = count(.!pred .& y); tn = count(.!pred .& .!y)
    prec = tp + fp == 0 ? 0.0 : tp / (tp + fp)
    rec = tp + fn == 0 ? 0.0 : tp / (tp + fn)
    f1 = prec + rec == 0 ? 0.0 : 2prec * rec / (prec + rec)
    return (f1=100 * f1, precision=100 * prec, recall=100 * rec,
            fpr=100 * (fp / max(1, fp + tn)))
end

function main()
    nseeds = isempty(ARGS) ? 5 : parse(Int, ARGS[1])
    feats = read_features(joinpath(DIR, "features.arrow"))
    width = length(feats)
    Xtr, Ytr = load("train")
    evals = [string(y) => load_year(y) for y in YEARS]
    base = JSON.parsefile("research/b5-apigraph/baselines.json")

    @printf("APIGraph width %d, train %d rows (%.1f%% malware), %d seeds\n\n",
            width, length(Ytr), 100count(Ytr) / length(Ytr), nseeds)

    println("=== gradient boosting, for the trade-off it sits at ===")
    println("  (from research/b5-apigraph/baselines.txt: precision 80-90%, recall 55-75%)")

    for (label, clauses, T) in ARMS
        @printf("\n=== %s ===\n", label)
        acc = Dict(k => (Float64[], Float64[], Float64[]) for (k, _) in evals)
        lits = Float64[]
        for seed in 1:nseeds
            m = TMClassifier([false, true], width;
                             clauses_per_class=clauses, T=T, S=S, L=L, LF=LF)
            for e in 1:EPOCHS
                train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=:none)
            end
            for (k, (X, Y)) in evals
                s = prf(predict(m, X), Y)
                push!(acc[k][1], s.f1); push!(acc[k][2], s.precision); push!(acc[k][3], s.recall)
            end
            c = Int[]
            for bank in (m.positive[1], m.negative[1]), j in 1:bank.nclauses
                n = length(clause_literals(bank, j)); n > 0 && push!(c, n)
            end
            push!(lits, isempty(c) ? 0.0 : sort(c)[cld(length(c), 2)])
        end
        @printf("%-6s %10s %11s %9s %11s %11s\n", "year", "F1", "precision", "recall", "LGB F1", "XGB F1")
        for (k, _) in evals
            f1s, ps, rs = acc[k]
            @printf("%-6s %10.2f %11.2f %9.2f %11.2f %11.2f\n",
                    k, mean(f1s), mean(ps), mean(rs), base["lightgbm"][k], base["xgboost"][k])
        end
        best_gaps = [mean(acc[k][1]) - max(base["lightgbm"][k], base["xgboost"][k]) for (k, _) in evals]
        @printf("  median literals/clause %.0f   mean gap to best booster %+.2f   worst %+.2f\n",
                median(lits), mean(best_gaps), minimum(best_gaps))
        @printf("RESULT arm=\"%s\" mean_gap=%.2f worst_gap=%.2f median_literals=%.0f\n",
                label, mean(best_gaps), minimum(best_gaps), median(lits))
    end
end

main()
