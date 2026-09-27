# QUESTION:  when the clause budget goes from 20 to 200, does the model learn 10x as many DISTINCT
#            patterns, or 10x copies of the same few?
# SURPRISE:  yes, and it is the last explanation standing. Seven knobs have been eliminated and three
#            measurements point the same way: 80% of the training data is inert (6k rows equal 30k),
#            10x the clauses buys +0.75 F1, and score resolution is adequate. If the extra clauses are
#            near-duplicates, the ceiling is the feedback rule's pattern diversity rather than any
#            hyperparameter -- which makes it a tm-lab question, not a TM-Cyber one. If they ARE
#            diverse, then diversity is fine and the limitation is that the patterns themselves are
#            individually weak, which is a different and harder problem.
# ARMS:      20 and 200 clauses per class, plus a RANDOM-CLAUSE control at each budget. The control is
#            essential: sparse clauses drawn over 1,159 features will have low pairwise overlap by
#            chance alone, so "overlap is low" means nothing without knowing what chance looks like at
#            the same clause sizes. Comparing trained overlap against random overlap at matched size is
#            the only reading that distinguishes learned redundancy from sparsity.
# PASS/FAIL: not a gate. SATURATED if trained clauses are markedly more similar to each other than
#            size-matched random clauses, and if that similarity rises with the budget. NOT SATURATED if
#            trained similarity stays near the random baseline as the budget grows.
#
#   julia --project=. -t 16 research/b5-diversity/run.jl [nseeds]

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: clause_literals
using TMCore
using Random, Printf, Statistics

const DIR = joinpath("data", "apigraph", "tmx", "apigraph")
const T_, S_, L_, LF_ = 10, 25, 64, 10
const EPOCHS = 30
const ARMS = (20, 200)

load_pool() = let
    inputs, _ = read_inputs(joinpath(DIR, "train.tmx"))
    meta = read_meta(joinpath(DIR, "train.meta.arrow"))
    inputs, collect(Bool.(meta.label))
end

"Literal sets of every non-empty clause in one polarity bank, as Sets of (feature, negated)."
function clause_sets(bank)
    out = Set{Tuple{Int,Bool}}[]
    for j in 1:bank.nclauses
        l = clause_literals(bank, j)
        isempty(l) || push!(out, Set(l))
    end
    return out
end

jacc(a, b) = (u = length(union(a, b)); u == 0 ? 0.0 : length(intersect(a, b)) / u)

"Mean pairwise Jaccard similarity, and the fraction of pairs above 0.9 (near-duplicates)."
function pairwise(sets)
    n = length(sets)
    n < 2 && return (NaN, NaN, NaN)
    tot = 0.0; cnt = 0; dup = 0; mx = 0.0
    for i in 1:(n - 1), j in (i + 1):n
        s = jacc(sets[i], sets[j])
        tot += s; cnt += 1; mx = max(mx, s)
        s >= 0.9 && (dup += 1)
    end
    return (tot / cnt, 100 * dup / cnt, mx)
end

"Random clauses matched to the given sizes, drawn uniformly over features and polarities."
function random_sets(sizes, width, rng)
    [Set((rand(rng, 1:width), rand(rng, Bool)) for _ in 1:k) for k in sizes]
end

function main()
    nseeds = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
    feats = read_features(joinpath(DIR, "features.arrow"))
    width = length(feats)
    Xtr, Ytr = load_pool()
    @printf("APIGraph  width %d  pool %d rows  T=%d S=%d L=%d LF=%d  %d seeds\n\n",
            width, length(Ytr), T_, S_, L_, LF_, nseeds)

    @printf("%-9s %-9s %9s %11s %12s %10s %13s\n",
            "clauses", "bank", "n", "med size", "mean Jacc", "dup >0.9", "random Jacc")
    for clauses in ARMS
        agg = Dict{String,Vector{NTuple{4,Float64}}}("positive" => [], "negative" => [])
        for seed in 1:nseeds
            m = TMClassifier([false, true], width;
                             clauses_per_class=clauses, T=T_, S=S_, L=L_, LF=LF_)
            for e in 1:EPOCHS
                train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=:none)
            end
            rng = MersenneTwister(9000 + seed)
            for (name, bank) in (("positive", m.positive[2]), ("negative", m.negative[2]))
                sets = clause_sets(bank)
                isempty(sets) && continue
                sizes = [length(s) for s in sets]
                mj, dup, _ = pairwise(sets)
                rj, _, _ = pairwise(random_sets(sizes, width, rng))
                push!(agg[name], (Float64(length(sets)), median(sizes), mj, dup))
                push!(agg[name], (NaN, NaN, rj, NaN))    # random baseline rides alongside
            end
        end
        for name in ("positive", "negative")
            rows = agg[name]
            trained = [r for r in rows if !isnan(r[1])]
            randj = [r[3] for r in rows if isnan(r[1])]
            isempty(trained) && continue
            @printf("%-9d %-9s %9.0f %11.0f %12.3f %9.1f%% %13.3f\n",
                    clauses, name, mean(r[1] for r in trained), mean(r[2] for r in trained),
                    mean(r[3] for r in trained), mean(r[4] for r in trained), mean(randj))
            @printf("RESULT clauses=%d bank=%s n=%.0f med_size=%.0f mean_jaccard=%.4f dup_frac=%.2f random_jaccard=%.4f\n",
                    clauses, name, mean(r[1] for r in trained), mean(r[2] for r in trained),
                    mean(r[3] for r in trained), mean(r[4] for r in trained), mean(randj))
        end
    end
    println("\nSATURATED if trained similarity greatly exceeds the size-matched random baseline and")
    println("rises with the budget. The random column is what sparsity alone produces.")
end

main()
