# QUESTION:  is FPTM's PR frontier limited by SCORE RESOLUTION — too few distinct margin values to
#            carve out a small pure high-confidence subset?
# SURPRISE:  yes, and it is the last standing hypothesis. Six knobs have been eliminated (clause count,
#            T, s, L, class balance, train-set size) and all six slid the operating point along the
#            frontier without moving it. Resolution is the only candidate left that explains the SHAPE
#            of the failure: recall collapsing to 11.8% when 94% precision is demanded is what a large
#            tied top-margin bucket looks like. The sibling project measured that a fuzzy vote buys ~3x
#            its binarisation's resolution and not LFx, with firing votes near the bottom of
#            [1, LF], so coarseness is expected -- what is unmeasured is whether it BINDS.
# ARMS:      margin cardinality and top-bucket purity at 20 and 200 clauses, against the boosters'
#            score cardinality on identical rows. The 200-clause arm is the discriminating one: if
#            resolution is the binding constraint, 10x the clauses must raise cardinality sharply, and
#            then the fact that F1 barely moved (+0.75) would REFUTE resolution as the cause rather
#            than support it. Testing one clause count could not distinguish those.
# PASS/FAIL: not a gate. RESOLUTION BINDS if the top-margin bucket is large and impure -- i.e. the
#            highest-scoring examples cannot be separated from benign ones at all. It DOES NOT BIND if
#            the top bucket is small and pure, in which case the ranking is fine-grained where it
#            matters and the deficit is elsewhere.
#
#   julia --project=. -t 16 research/b5-resolution/run.jl [nseeds]

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
using .Tmx: read_inputs, read_meta, read_features
using TMCore
using Random, Printf, Statistics

const DIR = joinpath("data", "apigraph", "tmx", "apigraph")
const YEAR = 2016                 # the year where the collapse was starkest
const T_, S_, L_, LF_ = 10, 25, 64, 10
const EPOCHS = 30
const CLAUSE_ARMS = (20, 200)

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

margins(m, X) = Float64[score(m, 2, x) - score(m, 1, x) for x in X]

function main()
    nseeds = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
    feats = read_features(joinpath(DIR, "features.arrow"))
    Xtr, Ytr = load("train")
    X, Y = load_year(YEAR)
    npos = count(Y)
    @printf("APIGraph %d: %d rows, %d malware (%.1f%%), width %d, %d seeds\n\n",
            YEAR, length(Y), npos, 100npos / length(Y), length(feats), nseeds)

    for clauses in CLAUSE_ARMS
        card = Int[]; topn = Int[]; topp = Float64[]
        need = Int[]; needp = Float64[]
        for seed in 1:nseeds
            m = TMClassifier([false, true], length(feats);
                             clauses_per_class=clauses, T=T_, S=S_, L=L_, LF=LF_)
            for e in 1:EPOCHS
                train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=:none)
            end
            mg = margins(m, X)
            u = sort(unique(mg); rev=true)
            push!(card, length(u))

            # the single highest-margin bucket: how many examples tie there, and how pure is it
            top = mg .== u[1]
            push!(topn, count(top))
            push!(topp, 100 * count(Y[top]) / count(top))

            # how many buckets must be taken, cumulatively, to cover 10% of the malware -- and how
            # pure that prefix is. This is the quantity a high-precision operating point needs.
            cum_tp = 0; cum_n = 0; k = 0
            for v in u
                sel = mg .== v
                cum_tp += count(Y[sel]); cum_n += count(sel); k += 1
                cum_tp >= 0.10 * npos && break
            end
            push!(need, k); push!(needp, 100 * cum_tp / cum_n)
        end
        @printf("%d clauses/class:\n", clauses)
        @printf("  distinct margin values        %.0f  (of %d examples)\n", mean(card), length(Y))
        @printf("  top bucket: size %.0f, purity %.1f%%  (base rate %.1f%%)\n",
                mean(topn), mean(topp), 100npos / length(Y))
        @printf("  buckets to reach 10%% recall   %.1f, precision there %.1f%%\n",
                mean(need), mean(needp))
        @printf("RESULT clauses=%d cardinality=%.0f top_size=%.0f top_purity=%.1f buckets_for_10pct_recall=%.1f precision_there=%.1f\n\n",
                clauses, mean(card), mean(topn), mean(topp), mean(need), mean(needp))
    end

    println("reference: a gradient booster emits a distinct float per example, so its score")
    println("cardinality equals the number of examples and its top bucket has size 1.")
end

main()
