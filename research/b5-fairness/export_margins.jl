# Exports FPTM decision margins on APIGraph so that run.py can compare FRONTIER against FRONTIER.
#
# Three recorded unfairnesses are being fixed at once, and all three currently flatter us:
#   1. earlier experiments swept FPTM's threshold and compared against boosters at their DEFAULT 0.5
#      cut. A swept model against an unswept one is not a comparison.
#   2. dominance counts came from seed 1 only, while the F1 figures averaged five seeds.
#   3. the swept threshold was chosen on the same data it was scored on, so it is an oracle value
#, not an operating point a deployment could reach.
#
# Fix for (3) drives the design: the 2012 pool is split 80/20, training uses the 80% and the threshold
# is chosen on the held-out 20%, then applied unchanged to every test year. That means this trains on
# LESS data than research/b5-apigraph/ did, so its absolute numbers are not comparable with that
# experiment's -- but both models here see exactly the same 80%, which is what makes the comparison
# itself fair. run.py trains the boosters on the identical split, using the indices exported here.
#
# The split is INTERLEAVED (every 5th row) and not a contiguous tail. A contiguous tail was tried
# first and produced a validation slice containing 0.0% malware, because the 2012 pool is ordered by
# class -- so the honest-threshold fix would have silently had no positives to choose a threshold on.
# Exporting the indices in place of the rule is what makes the two languages provably agree.
#
#   julia --project=. -t 16 research/b5-fairness/export_margins.jl [nseeds]

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
using .Tmx: read_inputs, read_meta, read_features
using TMCore
using Random, Printf, Statistics, JSON

const DIR = joinpath("data", "apigraph", "tmx", "apigraph")
const OUT = joinpath("data", "b5-fairness")
const YEARS = 2013:2018
const CLAUSES, T, S, L, LF = 20, 10, 25, 64, 10
const EPOCHS = 30
const VAL_FRAC = 0.2
const TM_LAB_COMMIT = "43dba5f"

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
    nseeds = isempty(ARGS) ? 5 : parse(Int, ARGS[1])
    mkpath(OUT)
    feats = read_features(joinpath(DIR, "features.arrow"))
    width = length(feats)

    Xpool, Ypool = load("train")
    n = length(Ypool)

    # INTERLEAVED, not a contiguous tail. The 2012 pool is ordered by class -- taking the last 20% by
    # row order produced a validation slice with 0.0% malware, which cannot select a threshold at all.
    # Every 5th row preserves the class balance whatever the ordering is, and the exact indices are
    # exported so the Python side cannot derive a different split.
    val_idx = [i for i in 1:n if i % 5 == 0]
    tr_idx  = [i for i in 1:n if i % 5 != 0]
    Xtr, Ytr = Xpool[tr_idx], Ypool[tr_idx]
    Xva, Yva = Xpool[val_idx], Ypool[val_idx]
    count(Yva) > 0 || error("validation slice has no malware -- threshold selection is impossible")

    @printf("APIGraph width %d  pool %d -> train %d (%.1f%% malware) / val %d (%.1f%% malware)\n",
            width, n, length(Ytr), 100mean(Ytr), length(Yva), 100mean(Yva))
    @printf("clauses=%d T=%d S=%d L=%d LF=%d epochs=%d  ceiling=LiteralCapped  parallel=:none  tm-lab %s\n",
            CLAUSES, T, S, L, LF, EPOCHS, TM_LAB_COMMIT)
    flush(stdout)

    evals = Dict{String,Tuple{Vector{TMInput},Vector{Bool}}}()
    for y in YEARS
        evals[string(y)] = load_year(y)
        @printf("  %d: %d rows, %d malware\n", y, length(evals[string(y)][2]),
                count(evals[string(y)][2]))
    end
    flush(stdout)

    out = Dict{String,Any}(
        "n_seeds" => nseeds, "width" => width, "val_frac" => VAL_FRAC,
        "n_train" => length(Ytr), "n_val" => length(Yva),
        "val_idx" => val_idx, "train_idx" => tr_idx,      # 1-based; Python subtracts one
        "labels" => Dict{String,Any}("val" => Int.(Yva)),
        "margins" => Dict{String,Any}("val" => Vector{Vector{Float64}}()),
    )
    for y in YEARS
        out["labels"][string(y)] = Int.(evals[string(y)][2])
        out["margins"][string(y)] = Vector{Vector{Float64}}()
    end

    for seed in 1:nseeds
        t0 = time()
        m = TMClassifier([false, true], width;
                         clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
        for e in 1:EPOCHS
            train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=:none)
        end
        push!(out["margins"]["val"], margins(m, Xva))
        for y in YEARS
            push!(out["margins"][string(y)], margins(m, evals[string(y)][1]))
        end
        @printf("  seed %d  %.1fs\n", seed, time() - t0)
        flush(stdout)
    end

    open(joinpath(OUT, "fptm_margins.json"), "w") do io
        JSON.print(io, out)
    end
    @printf("wrote %s\n", joinpath(OUT, "fptm_margins.json"))
end

main()
