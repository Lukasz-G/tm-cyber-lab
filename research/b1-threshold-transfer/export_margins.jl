# Exports FPTM margins on LAMDA so that run.py can repeat the threshold-transfer comparison here.
#
# research/b5-fairness/ measured, on APIGraph, that gradient boosting holds the better precision/recall
# frontier (+3.77 average precision on 6 of 6 years) while its OPERATING POINT does not survive the
# drift: choosing a threshold on held-out data instead of on the test year costs FPTM 1.73 F1 and costs
# the boosters 6 to 9. That was written up as the one advantage here that survived a cross-dataset
# check. It had not -- b5-fairness runs on APIGraph alone. This is the missing half.
#
# It matters because the last result that looked convincing on one corpus was the LAMDA
# drift-robustness advantage, and it did not replicate on APIGraph (research/b5-apigraph/). The prior
# from this project's own record is therefore that a single-corpus advantage is unreliable.
#
# One structural difference to keep in view when reading the numbers: LAMDA's training pool is close to
# balanced (~48% malware) while APIGraph is ~10%. Threshold transfer is a statement about where a
# decision boundary sits, so the base rate is exactly the kind of thing that could make it behave
# differently, and that is a reason to run this rather than to assume it.
#
#   julia --project=. -t 16 research/b1-threshold-transfer/export_margins.jl [nseeds]

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
using .Tmx: read_inputs, read_meta, read_features
using TMCore
using Random, Printf, Statistics, JSON

const TMX_DIR = joinpath("data", "lamda", "tmx")
const OUT = joinpath("data", "b1-threshold-transfer")
const TRAIN_YEARS = (2013, 2014)
const TEST_YEARS = (2016, 2017, 2018, 2019, 2020, 2021, 2022)
const CLAUSES, T, S, L, LF = 20, 10, 100, 64, 10
const EPOCHS = 30
const MAX_EVAL = 25_000          # rows per test year, sampled deterministically
const TM_LAB_COMMIT = "43dba5f"

function portion_idx(year, which)
    meta = read_meta(joinpath(TMX_DIR, "$year.meta.arrow"))
    idx = [i for i in eachindex(meta.file_split) if meta.file_split[i] == String(which)]
    return meta, idx
end

function main()
    nseeds = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
    mkpath(OUT)
    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    width = length(feats)

    # --- training pool, interleaved so the validation slice keeps the class balance ---
    Xall = TMInput[]; Yall = Bool[]
    for y in TRAIN_YEARS
        meta, idx = portion_idx(y, :train)
        xs, _ = read_inputs(joinpath(TMX_DIR, "$y.tmx"); rows=idx)
        append!(Xall, xs); append!(Yall, Bool[Bool(meta.label[i]) for i in idx])
    end
    n = length(Yall)
    val_pos = [i for i in 1:n if i % 5 == 0]
    tr_pos  = [i for i in 1:n if i % 5 != 0]
    Xtr, Ytr = Xall[tr_pos], Yall[tr_pos]
    Xva, Yva = Xall[val_pos], Yall[val_pos]
    count(Yva) > 0 || error("validation slice has no malware")

    @printf("LAMDA width %d  pool %d -> train %d (%.1f%% malware) / val %d (%.1f%%)\n",
            width, n, length(Ytr), 100mean(Ytr), length(Yva), 100mean(Yva))
    @printf("clauses=%d T=%d S=%d L=%d LF=%d epochs=%d  ceiling=LiteralCapped  parallel=:none  tm-lab %s\n",
            CLAUSES, T, S, L, LF, EPOCHS, TM_LAB_COMMIT)
    flush(stdout)

    # --- IID: the held-out test portions of the training years ---
    Xiid = TMInput[]; Yiid = Bool[]
    iid_src = Dict{String,Vector{Int}}()
    for y in TRAIN_YEARS
        meta, idx = portion_idx(y, :test)
        xs, _ = read_inputs(joinpath(TMX_DIR, "$y.tmx"); rows=idx)
        append!(Xiid, xs); append!(Yiid, Bool[Bool(meta.label[i]) for i in idx])
        iid_src[string(y)] = idx
    end

    evals = Dict{String,Tuple{Vector{TMInput},Vector{Bool},Vector{Int}}}()
    evals["IID"] = (Xiid, Yiid, Int[])
    for y in TEST_YEARS
        meta, idx = portion_idx(y, :test)
        sel = if length(idx) > MAX_EVAL
            randperm(MersenneTwister(31337), length(idx))[1:MAX_EVAL] |> sort
        else
            collect(1:length(idx))
        end
        keep = idx[sel]
        xs, _ = read_inputs(joinpath(TMX_DIR, "$y.tmx"); rows=keep)
        evals[string(y)] = (xs, Bool[Bool(meta.label[i]) for i in keep], keep)
        @printf("  %s: %d rows (%d malware)%s\n", string(y), length(keep),
                count(evals[string(y)][2]), length(idx) > MAX_EVAL ? " [sampled]" : "")
    end
    flush(stdout)

    out = Dict{String,Any}(
        "n_seeds" => nseeds, "width" => width,
        "train_pos" => tr_pos, "val_pos" => val_pos,   # positions within the concatenated train pool
        "iid_rows" => iid_src,                          # per-year row indices making up IID
        "eval_rows" => Dict(k => v[3] for (k, v) in evals if k != "IID"),
        "labels" => Dict{String,Any}("val" => Int.(Yva)),
        "margins" => Dict{String,Any}("val" => Vector{Vector{Float64}}()),
    )
    for (k, v) in evals
        out["labels"][k] = Int.(v[2])
        out["margins"][k] = Vector{Vector{Float64}}()
    end

    marg(m, X) = Float64[score(m, 2, x) - score(m, 1, x) for x in X]

    for seed in 1:nseeds
        t0 = time()
        m = TMClassifier([false, true], width;
                         clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
        for e in 1:EPOCHS
            train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=:none)
        end
        push!(out["margins"]["val"], marg(m, Xva))
        for (k, v) in evals
            push!(out["margins"][k], marg(m, v[1]))
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
