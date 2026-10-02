# QUESTION:  across every evaluation period, and when the model is RETRAINED, not ablated at
#            inference, does removing the 38 attributed features a frequency ranking misses actually
#            improve detection under drift?
# SURPRISE:  yes, and it is the arm that decides whether the previous result is a recommendation or a
#            curiosity. research/interp-residual/ found that forcing those features to zero at inference
#            IMPROVES drifted-year F1 -- by 10.32 on 2021 and 2.63 on 2019, against 1.24 and 0.25 for
#            size-matched random sets. But inference ablation answers "does this model use them"; it
#            cannot answer "is there a better model that ignores them", and the two come apart whenever a
#            model can redistribute the work onto other features. A booster deprived of one feature finds
#            a correlated substitute; a twenty-clause rule ensemble may not have the capacity to. If
#            retraining reproduces the gain, exact attribution has located a prunable defect from training
#            data alone and that is a deployable procedure. If retraining erases it -- because the fresh
#            model simply relearns the same shortcut through other literals -- then the inference result
#            was about this particular model and not about the features, and the recommendation dies.
# ARMS:      four, all RETRAINED from scratch per seed, and the third is what makes the second readable:
#              1. BASELINE      -- all features. The reference, and the model whose attribution defines
#                                  the sets below.
#              2. ZERO-RESIDUAL -- retrained with the attributed-but-not-frequent 38 forced to zero.
#              3. ZERO-RANDOM   -- retrained with a random 38 from the nonzero-attribution support forced
#                                  to zero. Matched on count, on being in play, and on the dead-channel
#                                  side effect described below.
#              4. ZERO-FREQONLY -- retrained with the 38 features frequency ranks highly and attribution
#                                  does not. The mirror: inference ablation said these HELP under drift,
#                                  so retraining without them should hurt, and if it does not then the
#                                  inference result was measuring something other than the features.
#            Evaluated on IID and every year 2016-2022 -- the full sweep, because the inference result was
#            four times weaker on 2019 than on 2021 and two years cannot distinguish an effect from a
#            coincidence.
# PASS/FAIL: the intervention is REAL if arm 2 beats both arm 1 and arm 3 on a majority of drifted years,
#            with the margin over arm 3 -- not over arm 1 -- being the claim, since arm 3 shares every
#            confound. It is an ARTEFACT of inference-time ablation if arm 2 and arm 3 are comparable once
#            the model is retrained, and in that case research/interp-residual/ must be restated as a
#            property of a fixed model and not as a candidate intervention.
#
#   julia --project=. -t 16 research/interp-residual-retrain/run.jl [nseeds]
#
# WHY FEATURES ARE ZEROED AND NOT DROPPED, which is a confound this project has paid for before. Deleting
# 38 of 4,561 columns changes the input width, and two things in an FPTM depend on width: the effective
# specificity s = width/S, which in a sibling measurement accounted for more of an apparent effect than
# the thing being studied, and the L growth gate, whose behaviour moves with the literal count. Zeroing
# keeps the width, S and s identical across all four arms.
#
# Zeroing is not free either -- an always-zero feature is satisfied for free by its negation, so a clause
# includes it at no evaluation cost, which inflates its literal count and shifts the L gate. That is
# exactly the dead-channel effect measured in the sibling project at +0.0109 accuracy for 32 dead bits.
# The defence is that arms 2, 3 and 4 each zero EXACTLY 38 features, so they carry an identical
# dead-channel perturbation and the comparison between them is clean. Arm 1 does not, which is precisely
# why the verdict rests on arm 2 against arm 3, not on arm 2 against arm 1.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: shapley_mean
using TMCore
using Random, Printf, Statistics, Base.Threads

const TMX_DIR = joinpath("data", "lamda", "tmx")
const TRAIN_YEARS = (2013, 2014)
const EVAL_YEARS = (2016, 2017, 2018, 2019, 2020, 2021, 2022)
const DRIFTED = ("2016", "2017", "2018", "2019", "2020", "2021", "2022")
const CLAUSES, T, S, L, LF = 20, 10, 100, 64, 10
const EPOCHS = 30
const CI = 2
const NBG, NEX = 100, 100
const K = 100
const MAX_EVAL = 20_000
const TM_LAB_COMMIT = "43dba5f"

function portion(year, which)
    meta = read_meta(joinpath(TMX_DIR, "$year.meta.arrow"))
    idx = [i for i in eachindex(meta.file_split) if meta.file_split[i] == String(which)]
    xs, _ = read_inputs(joinpath(TMX_DIR, "$year.tmx"); rows=idx)
    return xs, Bool[Bool(meta.label[i]) for i in idx]
end

function load_train()
    X = TMInput[]; Y = Bool[]
    for y in TRAIN_YEARS
        xs, ys = portion(y, :train); append!(X, xs); append!(Y, ys)
    end
    return X, Y
end

function subsample(X, Y, n, seed)
    length(Y) <= n && return X, Y
    sel = sort(randperm(MersenneTwister(seed), length(Y))[1:n])
    return X[sel], Y[sel]
end

"""
Copy of `X` with every feature in `drop` forced to zero, at the same width.

Built row by row into packed TMInputs and not held as dense Bool vectors: at 150k rows and width
4,561 the dense form is ~700 MB while the packed form is ~86 MB, and nothing downstream needs the dense
version.
"""
function zeroed(X, drop, width)
    isempty(drop) && return X
    dset = Set(drop)
    out = Vector{TMInput}(undef, length(X))
    buf = Vector{Bool}(undef, width)
    for r in eachindex(X)
        x = X[r]
        @inbounds for i in 1:width
            buf[i] = x[i]
        end
        for f in dset
            buf[f] = false
        end
        out[r] = TMInput(buf)
    end
    return out
end

function f1(m, X, Y)
    tp = 0; fp = 0; fn = 0
    for (x, y) in zip(X, Y)
        p = predict(m, x)
        p && y && (tp += 1)
        p && !y && (fp += 1)
        !p && y && (fn += 1)
    end
    prec = tp + fp == 0 ? 0.0 : tp / (tp + fp)
    rec = tp + fn == 0 ? 0.0 : tp / (tp + fn)
    return prec + rec == 0 ? 0.0 : 200 * prec * rec / (prec + rec)
end

function fit(X, Y, width, seed)
    m = TMClassifier([false, true], width;
                     clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    for e in 1:EPOCHS
        train!(m, X, Y; rng=MersenneTwister(1000seed + e), parallel=:none)
    end
    return m
end

function feature_counts(X, Y, width)
    n11 = zeros(Int, width); n10 = zeros(Int, width)
    for r in eachindex(X)
        x = X[r]; mal = Y[r]
        @inbounds for i in 1:width
            x[i] && (mal ? (n11[i] += 1) : (n10[i] += 1))
        end
    end
    return n11, n10
end

function shapley_importance(m, ex, bg, width)
    parts = [zeros(Float64, width) for _ in 1:nthreads()]
    @threads for r in eachindex(ex)
        phi = shapley_mean(m, CI, ex[r], bg)
        p = parts[threadid()]
        @inbounds for i in 1:width
            p[i] += abs(phi[i])
        end
    end
    return reduce(+, parts) ./ length(ex)
end

topk(v, k) = partialsortperm(v, 1:min(k, length(v)); rev=true)

function main()
    nseeds = isempty(ARGS) ? 5 : parse(Int, ARGS[1])
    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    width = length(feats)
    Xtr, Ytr = load_train()
    npos = count(Ytr); nneg = length(Ytr) - npos
    n11, n10 = feature_counts(Xtr, Ytr, width)
    freq = [abs(n11[i] / npos - n10[i] / nneg) for i in 1:width]

    @printf("LAMDA width %d  train %d  %d seeds  k=%d  RETRAINED per arm\n",
            width, length(Ytr), nseeds, K)
    @printf("clauses=%d T=%d S=%d L=%d LF=%d epochs=%d  ceiling=LiteralCapped  parallel=:none  tm-lab %s\n",
            CLAUSES, T, S, L, LF, EPOCHS, TM_LAB_COMMIT)
    println("features are ZEROED, not dropped, so width / S / s are identical in every arm;")
    println("arms 2-4 each zero exactly 38, so they share the dead-channel perturbation too.\n")
    flush(stdout)

    evalsets = Dict{String,Tuple{Vector{TMInput},Vector{Bool}}}()
    Xi = TMInput[]; Yi = Bool[]
    for y in TRAIN_YEARS
        xs, ys = portion(y, :test); append!(Xi, xs); append!(Yi, ys)
    end
    evalsets["IID"] = subsample(Xi, Yi, MAX_EVAL, 4242)
    for y in EVAL_YEARS
        xs, ys = portion(y, :test)
        evalsets[string(y)] = subsample(xs, ys, MAX_EVAL, 4242)
    end
    order = ["IID", DRIFTED...]
    for k in order
        @printf("  %-5s %6d rows, %5d malware\n", k, length(evalsets[k][2]), count(evalsets[k][2]))
    end
    println()
    flush(stdout)

    ARMS = ("baseline", "zero-residual", "zero-random", "zero-freqonly")
    res = Dict{Tuple{String,String},Vector{Float64}}()
    push_res!(k, v) = (haskey(res, k) ? push!(res[k], v) : (res[k] = [v]))

    for seed in 1:nseeds
        t0 = time()
        base = fit(Xtr, Ytr, width, seed)

        bg, ex = Xtr[1:NBG], Xtr[(end - NEX + 1):end]
        shap = shapley_importance(base, ex, bg, width)
        support = [i for i in 1:width if shap[i] != 0]
        mtop = Set(topk(shap, K)); ftop = Set(topk(freq, K))
        residual = collect(setdiff(mtop, ftop))
        freqonly = collect(setdiff(ftop, mtop))
        pool = setdiff(support, mtop)
        rng = MersenneTwister(555 + seed)
        randset = pool[randperm(rng, length(pool))[1:min(length(residual), length(pool))]]

        @printf("seed %d: support %d | residual %d, freq-only %d, random %d  (%.0fs to train+attribute)\n",
                seed, length(support), length(residual), length(freqonly), length(randset), time() - t0)
        @printf("RESULT seed=%d support=%d n_residual=%d n_freqonly=%d n_random=%d\n",
                seed, length(support), length(residual), length(freqonly), length(randset))
        flush(stdout)

        drops = Dict("baseline" => Int[], "zero-residual" => residual,
                     "zero-random" => randset, "zero-freqonly" => freqonly)

        for arm in ARMS
            d = drops[arm]
            m = arm == "baseline" ? base : fit(zeroed(Xtr, d, width), Ytr, width, seed)
            for k in order
                X, Y = evalsets[k]
                score = f1(m, zeroed(X, d, width), Y)
                push_res!((arm, k), score)
                @printf("RESULT seed=%d arm=%s eval=%s f1=%.2f\n", seed, arm, k, score)
            end
            @printf("  %-14s done\n", arm)
            flush(stdout)
        end
    end

    println("\n", "-"^86)
    @printf("%-6s %10s %14s %13s %15s %14s\n",
            "eval", "baseline", "zero-residual", "zero-random", "resid - random", "zero-freqonly")
    println("-"^86)
    g(a, k) = haskey(res, (a, k)) ? mean(res[(a, k)]) : NaN
    wins = 0
    for k in order
        delta = g("zero-residual", k) - g("zero-random", k)
        k != "IID" && delta > 0 && (wins += 1)
        @printf("%-6s %10.2f %14.2f %13.2f %+15.2f %14.2f\n",
                k, g("baseline", k), g("zero-residual", k), g("zero-random", k),
                delta, g("zero-freqonly", k))
        @printf("RESULT mean eval=%s baseline=%.2f residual=%.2f random=%.2f delta=%+.2f freqonly=%.2f\n",
                k, g("baseline", k), g("zero-residual", k), g("zero-random", k),
                delta, g("zero-freqonly", k))
    end
    println("-"^86)
    @printf("zero-residual beats zero-random on %d of %d drifted periods\n", wins, length(DRIFTED))
    @printf("RESULT verdict residual_beats_random=%d/%d\n", wins, length(DRIFTED))
    println()
    println("The claim is the 'resid - random' column, NOT the comparison against baseline: arms 2-4")
    println("share a 38-feature dead-channel perturbation that the baseline does not carry, so only")
    println("arm 2 against arm 3 isolates the identity of the features from the fact of removing some.")
end

main()
