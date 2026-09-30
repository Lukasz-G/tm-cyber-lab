# QUESTION:  the ~38% of the model's attributed top-100 that a frequency ranking does NOT recover --
#            does it actually carry the model's decisions, or is it the tail of a ranking that happens
#            to disagree there?
# SURPRISE:  yes, and it is the only route left to reinstating any interpretability claim in this
#            project. research/interp-dataset-control/ retracted the readability claim because a plain
#            document-frequency difference recovers 62 of the model's top-100 features. Its
#            pre-registration was explicit that the remaining 38 licenses "the clauses select features a
#            univariate ranking does not" ONLY if that part is shown to matter, and it was not shown. So
#            this settles it either way. If ablating the residual costs more per feature than ablating a
#            size-matched random set, the residual is load-bearing and a narrow claim becomes
#            defensible: the attribution finds features frequency misses, and they are features the
#            model uses. If it costs no more than random, the residual is ranking noise, the retraction
#            stands unqualified, and the interpretability line closes rather than staying open as a
#            maybe.
# ARMS:      six ablations of the test input, and the size-matched random pairs are the point:
#              1. RESIDUAL      -- the model's top-100 minus what frequency also picks (~38 features).
#              2. random, |1|   -- the control for arm 1. A set of the same size drawn from features the
#                                  model attributes nonzero value to, so it is matched on "in play" as
#                                  well as on size.
#              3. OVERLAP       -- the model's top-100 that frequency also picks (~62 features).
#              4. random, |3|   -- the control for arm 3.
#              5. FREQ-ONLY     -- frequency's top-100 minus the model's. Features a univariate ranking
#                                  would hand an analyst and the attribution says are worth nothing. If
#                                  ablating these costs as much as the residual, attribution adds
#                                  nothing over frequency even where they disagree.
#              6. none          -- unablated, the reference F1.
#            Arms 2 and 4 are drawn from the nonzero-attribution support rather than from all 4,561
#            features, because 4,065 of those features have exactly zero attribution and ablating them
#            is guaranteed to do nothing. A random control over the full width would make any real set
#            look load-bearing by comparison, which would be a rigged comparison.
# PASS/FAIL: not a gate; it decides what may be written. The residual is LOAD-BEARING if arm 1 costs
#            materially more F1 than arm 2 at matched size, on the drifted years as well as IID. It is
#            NOISE if arms 1 and 2 are comparable, and then the interpretability line is closed.
#
#   julia --project=. -t 16 research/interp-residual/run.jl [nseeds]
#
# Ablation is at INFERENCE, by forcing the selected features to zero, not by retraining without them.
# That is the right test for an attribution claim: attribution says these features drive THIS model's
# output, so removing them from the input must change that output. Retraining without them would ask a
# different question -- whether an equally good model exists that ignores them -- and the answer to that
# can be yes for features which nonetheless drive this one.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: shapley_mean
using TMCore
using Random, Printf, Statistics, Base.Threads

const TMX_DIR = joinpath("data", "lamda", "tmx")
const TRAIN_YEARS = (2013, 2014)
const EVAL = ("IID", "2019", "2021")     # one in-distribution, two drifted
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
        xs, ys = portion(y, :train)
        append!(X, xs); append!(Y, ys)
    end
    return X, Y
end

function subsample(X, Y, n, seed)
    length(Y) <= n && return X, Y
    sel = sort(randperm(MersenneTwister(seed), length(Y))[1:n])
    return X[sel], Y[sel]
end

"Dense Bool copy of the rows, so features can be forced to zero."
densify(X, width) = [Bool[x[i] for i in 1:width] for x in X]

function f1_dense(m, rows, Y, drop)
    tp = 0; fp = 0; fn = 0
    dropset = Set(drop)
    for (r, y) in zip(rows, Y)
        if isempty(dropset)
            p = predict(m, TMInput(r))
        else
            v = copy(r)
            for f in dropset
                v[f] = false
            end
            p = predict(m, TMInput(v))
        end
        p && y ? (tp += 1) : nothing
        p && !y ? (fp += 1) : nothing
        !p && y ? (fn += 1) : nothing
    end
    prec = tp + fp == 0 ? 0.0 : tp / (tp + fp)
    rec = tp + fn == 0 ? 0.0 : tp / (tp + fn)
    return prec + rec == 0 ? 0.0 : 200 * prec * rec / (prec + rec)
end

function feature_counts(X, Y, width)
    n11 = zeros(Int, width); n10 = zeros(Int, width)
    for r in eachindex(X)
        x = X[r]; mal = Y[r]
        @inbounds for i in 1:width
            if x[i]
                mal ? (n11[i] += 1) : (n10[i] += 1)
            end
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
    nseeds = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    width = length(feats)
    Xtr, Ytr = load_train()
    npos = count(Ytr); nneg = length(Ytr) - npos

    n11, n10 = feature_counts(Xtr, Ytr, width)
    freq = [abs(n11[i] / npos - n10[i] / nneg) for i in 1:width]

    @printf("LAMDA width %d  train %d  %d seeds  k=%d  ablation at inference\n",
            width, length(Ytr), nseeds, K)
    @printf("clauses=%d T=%d S=%d L=%d LF=%d epochs=%d  ceiling=LiteralCapped  tm-lab %s\n\n",
            CLAUSES, T, S, L, LF, EPOCHS, TM_LAB_COMMIT)
    flush(stdout)

    evalsets = Dict{String,Tuple{Vector{Vector{Bool}},Vector{Bool}}}()
    for name in EVAL
        if name == "IID"
            X = TMInput[]; Y = Bool[]
            for y in TRAIN_YEARS
                xs, ys = portion(y, :test); append!(X, xs); append!(Y, ys)
            end
        else
            X, Y = portion(parse(Int, name), :test)
        end
        X, Y = subsample(X, Y, MAX_EVAL, 4242)
        evalsets[name] = (densify(X, width), Y)
        @printf("  %s: %d rows, %d malware\n", name, length(Y), count(Y))
    end
    println()
    flush(stdout)

    agg = Dict{Tuple{String,String},Vector{Float64}}()
    push_agg!(k, v) = (haskey(agg, k) ? push!(agg[k], v) : (agg[k] = [v]))

    for seed in 1:nseeds
        m = TMClassifier([false, true], width;
                         clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
        for e in 1:EPOCHS
            train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=:none)
        end

        bg, ex = Xtr[1:NBG], Xtr[(end - NEX + 1):end]
        shap = shapley_importance(m, ex, bg, width)
        support = [i for i in 1:width if shap[i] != 0]

        mtop = Set(topk(shap, K))
        ftop = Set(topk(freq, K))
        residual = collect(setdiff(mtop, ftop))
        overlap  = collect(intersect(mtop, ftop))
        freqonly = collect(setdiff(ftop, mtop))

        rng = MersenneTwister(555 + seed)
        pool = setdiff(support, mtop)            # nonzero-attribution features outside the top-k
        rnd(n) = length(pool) >= n ? pool[randperm(rng, length(pool))[1:n]] : pool
        rand_res = rnd(length(residual))
        rand_ovl = rnd(length(overlap))

        @printf("seed %d: support %d features | residual %d, overlap %d, freq-only %d\n",
                seed, length(support), length(residual), length(overlap), length(freqonly))
        @printf("RESULT seed=%d support=%d n_residual=%d n_overlap=%d\n",
                seed, length(support), length(residual), length(overlap))
        flush(stdout)

        for name in EVAL
            rows, Y = evalsets[name]
            base = f1_dense(m, rows, Y, Int[])
            for (arm, drop) in (("residual", residual), ("random|res|", rand_res),
                                ("overlap", overlap), ("random|ovl|", rand_ovl),
                                ("freq-only", freqonly))
                f = f1_dense(m, rows, Y, drop)
                push_agg!((name, arm), base - f)
                @printf("RESULT seed=%d eval=%s arm=%s n=%d base_f1=%.2f abl_f1=%.2f drop=%.2f\n",
                        seed, name, arm, length(drop), base, f, base - f)
            end
            push_agg!((name, "base"), base)
            flush(stdout)
        end
    end

    println("\n", "-"^84)
    @printf("%-6s %9s %11s %13s %11s %13s %12s\n",
            "eval", "base F1", "residual", "random|res|", "overlap", "random|ovl|", "freq-only")
    println("-"^84)
    for name in EVAL
        g(a) = haskey(agg, (name, a)) ? mean(agg[(name, a)]) : NaN
        @printf("%-6s %9.2f %11.2f %13.2f %11.2f %13.2f %12.2f\n",
                name, g("base"), g("residual"), g("random|res|"),
                g("overlap"), g("random|ovl|"), g("freq-only"))
        @printf("RESULT mean eval=%s base=%.2f residual=%.2f random_res=%.2f overlap=%.2f random_ovl=%.2f freq_only=%.2f\n",
                name, g("base"), g("residual"), g("random|res|"),
                g("overlap"), g("random|ovl|"), g("freq-only"))
    end
    println("-"^84)
    println("Columns after 'base F1' are F1 DROPS caused by forcing that feature set to zero.")
    println("Compare 'residual' against 'random|res|' at matched size: materially larger means the")
    println("attribution finds features a frequency ranking misses AND the model uses them. Comparable")
    println("means the residual is ranking noise and the interpretability line closes.")
end

main()
