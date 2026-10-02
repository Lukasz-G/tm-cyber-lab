# QUESTION:  is the pruning gain in research/interp-residual-retrain/ a fact about WHICH features the exact
#            attribution picks out, or merely a fact about HOW DENSE they are?
# SURPRISE:  either answer kills something. The published comparison zeroes 38 attributed-but-not-frequent
#            features and compares against 38 drawn uniformly from the nonzero-attribution support, matched
#            on count. But zeroing is NOT sign-neutral: it removes the evidence for a positive literal and
#            SATISFIES a negated one. And the two sets are not matched on the variable that decides which
#            of those happens. research/blacklist-anatomy/ measured the residual at a training-period
#            presence rate of 11.2% against 1.5-2.3% for a uniform draw, and at 79.5% negated literals
#            against 94.8% -- because a feature that is almost always absent is included negated. So the
#            uniform control receives a systematically weaker intervention than the residual does, and the
#            published +5.28 F1 over 2019-2022 could be reporting density, not identity.
#            If a density-matched control gains as much as the residual, the drift-forensics claim is an
#            artefact of how much the zeroing does, and it must be withdrawn. If it does not, the claim
#            survives its own strongest objection and the caveat currently in the manuscript is deleted
#, not apologised for.
# ARMS:      four, all RETRAINED from scratch per seed. Arms 1-3 reproduce the published comparison exactly
#            so the new arm is read on the same scale, and the reproduction is itself a check.
#              1. BASELINE   -- all features.
#              2. ZERO-RESID -- the 38 attributed-but-not-frequent features forced to zero.
#              3. ZERO-UNIF  -- 38 drawn uniformly from the nonzero-attribution support. The PUBLISHED
#                               control: matched on count only.
#              4. ZERO-DENS  -- 38 from the same pool, matched feature-by-feature on training-period
#                               PRESENCE RATE. The isolating arm. Presence rate is chosen and not sign
#                               composition because it is upstream: it is model-free, and it is what causes
#                               the sign difference. The achieved negated share is therefore an OUTCOME
#                               here and is printed, not an objective that was optimised for.
#            zero-freqonly is dropped from the published set: its question (does the mirror arm survive
#            retraining) is settled, and the compute goes to the arm that can overturn a live claim.
# PASS/FAIL: the claim SURVIVES if resid - dens stays positive across 2019-2022 at a magnitude comparable
#            to resid - unif. It is an ARTEFACT OF DENSITY if resid - dens collapses toward zero while
#            resid - unif reproduces, in which case interp-residual-retrain's column is restated as a
#            density effect and the paper's pruning subsection becomes a diagnosis only.
#            Reported either way, and the match quality is reported with it: if the pool cannot supply 38
#            features at the residual's presence rate the arm is weak, and the printed mismatch says so
#            instead of letting a failed match pass as a null result.
#
#   julia --project=. -t 16 research/residual-density-control/run.jl [nseeds]
#
# Features are ZEROED and not dropped, for the reason the published run gives: deleting 38 of 4,561 columns
# changes the input width, and both the effective specificity s = width/S and the L growth gate move with
# width. Zeroing holds width, S and s identical across all four arms, and arms 2-4 each zero exactly 38
# features so they share the dead-channel perturbation. Arm 1 does not carry it, which is why every verdict
# here is read between arms 2, 3 and 4.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: shapley_mean, clause_literals
using TMCore
using Random, Printf, Statistics, Base.Threads

const TMX_DIR = joinpath("data", "lamda", "tmx")
const TRAIN_YEARS = (2013, 2014)
const EVAL_YEARS = (2016, 2017, 2018, 2019, 2020, 2021, 2022)
const DRIFTED = ("2016", "2017", "2018", "2019", "2020", "2021", "2022")
const LATE = ("2019", "2020", "2021", "2022")
const EARLY = ("2016", "2017", "2018")
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

"Copy of `X` with every feature in `drop` forced to zero, at the same width, kept packed."
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

"Per-feature malware and benign presence counts over the training pool."
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

"""
Per-feature tally of included literals in `m` for class `ci`: how many mention the feature negated and
how many positive. Used only to REPORT the sign composition each arm ends up with; nothing selects on it.
"""
function literal_signs(m, ci, width)
    neg = zeros(Int, width); pos = zeros(Int, width)
    for bank in (m.positive[ci], m.negative[ci]), j in 1:bank.nclauses
        for (f, isneg) in clause_literals(bank, j)
            1 <= f <= width || continue
            isneg ? (neg[f] += 1) : (pos[f] += 1)
        end
    end
    return neg, pos
end

"Negated share (%) of the included literals touching `feats`, and how many literals that is."
function sign_profile(neg, pos, feats)
    n = 0; p = 0
    for f in feats
        n += neg[f]; p += pos[f]
    end
    return (n + p == 0 ? NaN : 100n / (n + p)), n + p
end

"""
Pick `length(target)` features from `pool`, one per target feature, each the unused pool feature whose
training-period presence rate is closest to that target's.

Greedy nearest-neighbour without replacement, taking the target features in descending density so the
hardest ones (the densest, where the pool is thinnest) are served first. Returns the selection and the
median absolute density mismatch, which is what says whether the arm is strong enough to read.
"""
function density_matched(pool, dens, target)
    avail = Set(pool)
    sel = Int[]; errs = Float64[]
    for r in sort(collect(target); by = f -> -dens[f])
        best = 0; bd = Inf
        for f in avail
            d = abs(dens[f] - dens[r])
            if d < bd
                bd = d; best = f
            end
        end
        best == 0 && break
        push!(sel, best); push!(errs, bd); delete!(avail, best)
    end
    return sel, (isempty(errs) ? NaN : median(errs))
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
    nrows = length(Ytr)
    npos = count(Ytr); nneg = nrows - npos
    n11, n10 = feature_counts(Xtr, Ytr, width)
    freq = [abs(n11[i] / npos - n10[i] / nneg) for i in 1:width]
    dens = [(n11[i] + n10[i]) / nrows for i in 1:width]

    @printf("LAMDA width %d  train %d  %d seeds  k=%d  RETRAINED per arm\n",
            width, nrows, nseeds, K)
    @printf("clauses=%d T=%d S=%d L=%d LF=%d epochs=%d  ceiling=LiteralCapped  parallel=:none  tm-lab %s\n",
            CLAUSES, T, S, L, LF, EPOCHS, TM_LAB_COMMIT)
    println("Arms 1-3 reproduce research/interp-residual-retrain/. Arm 4 is the isolating control:")
    println("38 features matched FEATURE BY FEATURE on training-period presence rate, from the same pool")
    println("the uniform control draws from. Sign composition is reported, never selected on.\n")
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

    ARMS = ("baseline", "zero-resid", "zero-unif", "zero-dens")
    res = Dict{Tuple{String,String},Vector{Float64}}()
    push_res!(k, v) = (haskey(res, k) ? push!(res[k], v) : (res[k] = [v]))
    prof = Dict{String,Vector{Float64}}()
    push_prof!(k, v) = (haskey(prof, k) ? push!(prof[k], v) : (prof[k] = [v]))

    for seed in 1:nseeds
        t0 = time()
        base = fit(Xtr, Ytr, width, seed)

        bg, ex = Xtr[1:NBG], Xtr[(end - NEX + 1):end]
        shap = shapley_importance(base, ex, bg, width)
        support = [i for i in 1:width if shap[i] != 0]
        mtop = Set(topk(shap, K)); ftop = Set(topk(freq, K))
        residual = collect(setdiff(mtop, ftop))
        pool = collect(setdiff(support, mtop))

        rng = MersenneTwister(555 + seed)
        unif = pool[randperm(rng, length(pool))[1:min(length(residual), length(pool))]]
        densmatched, mismatch = density_matched(pool, dens, residual)

        neg, pos = literal_signs(base, CI, width)
        sets = Dict("zero-resid" => residual, "zero-unif" => unif, "zero-dens" => densmatched)

        @printf("seed %d: support %d  pool %d  residual %d  (%.0fs to train+attribute)\n",
                seed, length(support), length(pool), length(residual), time() - t0)
        for a in ("zero-resid", "zero-unif", "zero-dens")
            sf, nl = sign_profile(neg, pos, sets[a])
            md = mean(dens[f] for f in sets[a])
            push_prof!(a * ":dens", 100md); push_prof!(a * ":neg", sf)
            @printf("  %-11s n=%2d  mean presence %5.2f%%  negated %5.1f%%  literals %4d\n",
                    a, length(sets[a]), 100md, sf, nl)
            @printf("RESULT seed=%d set=%s n=%d presence=%.3f negated=%.2f literals=%d\n",
                    seed, a, length(sets[a]), 100md, sf, nl)
        end
        @printf("  density match: median |dens error| = %.4f pp\n", 100mismatch)
        @printf("RESULT seed=%d density_match_median_pp=%.4f\n", seed, 100mismatch)
        flush(stdout)

        drops = Dict("baseline" => Int[], "zero-resid" => residual,
                     "zero-unif" => unif, "zero-dens" => densmatched)

        for arm in ARMS
            d = drops[arm]
            m = arm == "baseline" ? base : fit(zeroed(Xtr, d, width), Ytr, width, seed)
            for k in order
                X, Y = evalsets[k]
                score = f1(m, zeroed(X, d, width), Y)
                push_res!((arm, k), score)
                @printf("RESULT seed=%d arm=%s eval=%s f1=%.2f\n", seed, arm, k, score)
            end
            @printf("  %-11s done\n", arm)
            flush(stdout)
        end
    end

    g(a, k) = haskey(res, (a, k)) ? mean(res[(a, k)]) : NaN

    println("\n", "-"^92)
    @printf("%-6s %10s %11s %10s %10s %14s %14s\n",
            "eval", "baseline", "zero-resid", "zero-unif", "zero-dens",
            "resid - unif", "resid - dens")
    println("-"^92)
    for k in order
        du = g("zero-resid", k) - g("zero-unif", k)
        dd = g("zero-resid", k) - g("zero-dens", k)
        @printf("%-6s %10.2f %11.2f %10.2f %10.2f %+14.2f %+14.2f\n",
                k, g("baseline", k), g("zero-resid", k), g("zero-unif", k), g("zero-dens", k), du, dd)
        @printf("RESULT mean eval=%s baseline=%.2f resid=%.2f unif=%.2f dens=%.2f d_unif=%+.2f d_dens=%+.2f\n",
                k, g("baseline", k), g("zero-resid", k), g("zero-unif", k), g("zero-dens", k), du, dd)
    end
    println("-"^92)

    blk(ks, a, b) = mean(g(a, k) - g(b, k) for k in ks)
    for (nm, ks) in (("2019-2022", LATE), ("2016-2018", EARLY))
        du = blk(ks, "zero-resid", "zero-unif")
        dd = blk(ks, "zero-resid", "zero-dens")
        @printf("%-10s  resid - unif %+6.2f   resid - dens %+6.2f\n", nm, du, dd)
        @printf("RESULT block=%s d_unif=%+.2f d_dens=%+.2f\n", nm, du, dd)
    end

    mp(k) = haskey(prof, k) ? mean(filter(!isnan, prof[k])) : NaN
    println()
    @printf("sign composition, mean over seeds (reported, not selected on):\n")
    for a in ("zero-resid", "zero-unif", "zero-dens")
        @printf("  %-11s presence %5.2f%%  negated %5.1f%%\n", a, mp(a * ":dens"), mp(a * ":neg"))
    end
    @printf("RESULT profile resid_neg=%.2f unif_neg=%.2f dens_neg=%.2f resid_pres=%.3f unif_pres=%.3f dens_pres=%.3f\n",
            mp("zero-resid:neg"), mp("zero-unif:neg"), mp("zero-dens:neg"),
            mp("zero-resid:dens"), mp("zero-unif:dens"), mp("zero-dens:dens"))

    println()
    println("VERDICT rule: the drift-forensics claim SURVIVES if 'resid - dens' over 2019-2022 stays")
    println("positive and comparable to 'resid - unif'. It is an ARTEFACT OF DENSITY if 'resid - dens'")
    println("collapses toward zero while 'resid - unif' reproduces the published +5.28. Read the density")
    println("match error first: a large one means the pool could not supply the control and the arm is")
    println("weak, not null.")
end

main()
