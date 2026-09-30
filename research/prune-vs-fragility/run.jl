# QUESTION:  does pruning the attributed-but-not-frequent features help only when the LABELS are stable
#            enough for "generalise" to mean something?
# SURPRISE:  yes, and it is the difference between a result and a curiosity.
#            research/interp-residual-retrain/ found the pruning gain splits cleanly by period: +5.28 F1
#            over a size-matched control across 2019-2022, -1.12 across 2016-2018, four contiguous wins
#            against three contiguous failures. research/b3-label-drift/ independently found 2017 and 2018
#            are exactly where LAMDA's label boundary is most fragile -- 51% and 58% of malware sitting at
#            4-6 detections, within two vendor votes of being discarded, against 21-27% elsewhere. Those
#            two facts were measured for unrelated reasons and they line up.
#            If the association holds, pruning has a stated precondition and becomes deployable advice
#            rather than an unexplained regime split. If it does not, the alignment was three points
#            agreeing by chance and the paragraph is deleted rather than softened.
# ARMS:      three retrained models, evaluated PER MONTH rather than per year, and two correlations:
#              1. zero-residual  -- the pruned model.
#              2. zero-random    -- size-matched control, so the delta isolates WHICH features were
#                                   removed rather than the fact of removing 38 of them.
#              3. baseline       -- unpruned, for the absolute reference.
#            then, across months: the pruning delta against label fragility (the test), AND the pruning
#            delta against the month's malware COUNT (the control correlation). The second is what
#            distinguishes a real dependence on label quality from fragility merely proxying small,
#            noisy months -- fragility and sample size could easily move together, and without this arm a
#            correlation with fragility would not be interpretable.
# PASS/FAIL: fixed here in advance, WITH the significance requirement that research/b3-label-drift/'s
#            rule lacked and was burned by -- its year-resolution rule fired on rho=+0.548 at p=0.16 over
#            eight points and the monthly answer turned out to be -0.344, the opposite sign.
#            SUPPORTED if the delta correlates NEGATIVELY with fragility at p < 0.05 over the monthly
#            series AND the control correlation with malware count is materially weaker. REJECTED if the
#            fragility correlation is not significant, or if the count correlation is comparable, in which
#            case the label-fragility paragraph is deleted from the manuscript.
#
#   julia --project=. -t 16 research/prune-vs-fragility/run.jl [nseeds]
#
# Month resolution is not a refinement, it is the whole point: at year resolution there are seven usable
# periods and a correlation of 0.5 there cannot be distinguished from zero. This is the same mistake the
# label-drift experiment made and corrected, and repeating it would be inexcusable.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: shapley_mean
using TMCore
using Random, Printf, Statistics, Base.Threads

const TMX_DIR = joinpath("data", "lamda", "tmx")
const TRAIN_YEARS = (2013, 2014)
const EVAL_YEARS = (2016, 2017, 2018, 2019, 2020, 2021, 2022)
const CLAUSES, T, S, L, LF = 20, 10, 100, 64, 10
const EPOCHS = 30
const CI = 2
const NBG, NEX = 100, 100
const K = 100
const MIN_MONTH = 150          # a month needs enough rows for an F1 that means anything
const TM_LAB_COMMIT = "43dba5f"

function portion(year, which)
    meta = read_meta(joinpath(TMX_DIR, "$year.meta.arrow"))
    idx = [i for i in eachindex(meta.file_split) if meta.file_split[i] == String(which)]
    xs, _ = read_inputs(joinpath(TMX_DIR, "$year.tmx"); rows=idx)
    return xs, Bool[Bool(meta.label[i]) for i in idx], [meta.month[i] for i in idx],
           [meta.vt_detection[i] for i in idx]
end

function load_train()
    X = TMInput[]; Y = Bool[]
    for y in TRAIN_YEARS
        xs, ys, _, _ = portion(y, :train); append!(X, xs); append!(Y, ys)
    end
    return X, Y
end

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
        p && y && (tp += 1); p && !y && (fp += 1); !p && y && (fn += 1)
    end
    prec = tp + fp == 0 ? 0.0 : tp / (tp + fp)
    rec = tp + fn == 0 ? 0.0 : tp / (tp + fn)
    return prec + rec == 0 ? 0.0 : 200 * prec * rec / (prec + rec)
end

fit(X, Y, width, seed) = begin
    m = TMClassifier([false, true], width; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    for e in 1:EPOCHS
        train!(m, X, Y; rng=MersenneTwister(1000seed + e), parallel=:none)
    end
    m
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

"Spearman rho and a two-sided p-value from the t approximation, adequate at n in the dozens."
function spearman(x, y)
    n = length(x)
    n < 4 && return (NaN, NaN)
    rank(v) = (p = sortperm(v); r = similar(v, Float64); for (i, j) in enumerate(p); r[j] = i; end; r)
    rx, ry = rank(collect(float.(x))), rank(collect(float.(y)))
    mx, my = mean(rx), mean(ry)
    num = sum((rx .- mx) .* (ry .- my))
    den = sqrt(sum((rx .- mx) .^ 2) * sum((ry .- my) .^ 2))
    den == 0 && return (NaN, NaN)
    rho = num / den
    abs(rho) >= 1 && return (rho, 0.0)
    t = rho * sqrt((n - 2) / (1 - rho^2))
    # two-sided p from the Student-t tail, via a normal approximation with a small-sample correction
    z = abs(t) * (1 - 1 / (4 * (n - 2))) / sqrt(1 + t^2 / (2 * (n - 2)))
    p = 2 * (1 - 0.5 * (1 + sign(z) * sqrt(1 - exp(-2 * z^2 / pi))))
    return (rho, clamp(p, 0.0, 1.0))
end

function main()
    nseeds = isempty(ARGS) ? 5 : parse(Int, ARGS[1])
    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    width = length(feats)
    Xtr, Ytr = load_train()
    npos = count(Ytr); nneg = length(Ytr) - npos
    n11, n10 = feature_counts(Xtr, Ytr, width)
    freq = [abs(n11[i] / npos - n10[i] / nneg) for i in 1:width]

    @printf("LAMDA width %d  train %d  %d seeds  per-MONTH evaluation  tm-lab %s\n",
            width, length(Ytr), nseeds, TM_LAB_COMMIT)
    println("features zeroed not dropped, so width / S / s identical across arms;")
    println("both pruned arms zero exactly the same number, so they share the dead-channel effect.\n")
    flush(stdout)

    # --- monthly evaluation sets, with that month's label fragility from vt_detection ---
    months = Tuple{Int,Int}[]
    mX = Dict{Tuple{Int,Int},Vector{TMInput}}()
    mY = Dict{Tuple{Int,Int},Vector{Bool}}()
    frag = Dict{Tuple{Int,Int},Float64}()
    nmal = Dict{Tuple{Int,Int},Int}()
    for y in EVAL_YEARS
        xs, ys, mo, vt = portion(y, :test)
        for m in 1:12
            sel = [i for i in eachindex(mo) if !ismissing(mo[i]) && mo[i] == m]
            length(sel) >= MIN_MONTH || continue
            malsel = [i for i in sel if ys[i]]
            length(malsel) >= 30 || continue
            v = [vt[i] for i in malsel if !ismissing(vt[i])]
            isempty(v) && continue
            key = (y, m)
            push!(months, key)
            mX[key] = xs[sel]; mY[key] = ys[sel]
            frag[key] = 100 * mean(v .< 10)       # share a threshold of 10 would relabel benign
            nmal[key] = length(malsel)
        end
    end
    @printf("%d months with >= %d rows and >= 30 malware\n", length(months), MIN_MONTH)
    @printf("  label fragility ranges %.1f%% to %.1f%%\n\n",
            minimum(values(frag)), maximum(values(frag)))
    flush(stdout)

    deltas = Dict(k => Float64[] for k in months)
    basef1 = Dict(k => Float64[] for k in months)

    for seed in 1:nseeds
        t0 = time()
        base = fit(Xtr, Ytr, width, seed)
        bg, ex = Xtr[1:NBG], Xtr[(end - NEX + 1):end]
        shap = shapley_importance(base, ex, bg, width)
        support = [i for i in 1:width if shap[i] != 0]
        mtop = Set(topk(shap, K)); ftop = Set(topk(freq, K))
        residual = collect(setdiff(mtop, ftop))
        pool = setdiff(support, mtop)
        rng = MersenneTwister(555 + seed)
        randset = pool[randperm(rng, length(pool))[1:min(length(residual), length(pool))]]

        mres = fit(zeroed(Xtr, residual, width), Ytr, width, seed)
        mrnd = fit(zeroed(Xtr, randset, width), Ytr, width, seed)

        for key in months
            X, Y = mX[key], mY[key]
            fr = f1(mres, zeroed(X, residual, width), Y)
            fn_ = f1(mrnd, zeroed(X, randset, width), Y)
            push!(deltas[key], fr - fn_)
            push!(basef1[key], f1(base, X, Y))
        end
        @printf("  seed %d  residual %d, random %d  (%.0fs)\n",
                seed, length(residual), length(randset), time() - t0)
        flush(stdout)
    end

    d = [mean(deltas[k]) for k in months]
    fg = [frag[k] for k in months]
    nm = [Float64(nmal[k]) for k in months]
    bf = [mean(basef1[k]) for k in months]

    println("\n", "-"^72)
    @printf("%-40s %8s %9s %5s\n", "correlation across months", "rho", "p", "n")
    println("-"^72)
    r1, p1 = spearman(fg, d)
    r2, p2 = spearman(nm, d)
    r3, p3 = spearman(bf, d)
    @printf("%-40s %+8.3f %9.4f %5d\n", "pruning delta vs LABEL FRAGILITY  (test)", r1, p1, length(d))
    @printf("%-40s %+8.3f %9.4f %5d\n", "pruning delta vs malware count (control)", r2, p2, length(d))
    @printf("%-40s %+8.3f %9.4f %5d\n", "pruning delta vs baseline F1  (context)", r3, p3, length(d))
    @printf("RESULT corr frag rho=%+.4f p=%.4g n=%d\n", r1, p1, length(d))
    @printf("RESULT corr count rho=%+.4f p=%.4g n=%d\n", r2, p2, length(d))
    @printf("RESULT corr basef1 rho=%+.4f p=%.4g n=%d\n", r3, p3, length(d))
    @printf("RESULT median delta=%.3f fragility=%.2f\n", median(d), median(fg))

    println()
    lo = [d[i] for i in eachindex(d) if fg[i] <= median(fg)]
    hi = [d[i] for i in eachindex(d) if fg[i] > median(fg)]
    @printf("mean pruning delta, months with LOW  fragility (<= median %.1f%%): %+.2f F1  (n=%d)\n",
            median(fg), mean(lo), length(lo))
    @printf("mean pruning delta, months with HIGH fragility (>  median):        %+.2f F1  (n=%d)\n",
            mean(hi), length(hi))
    @printf("RESULT split low=%.3f high=%.3f n_low=%d n_high=%d\n",
            mean(lo), mean(hi), length(lo), length(hi))

    println("\n", "-"^72)
    if p1 < 0.05 && r1 < 0 && abs(r1) > abs(r2) * 1.5
        println("SUPPORTED: pruning helps more where the labels are less fragile, significantly, and the")
        println("association is not explained by month size. Pruning has a stated precondition.")
    elseif p1 >= 0.05
        println("REJECTED: the fragility association is not significant over the monthly series. Delete the")
        println("label-fragility paragraph from the manuscript rather than softening it.")
    else
        println("AMBIGUOUS: significant but not cleanly separated from the month-size control. Report as an")
        println("observation, not as a precondition.")
    end
    open(joinpath(@__DIR__, "months.csv"), "w") do io
        println(io, "year,month,fragility,malware,base_f1,delta")
        for (i, k) in enumerate(months)
            @printf(io, "%d,%d,%.3f,%d,%.3f,%.4f\n", k[1], k[2], fg[i], nmal[k], bf[i], d[i])
        end
    end
    println("wrote months.csv")
end

main()
