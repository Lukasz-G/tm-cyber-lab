# QUESTION:  does the tolerance LF -- the parameter that makes a Fuzzy-Pattern TM fuzzy -- reshape WHERE
#            the model's attribution goes, or only how well it classifies?
# SURPRISE:  yes, and it is the one question in this paper that a classical Tsetlin machine cannot be asked.
#            LF is the ceiling in the clause vote max(0, ceiling - misses). At LF = 1 the vote is
#            1[misses = 0], which is an ordinary conjunction and so a classical TM; as LF grows the clause
#            keeps voting while several literals fail. Our closed form is built directly on that structure,
#            so it can measure what tolerance does to credit assignment rather than only to accuracy.
#            Two outcomes are interesting and they point opposite ways. If raising LF SPREADS attribution
#            over more features, tolerance is buying genuine partial matching -- the clause really is
#            scoring sub-patterns -- and the attribution support should grow. If attribution instead
#            CONCENTRATES, the extra tolerance is being absorbed by a few literals doing the same work more
#            loosely, which would make LF a capacity knob rather than a representational one, and would
#            bear on the sibling finding that a fuzzy vote buys about three times the resolution of its own
#            binarisation rather than LF times.
# ARMS:      LF in {1, 2, 5, 10, 20, 40}, each under TWO threshold policies, because sweeping LF alone
#            would confound it with a mis-scaled T:
#              1. T FIXED at 10, the gate configuration's value. Isolates what LF does at constant T, but
#                 progressively under-scales T as LF grows.
#              2. T RESCALED as T = round(sqrt(CLAUSES/2 * LF)), the published relation. Isolates LF with
#                 the threshold kept where the relation says it belongs.
#            Agreement between the two arms is what licenses attributing an effect to LF; disagreement
#            localises it to the threshold instead, and either way the pair says more than either alone.
#            LF = 1 is not merely the smallest point on the sweep -- it is the classical TM, and so is the
#            control for "is any of this fuzziness-specific".
# PASS/FAIL: not a gate. REPRESENTATIONAL if the attribution support and its concentration move
#            monotonically with LF and both T policies agree. A CAPACITY KNOB if attribution is
#            essentially unchanged while only accuracy moves. CONFOUNDED if the two T policies disagree,
#            in which case nothing is attributed to LF and the experiment reports that.
#
#   julia --project=. -t 16 research/lf-attribution/run.jl [nseeds]
#
# Everything except LF and T is held at the gate configuration, so this sweep is comparable with every
# other LAMDA result in this repository.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: shapley_mean, clause_literals
using TMCore
using Random, Printf, Statistics, Base.Threads

const TMX_DIR = joinpath("data", "lamda", "tmx")
const TRAIN_YEARS = (2013, 2014)
const CLAUSES, S, L = 20, 100, 64
const EPOCHS = 30
const CI = 2
const NBG, NEX = 100, 100
const LFS = (1, 2, 5, 10, 20, 40)
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

"Share of total attribution mass held by the top-k features. Higher means more concentrated."
function topk_share(v, k)
    tot = sum(v)
    tot == 0 && return NaN
    s = sort(v; rev=true)
    return 100 * sum(s[1:min(k, length(s))]) / tot
end

"Gini coefficient of the attribution vector: 0 is perfectly even, 1 maximally concentrated."
function gini(v)
    w = sort(filter(>(0), v))
    isempty(w) && return NaN
    n = length(w); c = cumsum(w)
    return (n + 1 - 2 * sum(c) / c[end]) / n
end

function literal_stats(m)
    lits = Int[]; neg = 0; pos = 0
    for bank in (m.positive[CI], m.negative[CI]), j in 1:bank.nclauses
        l = clause_literals(bank, j)
        isempty(l) && continue
        push!(lits, length(l))
        for (_, isneg) in l
            isneg ? (neg += 1) : (pos += 1)
        end
    end
    med = isempty(lits) ? 0 : sort(lits)[cld(length(lits), 2)]
    return med, (neg + pos == 0 ? NaN : 100neg / (neg + pos))
end

function main()
    nseeds = isempty(ARGS) ? 2 : parse(Int, ARGS[1])
    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    width = length(feats)
    Xtr, Ytr = load_train()

    Xi = TMInput[]; Yi = Bool[]
    for y in TRAIN_YEARS
        xs, ys = portion(y, :test); append!(Xi, xs); append!(Yi, ys)
    end
    Xiid, Yiid = subsample(Xi, Yi, MAX_EVAL, 4242)
    Xd, Yd = portion(2021, :test)
    Xdr, Ydr = subsample(Xd, Yd, MAX_EVAL, 4242)

    @printf("LAMDA width %d  train %d  IID %d  2021 %d  %d seeds  tm-lab %s\n",
            width, length(Ytr), length(Yiid), length(Ydr), nseeds, TM_LAB_COMMIT)
    @printf("held fixed: clauses=%d S=%d L=%d epochs=%d  ceiling=LiteralCapped  parallel=:none\n",
            CLAUSES, S, L, EPOCHS)
    println("LF=1 is max(0, 1 - misses) = 1[misses=0], i.e. an ordinary conjunction: the classical TM.\n")
    flush(stdout)

    @printf("%-4s %-9s %4s %9s %10s %9s %8s %9s %8s %8s\n",
            "LF", "T policy", "T", "literals", "tolerance", "support", "top10%", "gini", "IID F1", "2021 F1")
    println("-"^92)

    for lf in LFS, policy in ("fixed", "scaled")
        T = policy == "fixed" ? 10 : max(1, round(Int, sqrt(CLAUSES / 2 * lf)))
        sup = Float64[]; t10 = Float64[]; gi = Float64[]
        f1i = Float64[]; f1d = Float64[]; med = Float64[]; negf = Float64[]
        for seed in 1:nseeds
            m = TMClassifier([false, true], width;
                             clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=lf)
            for e in 1:EPOCHS
                train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=:none)
            end
            shap = shapley_importance(m, Xtr[(end - NEX + 1):end], Xtr[1:NBG], width)
            ml, nf = literal_stats(m)
            push!(sup, count(!iszero, shap)); push!(t10, topk_share(shap, 10))
            push!(gi, gini(shap)); push!(med, ml); push!(negf, nf)
            push!(f1i, f1(m, Xiid, Yiid)); push!(f1d, f1(m, Xdr, Ydr))
        end
        tol = mean(med) == 0 ? NaN : 100 * lf / mean(med)
        @printf("%-4d %-9s %4d %9.0f %9.1f%% %9.0f %7.1f%% %9.3f %8.2f %8.2f\n",
                lf, policy, T, mean(med), tol, mean(sup), mean(t10), mean(gi), mean(f1i), mean(f1d))
        @printf("RESULT lf=%d policy=%s T=%d literals=%.0f tolerance=%.2f support=%.0f top10=%.2f gini=%.4f neg=%.2f iid_f1=%.2f drift_f1=%.2f\n",
                lf, policy, T, mean(med), tol, mean(sup), mean(t10), mean(gi), mean(negf),
                mean(f1i), mean(f1d))
        flush(stdout)
    end

    println("-"^92)
    println("'support' is how many of the 4,561 features carry nonzero exact attribution; 'top10%' is the")
    println("share of attribution mass in the ten largest; gini is 0 for an even spread and 1 for all mass")
    println("on one feature. REPRESENTATIONAL if support and concentration move with LF under BOTH T")
    println("policies; a CAPACITY KNOB if only the F1 columns move; CONFOUNDED if the policies disagree.")
end

main()
