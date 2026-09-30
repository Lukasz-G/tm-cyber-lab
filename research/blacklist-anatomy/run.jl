# QUESTION:  when ~80% of a clause's literals require a feature to be ABSENT, is the detector discriminating
#            on the absence of BENIGN-characteristic features -- a "not-goodware" detector -- or are the
#            negated literals near-vacuous, satisfied by almost anything?
# SURPRISE:  yes either way, and the paper currently asserts an answer it has not measured.
#            research/interp-dataset-control/ established the sign composition (about four in five included
#            literals negated) and concluded that the clauses give an analyst no indicator of compromise.
#            That conclusion is right under either reading, but the two are different models of the world.
#            If the negated literals sit on features that are COMMON IN BENIGN and RARE IN MALWARE, the
#            detector has learned a coherent and reportable rule -- "this lacks what goodware has" -- which
#            is weak as an IOC but is genuine discriminative structure, and it would change how the
#            interpretability section should be written. If instead they sit on features that are rare
#            everywhere, they are satisfied by roughly any input and the clause's real work is done by its
#            handful of positive literals, which is a different and worse story.
#            There is also a methodological consequence. This project ablates features BY ZEROING THEM. For
#            a positive literal zeroing removes evidence, but for a NEGATED literal zeroing SATISFIES it.
#            Zeroing is therefore not a neutral intervention on a blacklist model, and the sign composition
#            of an ablated set determines which direction the intervention pushes. That bears directly on
#            research/interp-residual/ and interp-residual-retrain/, and it has not been checked.
# ARMS:      three feature sets, compared on the same two axes (class-conditional presence, and the sign
#            with which they enter clauses), so that "negated" is never read without knowing what the
#            feature's base rates are:
#              1. the exact-attribution top-100 -- what the model relies on.
#              2. the RESIDUAL and OVERLAP partitions of it, separately, because the ablation experiments
#                 treat these as interchangeable apart from size and they may not be.
#              3. a random control drawn from the nonzero-attribution support, which fixes what "rare
#                 everywhere" looks like on this data -- without it, a low presence rate cannot be read.
# PASS/FAIL: not a gate; it decides how the interpretability section must be written. NOT-GOODWARE if the
#            top attributed features are markedly more common in benign than in malware and enter
#            predominantly as negated literals. VACUOUS if their presence rates are near the random
#            control's, in which case the negated literals carry little and the positive ones do the work.
#
#   julia --project=. -t 16 research/blacklist-anatomy/run.jl [nseeds]
#
# Writes features.csv: one row per top-100 feature with its class-conditional presence rates, its signed
# literal counts and its exact attribution, so the figure can be drawn from it without retraining.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: shapley_mean, clause_literals
using TMCore
using Random, Printf, Statistics, Base.Threads

const TMX_DIR = joinpath("data", "lamda", "tmx")
const TRAIN_YEARS = (2013, 2014)
const CLAUSES, T, S, L, LF = 20, 10, 100, 64, 10
const EPOCHS = 30
const CI = 2
const NBG, NEX = 100, 100
const K = 100
const TM_LAB_COMMIT = "43dba5f"

function load_train()
    X = TMInput[]; Y = Bool[]
    for y in TRAIN_YEARS
        meta = read_meta(joinpath(TMX_DIR, "$y.meta.arrow"))
        idx = [i for i in eachindex(meta.file_split) if meta.file_split[i] == "train"]
        xs, _ = read_inputs(joinpath(TMX_DIR, "$y.tmx"); rows=idx)
        append!(X, xs); append!(Y, Bool[Bool(meta.label[i]) for i in idx])
    end
    return X, Y
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

"Signed literal counts per feature across BOTH polarity banks of the explained class."
function literal_signs(m, width)
    pos = zeros(Int, width); neg = zeros(Int, width)
    for bank in (m.positive[CI], m.negative[CI]), j in 1:bank.nclauses
        for (f, isneg) in clause_literals(bank, j)
            isneg ? (neg[f] += 1) : (pos[f] += 1)
        end
    end
    return pos, neg
end

topk(v, k) = partialsortperm(v, 1:min(k, length(v)); rev=true)

function describe(name, idx, pmal, pben, pos, neg)
    isempty(idx) && return
    nneg = sum(neg[i] for i in idx); npos = sum(pos[i] for i in idx)
    benign_leaning = count(i -> pben[i] > pmal[i], idx)
    @printf("%-22s n=%3d  P(f|mal) %6.3f  P(f|ben) %6.3f  benign-leaning %3d/%-3d  negated %5.1f%%\n",
            name, length(idx), mean(pmal[i] for i in idx), mean(pben[i] for i in idx),
            benign_leaning, length(idx), 100nneg / max(1, nneg + npos))
    @printf("RESULT set=%s n=%d p_mal=%.4f p_ben=%.4f benign_leaning=%d negated_pct=%.2f\n",
            name, length(idx), mean(pmal[i] for i in idx), mean(pben[i] for i in idx),
            benign_leaning, 100nneg / max(1, nneg + npos))
end

function main()
    nseeds = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    width = length(feats)
    X, Y = load_train()
    nm = count(Y); nb = length(Y) - nm
    n11, n10 = feature_counts(X, Y, width)
    pmal = n11 ./ nm            # P(feature present | malware)
    pben = n10 ./ nb            # P(feature present | benign)
    freq = abs.(pmal .- pben)

    @printf("LAMDA width %d  train %d (%d malware / %d benign)  overall density %.2f%%\n",
            width, length(Y), nm, nb, 100 * mean(pmal) )
    @printf("clauses=%d T=%d S=%d L=%d LF=%d  ceiling=LiteralCapped  tm-lab %s  %d seeds\n\n",
            CLAUSES, T, S, L, LF, TM_LAB_COMMIT, nseeds)
    flush(stdout)

    rows = String[]
    for seed in 1:nseeds
        m = TMClassifier([false, true], width;
                         clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
        for e in 1:EPOCHS
            train!(m, X, Y; rng=MersenneTwister(1000seed + e), parallel=:none)
        end
        shap = shapley_importance(m, X[1:NBG], X[(end - NEX + 1):end], width)
        pos, neg = literal_signs(m, width)
        support = [i for i in 1:width if shap[i] != 0]

        mtop = topk(shap, K)
        ftop = Set(topk(freq, K))
        residual = [i for i in mtop if !(i in ftop)]
        overlap  = [i for i in mtop if i in ftop]
        pool = setdiff(support, Set(mtop))
        rng = MersenneTwister(555 + seed)
        control = pool[randperm(rng, length(pool))[1:min(K, length(pool))]]

        @printf("seed %d\n", seed)
        describe("top-100 attributed", collect(mtop), pmal, pben, pos, neg)
        describe("  overlap (62)", overlap, pmal, pben, pos, neg)
        describe("  residual (38)", residual, pmal, pben, pos, neg)
        describe("random control", control, pmal, pben, pos, neg)
        describe("ALL features", collect(1:width), pmal, pben, pos, neg)
        println()
        flush(stdout)

        if seed == 1
            for i in mtop
                push!(rows, @sprintf("%d,%s,%.6f,%.6f,%d,%d,%.6f,%s",
                        i, replace(String(feats[i]), ',' => ';'), pmal[i], pben[i],
                        pos[i], neg[i], shap[i], (i in ftop) ? "overlap" : "residual"))
            end
        end
    end

    open(joinpath(@__DIR__, "features.csv"), "w") do io
        println(io, "feature_index,feature_name,p_malware,p_benign,n_positive_literals,n_negated_literals,shapley,partition")
        for r in rows
            println(io, r)
        end
    end
    println("wrote features.csv (seed 1 top-100)")
    println()
    println("NOT-GOODWARE if the top attributed features are markedly more common in BENIGN than in")
    println("malware and enter mostly as negated literals. VACUOUS if their presence rates sit near the")
    println("random control's, in which case the negated literals carry little.")
end

main()
