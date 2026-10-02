# QUESTION:  the paper's blacklist figure asserts that the features carrying the model "are indicators".
#            That rests on a class-conditional presence ratio and on nothing about what the features MEAN.
#            Are the attributed features things an incident responder would recognise -- permissions,
#            restricted or suspicious API calls -- or are they app-identity artefacts?
# SURPRISE:  the question is answerable here and nowhere else in this project. LAMDA ships its Drebin
#            vocabulary STRIPPED: its parquet columns are feat_0 ... feat_4560, so no semantic check is
#            possible on the primary corpus at all. APIGraph ships the real token names, and our packed
#            matrix already carries them: 1,159 entries matching the 2012 selected vocabulary exactly.
#            The design of the vocabulary makes this a sharp test rather than a browse. Of 1,159 features,
#            615 (53.1%) are activitylist_ -- Java class names of app screens, which identify an
#            application and tell a responder nothing transferable -- while the categories that constitute
#            an indicator of compromise, requestedpermissionlist / usedpermissionslist / restrictedapilist
#            / suspiciousapilist, are 267 (23.0%). So chance alone puts 23% of any feature set in the
#            indicator classes and 53% in app identity. Enrichment above 23% is evidence for the claim in
#            the paper; enrichment in activitylist_ instead refutes it and the caption must change.
# ARMS:      three rankings of the same 1,159 features, so the comparison isolates what the attribution
#            contributes over the corpus:
#              1. EXACT SHAPLEY  -- top-k by mean |phi| from the closed form. The claim under test.
#              2. FREQUENCY      -- top-k by |P(f|malware) - P(f|benign)|, no model at all. This is the
#                                  control that retracted the readability claim on LAMDA, and it is the
#                                  one that matters: if a frequency count finds the same proportion of
#                                  indicator-class features, the attribution adds no semantic value.
#              3. VOCABULARY     -- the base rate over all 1,159, which fixes what "enriched" means.
#            A fourth quantity is reported per arm and is not an arm: the share of the set that is
#            activitylist_, since app-identity features are the specific failure mode.
# PASS/FAIL: SUBSTANTIATED if the exact-attribution top-100 is enriched in the four indicator categories
#            against the 23.0% base rate AND is not dominated by activitylist_. WITHDRAW the caption if
#            the attributed set tracks the base rate or is activity-dominated. NO SEMANTIC ADVANTAGE over
#            the corpus if arm 1 and arm 2 agree, in which case the claim survives as a property of the
#            data and the paper must say so, exactly as it does for readability.
#
#   julia --project=. -t 16 research/groundtruth-overlap/run.jl [nseeds]
#
# This is measurement 5 of the five the project required before any claim about what clauses mean, and it
# is the one singled out as better in this domain than in image or text tasks because the feature names
# carry meaning. It went unrun until 2026-10-02 and is only possible on the secondary corpus.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: shapley_mean, clause_literals
using TMCore
using Random, Printf, Statistics, Base.Threads

const DIR = joinpath("data", "apigraph", "tmx", "apigraph")
const CLAUSES, T, S, L, LF = 20, 10, 100, 64, 10
const EPOCHS = 30
const CI = 2
const NBG, NEX = 100, 100
const KS = (20, 100, 500)
const TM_LAB_COMMIT = "43dba5f"

# The four Drebin categories that constitute an indicator of compromise: a capability the application
# requests or an API it reaches for. The rest name the application's own parts.
const INDICATOR = ("requestedpermissionlist", "usedpermissionslist",
                   "restrictedapilist", "suspiciousapilist")
const IDENTITY = ("activitylist", "servicelist", "broadcastreceiverlist", "contentproviderlist")

category(name) = first(split(name, '_'; limit=2))
is_indicator(name) = category(name) in INDICATOR
is_identity(name) = category(name) in IDENTITY

"Strip the category prefix so a printed example reads as the token a responder would know."
function short(name)
    parts = split(String(name), '_'; limit=2)
    return length(parts) == 2 ? parts[2] : String(name)
end

function load(stem)
    inputs, _ = read_inputs(joinpath(DIR, "$stem.tmx"))
    meta = read_meta(joinpath(DIR, "$stem.meta.arrow"))
    return inputs, collect(Bool.(meta.label))
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

"Share of `idx` in the indicator categories and in the app-identity categories, as percentages."
function composition(feats, idx)
    isempty(idx) && return (NaN, NaN)
    ind = count(i -> is_indicator(String(feats[i])), idx)
    ide = count(i -> is_identity(String(feats[i])), idx)
    return 100ind / length(idx), 100ide / length(idx)
end

function main()
    nseeds = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
    feats = read_features(joinpath(DIR, "features.arrow"))
    width = length(feats)
    Xtr, Ytr = load("train")
    npos = count(Ytr); nneg = length(Ytr) - npos
    n11, n10 = feature_counts(Xtr, Ytr, width)
    freq = [abs(n11[i] / npos - n10[i] / nneg) for i in 1:width]

    @printf("APIGraph: width %d  train %d rows, %d malware (%.1f%%)  %d seeds  tm-lab %s\n",
            width, length(Ytr), npos, 100npos / length(Ytr), nseeds, TM_LAB_COMMIT)
    @printf("clauses=%d T=%d S=%d L=%d LF=%d epochs=%d  ceiling=LiteralCapped  parallel=:none\n\n",
            CLAUSES, T, S, L, LF, EPOCHS)

    println("Vocabulary by Drebin category, which is what fixes the base rate:")
    cats = Dict{String,Int}()
    for f in feats
        c = category(String(f)); cats[c] = get(cats, c, 0) + 1
    end
    for (c, n) in sort(collect(cats); by = p -> -p[2])
        tag = c in INDICATOR ? "INDICATOR" : (c in IDENTITY ? "identity" : "")
        @printf("  %-24s %5d  %5.1f%%  %s\n", c, n, 100n / width, tag)
    end
    bi, bd = composition(feats, 1:width)
    @printf("\nBASE RATE over all %d features: indicator %.1f%%   app-identity %.1f%%\n", width, bi, bd)
    @printf("RESULT base width=%d indicator=%.2f identity=%.2f\n\n", width, bi, bd)
    flush(stdout)

    acc = Dict{Tuple{String,Int},Vector{Float64}}()
    acc2 = Dict{Tuple{String,Int},Vector{Float64}}()
    push2!(d, k, v) = (haskey(d, k) ? push!(d[k], v) : (d[k] = [v]))

    for seed in 1:nseeds
        m = TMClassifier([false, true], width;
                         clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
        for e in 1:EPOCHS
            train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=:none)
        end
        shap = shapley_importance(m, Xtr[(end - NEX + 1):end], Xtr[1:NBG], width)
        support = count(!iszero, shap)

        for k in KS
            si, sd = composition(feats, topk(shap, k))
            fi, fd = composition(feats, topk(freq, k))
            push2!(acc, ("shapley", k), si); push2!(acc2, ("shapley", k), sd)
            push2!(acc, ("frequency", k), fi); push2!(acc2, ("frequency", k), fd)
            @printf("RESULT seed=%d k=%d shapley_ind=%.2f shapley_ide=%.2f freq_ind=%.2f freq_ide=%.2f\n",
                    seed, k, si, sd, fi, fd)
        end

        if seed == 1
            println("\nThe ten features carrying the most exact attribution, seed 1, named:")
            for (rank, i) in enumerate(topk(shap, 10))
                nm = String(feats[i])
                tag = is_indicator(nm) ? "INDICATOR" : (is_identity(nm) ? "identity " : "other    ")
                @printf("  %2d. %s  %-22s %s\n", rank, tag, category(nm), short(nm))
            end
            println("\nand the ten a frequency count ranks highest, for comparison:")
            for (rank, i) in enumerate(topk(freq, 10))
                nm = String(feats[i])
                tag = is_indicator(nm) ? "INDICATOR" : (is_identity(nm) ? "identity " : "other    ")
                @printf("  %2d. %s  %-22s %s\n", rank, tag, category(nm), short(nm))
            end
            println()
        end
        @printf("seed %d done (attribution support %d of %d)\n", seed, support, width)
        flush(stdout)
    end

    g(d, a, k) = mean(d[(a, k)])
    println("\n", "-"^74)
    @printf("%-11s %5s %14s %16s\n", "ranking", "k", "indicator %", "app-identity %")
    println("-"^74)
    for k in KS, a in ("shapley", "frequency")
        @printf("%-11s %5d %13.1f%% %15.1f%%\n", a, k, g(acc, a, k), g(acc2, a, k))
        @printf("RESULT mean ranking=%s k=%d indicator=%.2f identity=%.2f\n",
                a, k, g(acc, a, k), g(acc2, a, k))
    end
    println("-"^74)
    @printf("base rate  %5s %13.1f%% %15.1f%%\n", "all", bi, bd)
    println()
    println("SUBSTANTIATED if the shapley rows are enriched in the indicator categories over the base")
    println("rate and are not activity-dominated. WITHDRAW the paper's caption if they track the base")
    println("rate or are identity-dominated. NO SEMANTIC ADVANTAGE over the corpus if shapley and")
    println("frequency agree, which would place this beside the readability retraction.")
end

main()
