# local-feature-strength -- Blakely & Granmo's LOCAL expression against exact Shapley.
#
# SURPRISE:  yes. interp-dataset-control already compared their GLOBAL Feature Strength (Eqn 4-6,
#            inclusion frequency over positive-polarity clauses) with the exact values and found 49/100
#            overlap. But arXiv:2007.13885 defines a second, LOCAL expression (Eqn 7-8) that is
#            evaluated per input, and we never tested it. Section 9.7 of the paper says inclusion
#            frequency "is data-independent given a fixed model, so it registers zero explanation drift
#            by construction" -- true of Eqn 4-6 and NOT of Eqn 7-8, which varies with X and can drift.
#            Anyone who knows that paper can ask the question, so it gets measured instead of argued.
#
#            Their Eqn 7, verbatim in our notation: for input X and predicted class i,
#
#              l[k, X] = SUM over positive-polarity clauses j of  C_ji(X)   where x_k = 1 and k in I_ji
#
#            so a bit gets credit when it is PRESENT in X and included NON-NEGATED in a clause that
#            fires, weighted by that clause's output. Eqn 8 aggregates bits into features. We add the
#            negated counterpart in the same shape (k in Ibar_ji and x_k = 0) as arm `both`, because
#            80% of our included literals are negated and omitting them would test a strawman.
#
# PASS/FAIL: there is no pass here, it is a measurement, but two outcomes change what the paper says.
#            If the local expression tracks the exact values (high overlap, drift series near arm 2's
#            0.294), then it IS a usable proxy at this width and Section 9.7 must say so and credit it.
#            If it does not, Section 9.7's claim stands but has to be restated about the local
#            expression specifically, not about "inclusion frequency" in general.
#
#   julia --project=. -t 16 research/local-feature-strength/run.jl [nmonths] [nseeds]
#
# The configuration is the B1 gate's, unchanged, and the month window and row budget are b2-drift's,
# so the drift series below is directly comparable to its arm 2 (fixed model, data moves, 0.294) and
# to the reported 0.958. Nothing is retuned for this experiment.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: shapley_mean, clause_literals
using TMCore
using Random, Printf, Statistics, Base.Threads

const TMX_DIR = joinpath("data", "lamda", "tmx")
const YEARS = (2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022)
const TRAIN_YEARS = (2013, 2014)
const CLAUSES, T, S, L, LF = 20, 10, 100, 64, 10
const EPOCHS = 30
const CI = 2
const NBG, NEX = 100, 100
const KS = (20, 100, 500)
const TM_LAB_COMMIT = "43dba5f"

struct MonthRef
    year::Int
    month::Int
    tr_idx::Vector{Int}
    te_idx::Vector{Int}
end

label(m, i) = Bool(m.label[i])

function month_refs(year)
    meta = read_meta(joinpath(TMX_DIR, "$year.meta.arrow"))
    out = MonthRef[]
    for mo in 1:12
        tr = [i for i in eachindex(meta.month)
              if !ismissing(meta.month[i]) && meta.month[i] == mo && meta.file_split[i] == "train"]
        te = [i for i in eachindex(meta.month)
              if !ismissing(meta.month[i]) && meta.month[i] == mo && meta.file_split[i] == "test"]
        (length(tr) >= NBG && length(te) >= NEX) || continue
        push!(out, MonthRef(year, mo, tr, te))
    end
    return out
end

function load_month(r::MonthRef)
    path = joinpath(TMX_DIR, "$(r.year).tmx")
    tr, _ = read_inputs(path; rows=r.tr_idx)
    te, _ = read_inputs(path; rows=r.te_idx)
    return tr[1:NBG], te[1:NEX]
end

function load_train_pool()
    X = TMInput[]; Y = Bool[]
    for y in TRAIN_YEARS
        meta = read_meta(joinpath(TMX_DIR, "$y.meta.arrow"))
        idx = [i for i in eachindex(meta.file_split) if meta.file_split[i] == "train"]
        xs, _ = read_inputs(joinpath(TMX_DIR, "$y.tmx"); rows=idx)
        append!(X, xs); append!(Y, Bool[label(meta, i) for i in idx])
    end
    return X, Y
end

function train_model(X, Y, width, seed)
    m = TMClassifier([false, true], width;
                     clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    for e in 1:EPOCHS
        train!(m, X, Y; rng=MersenneTwister(1000seed + e), parallel=:none)
    end
    return m
end

# ---------------------------------------------------------------------------------------------
# Blakely & Granmo, Eqn 7-8

"""
    clause_output(lits, ceil, x)

A clause's output on one input: `max(0, ceiling - misses)`, the same quantity the Shapley derivation
solves. For the classical machine of their paper the ceiling is 1 and this is the Boolean clause
output of their Eqn 7; for the fuzzy vote used here it is the graded one, which is the faithful
reading of "C_ji(X)" for this model.
"""
function clause_output(lits, ceil::Int, x::TMInput)
    misses = 0
    @inbounds for (f, neg) in lits
        sat = neg ? !x[f] : x[f]
        sat || (misses += 1)
    end
    return max(0, ceil - misses)
end

"""
    local_strength(m, x, width; negated)

Eqn 7-8 for one input. Positive-polarity clauses only, as their text requires:
"we are only interested in indices pertaining to the positive polarity clauses".

`negated = false` is Eqn 7 as written -- a bit scores only where it is PRESENT in X and included
non-negated. `negated = true` adds the mirror term for negated inclusions satisfied by absence,
which their Eqn 8 carries as the second aggregate. We report both, since this model's literals are
about 80% negated and the as-written form would ignore most of it.
"""
function local_strength(m, x::TMInput, width::Int; negated::Bool)
    out = zeros(Float64, width)
    bank = m.positive[CI]
    for j in 1:bank.nclauses
        lits = clause_literals(bank, j)
        isempty(lits) && continue
        c = Int(bank.count[j])
        ceil = c == 0 ? LF : min(c, LF)
        v = clause_output(lits, ceil, x)
        v == 0 && continue
        @inbounds for (f, neg) in lits
            if !neg && x[f]
                out[f] += v
            elseif neg && negated && !x[f]
                out[f] += v
            end
        end
    end
    return out
end

"Mean over the explained rows, matching how the exact values are aggregated."
function local_importance(m, ex, width; negated::Bool)
    parts = [zeros(Float64, width) for _ in 1:nthreads()]
    @threads for r in eachindex(ex)
        l = local_strength(m, ex[r], width; negated=negated)
        p = parts[threadid()]
        @inbounds for i in 1:width
            p[i] += l[i]
        end
    end
    return reduce(+, parts) ./ length(ex)
end

"Global Feature Strength, Eqn 4-6, for the side-by-side column."
function inclusion_frequency(m, width)
    out = zeros(Float64, width)
    bank = m.positive[CI]
    n = 0
    for j in 1:bank.nclauses
        lits = clause_literals(bank, j)
        isempty(lits) && continue
        n += 1
        for (f, _) in lits
            out[f] += 1
        end
    end
    n > 0 && (out ./= n)
    return out
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
overlap(a, b) = length(intersect(Set(a), Set(b)))
jdist(a, b) = (u = length(union(Set(a), Set(b))); u == 0 ? 1.0 : 1 - overlap(a, b) / u)

function main()
    nmonths = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10^6
    nseeds  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    width = length(feats)

    refs = MonthRef[]
    for y in YEARS, r in month_refs(y)
        push!(refs, r)
    end
    length(refs) > nmonths && (refs = refs[1:nmonths])

    X, Y = load_train_pool()
    @printf("LAMDA width %d  train %d rows  %d months  %d seeds  tm-lab %s\n",
            width, length(Y), length(refs), nseeds, TM_LAB_COMMIT)
    @printf("config clauses=%d T=%d S=%d L=%d LF=%d epochs=%d ceiling=LiteralCapped threads=%d\n\n",
            CLAUSES, T, S, L, LF, EPOCHS, nthreads())
    flush(stdout)

    # ---------------------------------------------------------------- part 1: agreement with exact
    println("PART 1 -- agreement with the exact values, on the training period's own rows")
    bg = X[1:NBG]
    ex = X[(end - NEX + 1):end]
    acc = Dict{Tuple{Symbol,Int},Vector{Int}}()
    for seed in 1:nseeds
        m = train_model(X, Y, width, seed)
        shap  = shapley_importance(m, ex, bg, width)
        lcl   = local_importance(m, ex, width; negated=false)
        lcl_b = local_importance(m, ex, width; negated=true)
        glb   = inclusion_frequency(m, width)
        @printf("seed %d: nonzero exact %d, nonzero local(as written) %d, nonzero local(both) %d\n",
                seed, count(!iszero, shap), count(!iszero, lcl), count(!iszero, lcl_b))
        for k in KS
            s = topk(shap, k)
            for (nm, v) in ((:local_written, lcl), (:local_both, lcl_b), (:global_incl, glb))
                o = overlap(s, topk(v, k))
                push!(get!(acc, (nm, k), Int[]), o)
                @printf("RESULT seed=%d k=%d rank=%s overlap=%d of %d\n", seed, k, nm, o, k)
            end
        end
        flush(stdout)
    end
    println("\n", "-"^78)
    @printf("%-6s %18s %18s %18s\n", "k", "local (Eqn 7)", "local (+negated)", "global (Eqn 4)")
    println("-"^78)
    for k in KS
        @printf("%-6d %13.1f /%3d %13.1f /%3d %13.1f /%3d\n", k,
                mean(acc[(:local_written, k)]), k,
                mean(acc[(:local_both, k)]), k,
                mean(acc[(:global_incl, k)]), k)
        @printf("RESULT mean k=%d local_written=%.2f local_both=%.2f global_incl=%.2f\n", k,
                mean(acc[(:local_written, k)]), mean(acc[(:local_both, k)]),
                mean(acc[(:global_incl, k)]))
    end
    println("-"^78)

    # ------------------------------------------------- part 2: can it drift? one fixed model, 88 months
    println("\nPART 2 -- month-over-month drift of the local expression, fixed model")
    println("Comparable to b2-drift arm 2 (exact, fixed model, data moves): mean 0.294.")
    m = train_model(X, Y, width, 1)
    prev_l = nothing; prev_lb = nothing; prev_s = nothing
    dl = Float64[]; dlb = Float64[]; ds = Float64[]
    open(joinpath(@__DIR__, "series.csv"), "w") do io
        println(io, "to_month,local_written,local_both,exact")
        for r in refs
            # The month's OWN background, which is what b2-drift arm 2 uses. A first version held the
            # training pool's first 100 rows fixed across all 88 months and the exact column came out
            # at 0.195 instead of arm 2's 0.294 -- a different quantity, and not one the paper has a
            # number for. The local expression takes no background at all, so its columns are
            # unaffected either way; that asymmetry is itself part of the finding.
            bg_m, ex_m = load_month(r)
            l  = topk(local_importance(m, ex_m, width; negated=false), 100)
            lb = topk(local_importance(m, ex_m, width; negated=true), 100)
            s  = topk(shapley_importance(m, ex_m, bg_m, width), 100)
            if prev_l !== nothing
                a, b, c = jdist(prev_l, l), jdist(prev_lb, lb), jdist(prev_s, s)
                push!(dl, a); push!(dlb, b); push!(ds, c)
                @printf(io, "%d-%02d,%.4f,%.4f,%.4f\n", r.year, r.month, a, b, c)
                @printf("  %d-%02d  local %.3f  local+neg %.3f  exact %.3f\n", r.year, r.month, a, b, c)
            end
            prev_l, prev_lb, prev_s = l, lb, s
            flush(stdout)
        end
    end
    @printf("\nRESULT drift n=%d local_written=%.4f local_both=%.4f exact=%.4f\n",
            length(dl), mean(dl), mean(dlb), mean(ds))
    @printf("  sd: local %.4f  local+neg %.4f  exact %.4f\n", std(dl), std(dlb), std(ds))
    println("\nIf the local columns sit near the exact one, their expression is a usable proxy at this")
    println("width and the paper must say so. If they sit far from it, the paper's claim stands but")
    println("has to be stated about this expression rather than about inclusion frequency.")
end

main()
