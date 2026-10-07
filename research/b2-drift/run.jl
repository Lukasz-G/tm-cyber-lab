# QUESTION:  measured exactly, with no estimator noise at all, how much does a malware detector's
#            explanation actually turn over from month to month?
# SURPRISE:  yes, either way. LAMDA reports Jaccard ~0.9 between
#            consecutive months over top-100 SHAP features and reads it as explanation drift;
#            research/b0-noise-floor/ showed that figure sits at its own estimator's noise floor
#            (0.958 reported against a 0.926 floor, cause localised to coalition sampling). That
#            established the published number cannot see a signal -- it did NOT establish what the
#            signal is. Exact attribution has no sampling noise by construction, so whatever churn
#            survives here is real. A LOW number says the published finding was an artefact and
#            explanations are stable, which is the result this project was set up to test. A HIGH
#            number says explanation drift is a property of the problem and not of the estimator,
#            which is a STRONGER result for the drift literature and, per the pre-registration, must
#            then be reported as the headline, not buried.
# ARMS:      the pre-registered five (docs/clause-attribution.md section 4), and the comparison
#            BETWEEN them is the result, so none is a control for the others in the usual sense:
#              1. noise floor -- same month, two seeds. For a TM this is model variation only, since
#                 the attribution itself is deterministic. It is the denominator for arms 2 and 3.
#              2. fixed model trained on 2013-14, attributions recomputed per month -> DATA drift
#                 alone. This is the quantity the published measurement cannot isolate.
#              3. model refitted per month, as the reference does. Arm 3 minus arm 2 is the
#                 contribution of refitting, which the published number folds in silently.
#              5. arm 2 over the month's whole test split instead of 100 rows, to show how much of
#                 any residual churn is the 100-row sample, not the coalition sampling.
#            Arm 4 (sampled KernelExplainer against the closed form on the identical model) needs
#            shap and therefore Python; it lives in arm4.py and is reported alongside.
# PASS/FAIL: not a gate -- this is the measurement the paper is about, and every outcome is
#            reportable. The pre-registered falsification conditions are what this is judged
#            against, and they are quoted in the README and not restated here so they cannot
#            drift. What would invalidate the run itself: arm 1 coming out at the same level as
#            arms 2 and 3, which would mean seed variation swamps everything and the design cannot
#            separate data drift from model variation at this clause budget.
#
#   julia --project=. -t 16 research/b2-drift/run.jl [nmonths] [nseeds]
#
# Protocol is fixed by docs/clause-attribution.md, pre-registered 2026-09-19, and is matched to the
# reference wherever matching is possible: 100 background rows from the month's train portion, 100
# explained rows from its test portion, importance = mean over explained rows of |phi|, Jaccard
# 1-|∩|/|∪| and Kendall (1-tau)/2 over the union's importance values, k in {100, 1000}, consecutive
# months, window 2013-2022. Later years are excluded because LAMDA's 2024 and 2025 malware counts
# are 794 and 23 against ~45,000 benign per year, which is antivirus label lag, not drift.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: shapley_mean, clause_literals
using TMCore
using Random, Printf, Statistics, Base.Threads

const TMX_DIR = joinpath("data", "lamda", "tmx")
const YEARS = (2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022)  # LAMDA has no 2015
const TRAIN_YEARS = (2013, 2014)

# The B1 gate configuration, unchanged. Attribution is being measured on the model the detection
# result was established for; retuning it here would make the two incomparable.
const CLAUSES, T, S, L, LF = 20, 10, 100, 64, 10
const EPOCHS = 30
const PARALLEL = :none          # training. Attribution threads separately and bit-identically.
const CI = 2                    # class index of `true` in [false, true]: malware
const NBG, NEX = 100, 100       # background and explained row counts, matching the reference
const KS = (100, 1000)
const TM_LAB_COMMIT = "43dba5f"
const CEILING = "LiteralCapped"

# ---------------------------------------------------------------------------------------------
# data

# Only the row INDICES are held between months. An earlier version eagerly loaded every month's full
# train and test split before starting, which for the whole 2013-2022 window is essentially all of
# LAMDA resident at once -- it passed 1.25 GB and was still loading when it was killed. Nothing here
# needs two months in memory simultaneously: each month's importance vectors are what the arms compare,
# and those are 4,561 floats.
struct MonthRef
    year::Int
    month::Int
    tr_idx::Vector{Int}
    te_idx::Vector{Int}
end

struct MonthData
    bg::Vector{TMInput}      # background: first NBG rows of the month's train portion
    ex::Vector{TMInput}      # explained:  first NEX rows of the month's test portion
    ex_full::Vector{TMInput} # the whole test portion, for arm 5
    tr::Vector{TMInput}      # the month's train portion, for arm 3's refit
    try_::Vector{Bool}
end

label(m, i) = Bool(m.label[i])

"Every month of `year` that has at least NBG train and NEX test rows, in calendar order. Indices only."
function month_refs(year)
    meta = read_meta(joinpath(TMX_DIR, "$year.meta.arrow"))
    out = MonthRef[]
    for mo in 1:12
        tr_idx = [i for i in eachindex(meta.month)
                  if !ismissing(meta.month[i]) && meta.month[i] == mo &&
                     meta.file_split[i] == "train"]
        te_idx = [i for i in eachindex(meta.month)
                  if !ismissing(meta.month[i]) && meta.month[i] == mo &&
                     meta.file_split[i] == "test"]
        (length(tr_idx) >= NBG && length(te_idx) >= NEX) || continue
        push!(out, MonthRef(year, mo, tr_idx, te_idx))
    end
    return out
end

function load_month(r::MonthRef)
    meta = read_meta(joinpath(TMX_DIR, "$(r.year).meta.arrow"))
    path = joinpath(TMX_DIR, "$(r.year).tmx")
    tr, _ = read_inputs(path; rows=r.tr_idx)
    te, _ = read_inputs(path; rows=r.te_idx)
    return MonthData(tr[1:NBG], te[1:NEX], te, tr,
                     Bool[label(meta, i) for i in r.tr_idx])
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

# ---------------------------------------------------------------------------------------------
# model and attribution

function train_model(X, Y, width, seed)
    m = TMClassifier([false, true], width;
                     clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    for e in 1:EPOCHS
        train!(m, X, Y; rng=MersenneTwister(1000seed + e), parallel=PARALLEL)
    end
    return m
end

"""
Per-feature importance: mean over explained rows of the absolute exact Shapley value against the
background set. This is the reference's aggregation (`mean(abs(shap[:, 1]))`), so the two are
comparable. Threaded across explained rows -- each row's vector is private and attribution writes
no model state, so the result is bit-identical at any thread count.
"""
function importance(model, ex, bg, width)
    parts = [zeros(Float64, width) for _ in 1:nthreads()]
    @threads for r in eachindex(ex)
        phi = shapley_mean(model, CI, ex[r], bg)
        p = parts[threadid()]
        @inbounds for i in 1:width
            p[i] += abs(phi[i])
        end
    end
    total = reduce(+, parts)
    return total ./ length(ex)
end

topk(imp, k) = partialsortperm(imp, 1:min(k, length(imp)); rev=true)

jaccard(a, b) = (sa = Set(a); sb = Set(b); u = length(union(sa, sb));
                 u == 0 ? 1.0 : 1 - length(intersect(sa, sb)) / u)

"Kendall tau-b over the union's importance VALUES, then (1 - tau)/2, as the reference computes it."
function kendall_dist(a, b, imp_a, imp_b)
    u = collect(union(Set(a), Set(b)))
    n = length(u)
    n < 2 && return 1.0
    xa = [imp_a[i] for i in u]; xb = [imp_b[i] for i in u]
    C = 0; D = 0; ta = 0; tb = 0
    for i in 1:(n - 1), j in (i + 1):n
        da = xa[i] - xa[j]; db = xb[i] - xb[j]
        if da == 0 && db == 0
            ta += 1; tb += 1
        elseif da == 0
            ta += 1
        elseif db == 0
            tb += 1
        else
            sign(da) == sign(db) ? (C += 1) : (D += 1)
        end
    end
    n0 = n * (n - 1) ÷ 2
    den = sqrt(max(0.0, (n0 - ta)) * max(0.0, (n0 - tb)))
    den == 0 && return 1.0
    tau = (C - D) / den
    return (1 - tau) / 2
end

# ---------------------------------------------------------------------------------------------

summarise(v) = isempty(v) ? (NaN, NaN, 0) : (mean(v), length(v) < 2 ? 0.0 : std(v), length(v))

function report(name, jac, ken, k)
    mj, sj, n = summarise(jac); mk, _, _ = summarise(ken)
    @printf("%-34s k=%-5d %8.3f %7.3f %7.3f %6d\n", name, k, mj, sj, mk, n)
    @printf("RESULT arm=%s k=%d jaccard=%.4f sd=%.4f kendall=%.4f n=%d\n", name, k, mj, sj, mk, n)
end

function main()
    nmonths = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : typemax(Int)
    nseeds  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2

    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    width = length(feats)

    months = MonthRef[]
    for y in YEARS
        append!(months, month_refs(y))
        length(months) >= nmonths && break
    end
    length(months) > nmonths && (months = months[1:nmonths])

    @printf("LAMDA width %d  %d months %d-%02d..%d-%02d  %d seeds\n",
            width, length(months), months[1].year, months[1].month,
            months[end].year, months[end].month, nseeds)
    @printf("gate config: clauses=%d T=%d S=%d L=%d LF=%d epochs=%d  ceiling=%s  train parallel=%s\n",
            CLAUSES, T, S, L, LF, EPOCHS, CEILING, PARALLEL)
    @printf("tm-lab %s  attribution threads=%d (bit-identical)  background=%d explained=%d\n\n",
            TM_LAB_COMMIT, nthreads(), NBG, NEX)
    flush(stdout)

    # --- the fixed model, for arms 2 and 5 ---
    Xtr, Ytr = load_train_pool()
    @printf("fixed model: training on %d rows from %s ... ", length(Ytr), string(TRAIN_YEARS))
    flush(stdout)
    t0 = time()
    fixed = train_model(Xtr, Ytr, width, 1)
    lits = Int[]
    for bank in (fixed.positive[CI], fixed.negative[CI]), j in 1:bank.nclauses
        n = length(clause_literals(bank, j)); n > 0 && push!(lits, n)
    end
    @printf("%.1fs, median %d literals/clause, LF/literals = %.1f%%\n\n",
            time() - t0, isempty(lits) ? 0 : median(lits),
            isempty(lits) ? NaN : 100LF / median(lits))
    @printf("RESULT fixed_model median_literals=%d tolerance_pct=%.2f\n\n",
            isempty(lits) ? 0 : median(lits), isempty(lits) ? NaN : 100LF / median(lits))

    # --- per-month importances ---
    imp_fixed = Vector{Vector{Float64}}(undef, length(months))     # arm 2
    imp_full  = Vector{Vector{Float64}}(undef, length(months))     # arm 5
    imp_refit = [Vector{Vector{Float64}}(undef, length(months)) for _ in 1:nseeds]  # arms 3, 1

    for (i, r) in enumerate(months)
        t0 = time()
        mo = load_month(r)
        imp_fixed[i] = importance(fixed, mo.ex, mo.bg, width)
        imp_full[i]  = importance(fixed, mo.ex_full, mo.bg, width)
        for s in 1:nseeds
            mdl = train_model(mo.tr, mo.try_, width, s)
            imp_refit[s][i] = importance(mdl, mo.ex, mo.bg, width)
        end
        @printf("  %d-%02d  train=%5d test=%5d  %.1fs\n",
                r.year, r.month, length(mo.try_), length(mo.ex_full), time() - t0)
        flush(stdout)
        mo = nothing                 # the month's rows are not needed again
        i % 12 == 0 && GC.gc()
    end

    # --- the arms ---
    println("\n", "-"^78)
    @printf("%-34s %-7s %8s %7s %7s %6s\n",
            "arm", "k", "jaccard", "sd", "kendall", "n")
    println("-"^78)

    # Per-pair values are also written to series.csv, because the arm means below are what the paper
    # tables report but the month-over-month CURVE is the thing a reader should actually see, and it
    # cannot be recovered from a mean.
    series = NamedTuple{(:k, :arm, :from, :to, :jaccard, :kendall),
                        Tuple{Int,String,String,String,Float64,Float64}}[]
    label(i) = @sprintf("%d-%02d", months[i].year, months[i].month)

    for k in KS
        tk_fixed = [topk(imp_fixed[i], k) for i in eachindex(months)]
        tk_full  = [topk(imp_full[i], k) for i in eachindex(months)]
        tk_refit = [[topk(imp_refit[s][i], k) for i in eachindex(months)] for s in 1:nseeds]

        # arm 1 -- same month, two seeds. Model variation only; attribution is deterministic.
        j1 = Float64[]; k1 = Float64[]
        for i in eachindex(months), s in 1:(nseeds - 1), s2 in (s + 1):nseeds
            push!(j1, jaccard(tk_refit[s][i], tk_refit[s2][i]))
            push!(k1, kendall_dist(tk_refit[s][i], tk_refit[s2][i],
                                   imp_refit[s][i], imp_refit[s2][i]))
        end
        report("1 noise floor (2 seeds, same month)", j1, k1, k)

        # arm 2 -- fixed model, consecutive months. Data drift alone.
        j2 = Float64[]; k2 = Float64[]
        for i in 1:(length(months) - 1)
            jv = jaccard(tk_fixed[i], tk_fixed[i + 1])
            kv = kendall_dist(tk_fixed[i], tk_fixed[i + 1], imp_fixed[i], imp_fixed[i + 1])
            push!(j2, jv); push!(k2, kv)
            push!(series, (k=k, arm="2-fixed", from=label(i), to=label(i + 1),
                           jaccard=jv, kendall=kv))
        end
        report("2 fixed model, data moves", j2, k2, k)

        # arm 3 -- refitted per month, consecutive months, as the reference does.
        j3 = Float64[]; k3 = Float64[]
        for s in 1:nseeds, i in 1:(length(months) - 1)
            jv = jaccard(tk_refit[s][i], tk_refit[s][i + 1])
            kv = kendall_dist(tk_refit[s][i], tk_refit[s][i + 1],
                              imp_refit[s][i], imp_refit[s][i + 1])
            push!(j3, jv); push!(k3, kv)
            s == 1 && push!(series, (k=k, arm="3-refit", from=label(i), to=label(i + 1),
                                     jaccard=jv, kendall=kv))
        end
        report("3 refit per month", j3, k3, k)

        # arm 5 -- fixed model, whole test split and not 100 rows.
        j5 = Float64[]; k5 = Float64[]
        for i in 1:(length(months) - 1)
            push!(j5, jaccard(tk_full[i], tk_full[i + 1]))
            push!(k5, kendall_dist(tk_full[i], tk_full[i + 1], imp_full[i], imp_full[i + 1]))
        end
        report("5 fixed model, whole month", j5, k5, k)

        m3, _, _ = summarise(j3); m2, _, _ = summarise(j2)
        @printf("\n  arm3 - arm2 = %+.3f  <- contribution of monthly refitting, which the published\n",
                m3 - m2)
        println("                          number folds in silently\n")
    end

    open(joinpath(@__DIR__, "series.csv"), "w") do io
        println(io, "k,arm,from,to,jaccard,kendall")
        for r in series
            @printf(io, "%d,%s,%s,%s,%.6f,%.6f\n", r.k, r.arm, r.from, r.to, r.jaccard, r.kendall)
        end
    end
    @printf("wrote series.csv (%d rows) -- the per-month curve behind the arm means above\n\n",
            length(series))

    println("-"^78)
    println("Reference for comparison, from research/b0-noise-floor/ on their MLP + KernelExplainer")
    println("at nsamples=100: reported consecutive-month Jaccard 0.958, against a same-month noise")
    println("floor of 0.926. Exact attribution has no coalition sampling, so any churn above arm 1")
    println("here is real. Arm 4 (sampled vs exact on the identical model) is in arm4.py.")
end

main()
