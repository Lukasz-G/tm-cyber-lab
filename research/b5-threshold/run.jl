# QUESTION:  is flat FPTM genuinely dominated by gradient boosting on APIGraph, or is it merely sitting
#            at a bad default operating point?
# SURPRISE:  yes. Every comparison in this project so far has used ONE operating point per model, which
#            cannot distinguish "worse model" from "same model, wrong threshold". A Tsetlin machine's
#            class score is an integer margin, so it has a threshold to sweep like any scorer, and
#            nobody has swept it. If the boosters' points lie INSIDE the swept FPTM curve, the model is
#            dominated and the limitation is representational. If they lie ON or OUTSIDE it, six
#            experiments' worth of single-point comparisons have been measuring a default rather than a
#            capability, and the conclusions need revisiting.
# ARMS:      the full FPTM precision/recall curve against the two boosters' single points. Sweeping the
#            margin is itself the control: it removes threshold choice as an explanation, which is the
#            one confound a single-point comparison cannot rule out.
# PASS/FAIL: not a gate. Two numbers decide the reading, per year:
#              recall at the booster's precision, and precision at the booster's recall.
#            If both are below the booster on most years, FPTM is dominated. If either exceeds it, the
#            default threshold was costing us and the earlier numbers understate the model.
#
#   julia --project=. -t 16 research/b5-threshold/run.jl [nseeds]
#
# Note this does NOT revisit the B5 verdict, which compared default-threshold models on a
# pre-registered criterion. It asks a different question: where the frontier actually is.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
using .Tmx: read_inputs, read_meta, read_features
using TMCore
using Random, Printf, Statistics, JSON

const DIR = joinpath("data", "apigraph", "tmx", "apigraph")
const YEARS = 2013:2018
const CLAUSES, T, S, L, LF = 20, 10, 25, 64, 10
const EPOCHS = 30

function load(stem)
    inputs, _ = read_inputs(joinpath(DIR, "$stem.tmx"))
    meta = read_meta(joinpath(DIR, "$stem.meta.arrow"))
    return inputs, collect(Bool.(meta.label))
end

function load_year(year)
    X = TMInput[]; Y = Bool[]
    for m in 1:12
        stem = @sprintf("%d-%02d", year, m)
        isfile(joinpath(DIR, "$stem.tmx")) || continue
        xs, ys = load(stem); append!(X, xs); append!(Y, ys)
    end
    return X, Y
end

"""
Margin for the positive class: score(malware) - score(benign).

`predict` takes the argmax, which is exactly the sign of this quantity, so a threshold of 0 reproduces
the default prediction. Sweeping it traces the model's own precision/recall curve without retraining.
"""
margins(m, X) = Float64[score(m, 2, x) - score(m, 1, x) for x in X]

"""
Precision and recall at every distinct threshold, walking the sorted margins from most to least
confident. Returns vectors of (recall, precision) plus the threshold, ordered by increasing recall.
"""
function pr_curve(marg::Vector{Float64}, y::Vector{Bool})
    ord = sortperm(marg; rev=true)
    npos = count(y)
    tp = 0; fp = 0
    rec = Float64[]; prec = Float64[]; thr = Float64[]
    i = 1
    while i <= length(ord)
        t = marg[ord[i]]
        while i <= length(ord) && marg[ord[i]] == t      # all ties share one threshold
            y[ord[i]] ? (tp += 1) : (fp += 1)
            i += 1
        end
        push!(rec, tp / npos); push!(prec, tp / (tp + fp)); push!(thr, t)
    end
    return rec, prec, thr
end

"Highest recall achievable at or above a target precision; NaN if unreachable."
function recall_at_precision(rec, prec, target)
    best = NaN
    for k in eachindex(rec)
        if prec[k] >= target && (isnan(best) || rec[k] > best)
            best = rec[k]
        end
    end
    return best
end

"Highest precision achievable at or above a target recall; NaN if unreachable."
function precision_at_recall(rec, prec, target)
    best = NaN
    for k in eachindex(rec)
        if rec[k] >= target && (isnan(best) || prec[k] > best)
            best = prec[k]
        end
    end
    return best
end

best_f1(rec, prec) = maximum(2 .* prec .* rec ./ max.(1e-12, prec .+ rec))

function main()
    nseeds = isempty(ARGS) ? 5 : parse(Int, ARGS[1])
    feats = read_features(joinpath(DIR, "features.arrow"))
    Xtr, Ytr = load("train")
    evals = [string(y) => load_year(y) for y in YEARS]
    base = JSON.parsefile("research/b5-apigraph/baselines.json")

    # Booster operating points, read off research/b5-apigraph/baselines.txt.
    bp = Dict("2013" => (lgb=(97.28, 74.41), xgb=(95.37, 73.92)),
              "2014" => (lgb=(96.18, 49.44), xgb=(90.15, 57.05)),
              "2015" => (lgb=(96.09, 45.50), xgb=(88.27, 52.87)),
              "2016" => (lgb=(94.13, 46.09), xgb=(82.79, 58.26)),
              "2017" => (lgb=(91.27, 53.87), xgb=(80.16, 65.56)),
              "2018" => (lgb=(88.86, 45.97), xgb=(84.65, 63.10)))

    @printf("APIGraph  width %d  %d clauses  T=%d S=%d L=%d LF=%d  %d seeds\n\n",
            length(feats), CLAUSES, T, S, L, LF, nseeds)

    acc = Dict(k => Dict(:default_f1 => Float64[], :best_f1 => Float64[],
                         :r_at_lgb_p => Float64[], :r_at_xgb_p => Float64[],
                         :p_at_lgb_r => Float64[], :p_at_xgb_r => Float64[]) for (k, _) in evals)
    for seed in 1:nseeds
        m = TMClassifier([false, true], length(feats);
                         clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
        for e in 1:EPOCHS
            train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=:none)
        end
        for (k, (X, Y)) in evals
            mg = margins(m, X)
            rec, prec, _ = pr_curve(mg, Y)
            # default prediction is margin > 0, matching predict's argmax with ties to the lower index
            pred = mg .> 0
            tp = count(pred .& Y); fp = count(pred .& .!Y); fn = count(.!pred .& Y)
            p0 = tp + fp == 0 ? 0.0 : tp / (tp + fp); r0 = tp + fn == 0 ? 0.0 : tp / (tp + fn)
            push!(acc[k][:default_f1], p0 + r0 == 0 ? 0.0 : 100 * 2p0 * r0 / (p0 + r0))
            push!(acc[k][:best_f1], 100 * best_f1(rec, prec))
            push!(acc[k][:r_at_lgb_p], 100 * recall_at_precision(rec, prec, bp[k].lgb[1] / 100))
            push!(acc[k][:r_at_xgb_p], 100 * recall_at_precision(rec, prec, bp[k].xgb[1] / 100))
            push!(acc[k][:p_at_lgb_r], 100 * precision_at_recall(rec, prec, bp[k].lgb[2] / 100))
            push!(acc[k][:p_at_xgb_r], 100 * precision_at_recall(rec, prec, bp[k].xgb[2] / 100))
        end
    end

    println("=== FPTM swept, against each booster's operating point ===")
    @printf("%-5s %8s %8s | %s\n", "year", "deflt", "bestF1", "recall at their precision / precision at their recall")
    @printf("%-5s %8s %8s | %18s %18s\n", "", "F1", "swept", "vs LightGBM", "vs XGBoost")
    dominated_lgb = 0; dominated_xgb = 0; n = 0
    for (k, _) in evals
        a = acc[k]
        rl, rx = mean(a[:r_at_lgb_p]), mean(a[:r_at_xgb_p])
        pl, px = mean(a[:p_at_lgb_r]), mean(a[:p_at_xgb_r])
        @printf("%-5s %8.2f %8.2f | R %5.1f vs %5.1f  P %5.1f vs %5.1f  R %5.1f vs %5.1f  P %5.1f vs %5.1f\n",
                k, mean(a[:default_f1]), mean(a[:best_f1]),
                rl, bp[k].lgb[2], pl, bp[k].lgb[1], rx, bp[k].xgb[2], px, bp[k].xgb[1])
        n += 1
        (isnan(rl) || rl < bp[k].lgb[2]) && (isnan(pl) || pl < bp[k].lgb[1]) && (dominated_lgb += 1)
        (isnan(rx) || rx < bp[k].xgb[2]) && (isnan(px) || px < bp[k].xgb[1]) && (dominated_xgb += 1)
    end

    @printf("\ndefault threshold costs: best swept F1 minus default F1 = %+.2f on average\n",
            mean([mean(acc[k][:best_f1]) - mean(acc[k][:default_f1]) for (k, _) in evals]))
    @printf("years where FPTM is strictly dominated: LightGBM %d/%d, XGBoost %d/%d\n",
            dominated_lgb, n, dominated_xgb, n)
    println("  (dominated = cannot reach their recall at their precision, NOR their precision at their recall)")
    for (k, _) in evals
        @printf("RESULT %s default_f1=%.2f best_f1=%.2f r_at_lgb_p=%.1f r_at_xgb_p=%.1f p_at_lgb_r=%.1f p_at_xgb_r=%.1f\n",
                k, mean(acc[k][:default_f1]), mean(acc[k][:best_f1]),
                mean(acc[k][:r_at_lgb_p]), mean(acc[k][:r_at_xgb_p]),
                mean(acc[k][:p_at_lgb_r]), mean(acc[k][:p_at_xgb_r]))
    end
    @printf("RESULT dominated_by_lgb=%d/%d dominated_by_xgb=%d/%d\n", dominated_lgb, n, dominated_xgb, n)
end

main()
