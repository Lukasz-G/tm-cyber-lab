# QUESTION:  does forcing shorter, more specific clauses move FPTM's precision/recall frontier OUTWARD
#            on APIGraph, or only along it?
# SURPRISE:  yes. The threshold sweep established that FPTM is dominated by both boosters at their own
#            operating points — so the limitation is representational in place of a threshold artefact
#            — and that clause COUNT only slides along the frontier. The remaining structural suspect
#            is clause LENGTH: a booster composes many shallow specific conjunctions, while a clause
#            here is a single ~69-literal conjunction with tolerance 10. `L` is what sets that length,
#            and it was carried over from the LAMDA configuration and never questioned.
#            The sibling project found that CAPPING `L` cost 13 points on image tasks, so the prior is
#            that lowering it hurts. If it helps here, that is a regime difference worth knowing; if it
#            hurts, clause length is exonerated and the limitation lies deeper.
# ARMS:      L in {8, 16, 32, 64} at fixed clause count, T, S and LF. Each arm is judged on its whole
#            swept frontier, not on its default operating point, because the sweep showed the default
#            argmax costs ~8.6 F1 — comparing default points across L would confound clause length with
#            how well the default threshold happens to suit each one.
# PASS/FAIL: not a gate. The frontier moves outward if an arm reaches a booster's recall at its
#            precision (or vice versa) on more years than L=64 does. Dominance counts, not F1.
#
#   julia --project=. -t 16 research/b5-clause-length/run.jl [nseeds]
#
# Caveat carried forward: the boosters are evaluated at THEIR default threshold, which is also not
# their optimum. Comparing a swept FPTM against an unswept booster on F1 would be unfair in our
# favour, which is why the verdict here is the threshold-free dominance test.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: clause_literals
using TMCore
using Random, Printf, Statistics, JSON

const DIR = joinpath("data", "apigraph", "tmx", "apigraph")
const YEARS = 2013:2018
const S, LF = 25, 10
const EPOCHS = 30
# (L, clauses) grid. The (8, 200) cell is the booster-shaped regime -- many shallow conjunctions --
# and was the only one of the four corners left untested by the first pass.
# (L, clauses, T). T is scaled per the published relation T ~ sqrt(CLAUSES/2 * LF): 10 at 20 clauses,
# 32 at 200. The earlier pass held T=10 everywhere, which confounded the 200-clause cells with an
# under-scaled threshold; both T values are now run at 200 so the confound is separated rather than
# assumed away.
const GRID = ((8, 20, 10), (16, 20, 10), (32, 20, 10), (64, 20, 10),
              (8, 200, 10), (16, 200, 10), (64, 200, 10),
              (8, 200, 32), (16, 200, 32), (64, 200, 32))

const BP = Dict("2013" => (lgb=(97.28, 74.41), xgb=(95.37, 73.92)),
                "2014" => (lgb=(96.18, 49.44), xgb=(90.15, 57.05)),
                "2015" => (lgb=(96.09, 45.50), xgb=(88.27, 52.87)),
                "2016" => (lgb=(94.13, 46.09), xgb=(82.79, 58.26)),
                "2017" => (lgb=(91.27, 53.87), xgb=(80.16, 65.56)),
                "2018" => (lgb=(88.86, 45.97), xgb=(84.65, 63.10)))

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

margins(m, X) = Float64[score(m, 2, x) - score(m, 1, x) for x in X]

function pr_curve(marg, y)
    ord = sortperm(marg; rev=true)
    npos = count(y); tp = 0; fp = 0
    rec = Float64[]; prec = Float64[]
    i = 1
    while i <= length(ord)
        t = marg[ord[i]]
        while i <= length(ord) && marg[ord[i]] == t
            y[ord[i]] ? (tp += 1) : (fp += 1); i += 1
        end
        push!(rec, tp / npos); push!(prec, tp / (tp + fp))
    end
    return rec, prec
end

function reach(rec, prec, target_p, target_r)
    r_at_p = NaN; p_at_r = NaN
    for k in eachindex(rec)
        prec[k] >= target_p && (isnan(r_at_p) || rec[k] > r_at_p) && (r_at_p = rec[k])
        rec[k] >= target_r && (isnan(p_at_r) || prec[k] > p_at_r) && (p_at_r = prec[k])
    end
    return r_at_p, p_at_r
end

best_f1(rec, prec) = maximum(2 .* prec .* rec ./ max.(1e-12, prec .+ rec))

function main()
    nseeds = isempty(ARGS) ? 5 : parse(Int, ARGS[1])
    feats = read_features(joinpath(DIR, "features.arrow"))
    width = length(feats)
    Xtr, Ytr = load("train")
    evals = [string(y) => load_year(y) for y in YEARS]

    @printf("APIGraph width %d  S=%d LF=%d  %d seeds  (L, clauses, T) grid
", width, S, LF, nseeds)
    println("judged on the swept frontier, not the default threshold\n")

    @printf("%-4s %8s %4s %9s %9s %11s %11s %13s
",
            "L", "clauses", "T", "literals", "bestF1", "domLGB", "domXGB", "meanF1 vs XGB")
    for (L, CLAUSES, T) in GRID
        bf = Dict(k => Float64[] for (k, _) in evals)
        doml = 0; domx = 0; lits = Float64[]
        for seed in 1:nseeds
            m = TMClassifier([false, true], width;
                             clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
            for e in 1:EPOCHS
                train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=:none)
            end
            c = Int[]
            for bank in (m.positive[1], m.negative[1]), j in 1:bank.nclauses
                n = length(clause_literals(bank, j)); n > 0 && push!(c, n)
            end
            push!(lits, isempty(c) ? 0.0 : sort(c)[cld(length(c), 2)])
            if seed == 1
                for (k, (X, Y)) in evals
                    rec, prec = pr_curve(margins(m, X), Y)
                    rl, pl = reach(rec, prec, BP[k].lgb[1] / 100, BP[k].lgb[2] / 100)
                    rx, px = reach(rec, prec, BP[k].xgb[1] / 100, BP[k].xgb[2] / 100)
                    (isnan(rl) || 100rl < BP[k].lgb[2]) && (isnan(pl) || 100pl < BP[k].lgb[1]) && (doml += 1)
                    (isnan(rx) || 100rx < BP[k].xgb[2]) && (isnan(px) || 100px < BP[k].xgb[1]) && (domx += 1)
                end
            end
            for (k, (X, Y)) in evals
                rec, prec = pr_curve(margins(m, X), Y)
                push!(bf[k], 100 * best_f1(rec, prec))
            end
        end
        base = JSON.parsefile("research/b5-apigraph/baselines.json")
        gl = mean([mean(bf[k]) - base["lightgbm"][k] for (k, _) in evals])
        gx = mean([mean(bf[k]) - base["xgboost"][k] for (k, _) in evals])
        @printf("%-4d %8d %4d %9.0f %9.2f %11s %11s %+13.2f
",
                L, CLAUSES, T, median(lits), mean([mean(bf[k]) for (k, _) in evals]),
                "$doml/6", "$domx/6", gx)
        @printf("RESULT L=%d clauses=%d T=%d median_literals=%.0f mean_best_f1=%.2f dom_lgb=%d dom_xgb=%d
",
                L, CLAUSES, T, median(lits), mean([mean(bf[k]) for (k, _) in evals]), doml, domx)
    end
    println("\ndominance is the verdict; the F1 columns compare a SWEPT FPTM against UNSWEPT boosters")
    println("and therefore flatter us — they are shown for shape, not as a claim.")
end

main()
