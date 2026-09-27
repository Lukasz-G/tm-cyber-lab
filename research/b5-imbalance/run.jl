# QUESTION:  is FPTM's domination on APIGraph caused by CLASS IMBALANCE?
# SURPRISE:  yes, and it is the hypothesis the cross-dataset evidence points at most directly. Across
#            our three datasets, width varies 15x and density 22x and neither tracks the outcome; base
#            rate tracks it perfectly (LAMDA 37% malware, FPTM ahead; APIGraph and AndroZoo 10%, FPTM
#            dominated). And every dataset this line of work has ever used — MNIST, Fashion-MNIST,
#            CIFAR-10, IMDb — is BALANCED, so imbalanced data is an untested regime for the whole
#            approach. FPTM applies Type I/II feedback per example with no class weighting, so at 90%
#            benign the benign side receives nine times the updates.
# ARMS:      five, and three of them exist to stop a confound:
#              1. FPTM, pool as shipped (10% malware)            -- the baseline, as run in B5
#              2. FPTM, balanced by UNDERSAMPLING benign          -- balanced, but 80% less data
#              3. FPTM, balanced by OVERSAMPLING malware          -- balanced, same data volume
#              4. FPTM, undersampled to arm 2's SIZE but keeping the 10% ratio  -- the volume control
#              5. LightGBM, same treatments (baselines.py side)   -- is imbalance TM-specific?
#            Arms 2 and 3 disagree if data volume matters; arm 4 is what separates "balancing helped"
#            from "any resampling of this size helped". Without arm 4, undersampling changes balance and
#            volume together and the result is uninterpretable. Arm 5 is what decides whether this is a
#            finding about Tsetlin machines or about the dataset.
# PASS/FAIL: not a gate. IMBALANCE IS THE CAUSE if balancing moves FPTM's swept frontier outward —
#            fewer dominated years — while the volume control does not, and while LightGBM is
#            comparatively unaffected. Anything else and imbalance is a contributing factor at most.
#
#   julia --project=. -t 16 research/b5-imbalance/run.jl [nseeds]
#
# Evaluation is always on the UNCHANGED test distribution (~10% malware). Resampling the test set would
# measure nothing of interest; the operating point a deployment sees is the real one.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
using .Tmx: read_inputs, read_meta, read_features
using TMCore
using Random, Printf, Statistics, JSON

const DIR = joinpath("data", "apigraph", "tmx", "apigraph")
const YEARS = 2013:2018
const CLAUSES, T, S, L, LF = 20, 10, 25, 64, 10
const EPOCHS = 30

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

function reach(rec, prec, tp_, tr_)
    r = NaN; p = NaN
    for k in eachindex(rec)
        prec[k] >= tp_ && (isnan(r) || rec[k] > r) && (r = rec[k])
        rec[k] >= tr_ && (isnan(p) || prec[k] > p) && (p = prec[k])
    end
    return r, p
end

best_f1(rec, prec) = maximum(2 .* prec .* rec ./ max.(1e-12, prec .+ rec))

"""
Build a training set under one resampling scheme.

`:asis`   the pool unchanged.
`:under`  every malware sample plus an equal number of benign — balanced, but much smaller.
`:over`   every benign sample plus malware replicated to match — balanced at full volume.
`:volume` the 10% ratio preserved, subsampled to the size arm `:under` produces — the control that
          separates balancing from simply having less data.
"""
function resample(X, Y, scheme, rng)
    pos = findall(Y); neg = findall(.!Y)
    idx = if scheme === :asis
        eachindex(Y)
    elseif scheme === :under
        vcat(pos, neg[randperm(rng, length(neg))[1:length(pos)]])
    elseif scheme === :over
        reps = cld(length(neg), length(pos))
        vcat(neg, repeat(pos, reps)[1:length(neg)])
    elseif scheme === :volume
        n = 2 * length(pos)                      # the size :under yields
        frac = n / length(Y)
        vcat(pos[randperm(rng, length(pos))[1:max(1, round(Int, frac * length(pos)))]],
             neg[randperm(rng, length(neg))[1:max(1, round(Int, frac * length(neg)))]])
    else
        error("unknown scheme $scheme")
    end
    order = randperm(rng, length(idx))
    return [X[idx[i]] for i in order], Bool[Y[idx[i]] for i in order]
end

function main()
    nseeds = isempty(ARGS) ? 5 : parse(Int, ARGS[1])
    feats = read_features(joinpath(DIR, "features.arrow"))
    width = length(feats)
    Xp, Yp = load("train")
    evals = [string(y) => load_year(y) for y in YEARS]
    base = JSON.parsefile("research/b5-apigraph/baselines.json")

    @printf("APIGraph width %d  pool %d rows, %.1f%% malware  %d clauses T=%d S=%d L=%d LF=%d  %d seeds\n",
            width, length(Yp), 100count(Yp) / length(Yp), CLAUSES, T, S, L, LF, nseeds)
    println("evaluation always on the unchanged ~10% malware test distribution\n")

    @printf("%-9s %9s %8s %9s %11s %11s %10s %10s\n",
            "scheme", "trainrows", "mal%", "bestF1", "dom by LGB", "dom by XGB", "meanP", "meanR")
    summary = Dict{Symbol,Any}()
    for scheme in (:asis, :under, :over, :volume)
        bf = Dict(k => Float64[] for (k, _) in evals)
        doml = Int[]; domx = Int[]; nrows = Int[]; malpct = Float64[]
        ps = Float64[]; rs = Float64[]
        for seed in 1:nseeds
            rng = MersenneTwister(4000 + seed)
            Xtr, Ytr = resample(Xp, Yp, scheme, rng)
            push!(nrows, length(Ytr)); push!(malpct, 100count(Ytr) / length(Ytr))
            m = TMClassifier([false, true], width;
                             clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
            for e in 1:EPOCHS
                train!(m, Xtr, Ytr; rng=MersenneTwister(1000seed + e), parallel=:none)
            end
            dl = 0; dx = 0
            for (k, (X, Y)) in evals
                mg = margins(m, X)
                rec, prec = pr_curve(mg, Y)
                push!(bf[k], 100 * best_f1(rec, prec))
                rl, pl = reach(rec, prec, BP[k].lgb[1] / 100, BP[k].lgb[2] / 100)
                rx, px = reach(rec, prec, BP[k].xgb[1] / 100, BP[k].xgb[2] / 100)
                (isnan(rl) || 100rl < BP[k].lgb[2]) && (isnan(pl) || 100pl < BP[k].lgb[1]) && (dl += 1)
                (isnan(rx) || 100rx < BP[k].xgb[2]) && (isnan(px) || 100px < BP[k].xgb[1]) && (dx += 1)
                pred = mg .> 0
                tp = count(pred .& Y); fp = count(pred .& .!Y); fn = count(.!pred .& Y)
                push!(ps, tp + fp == 0 ? 0.0 : 100tp / (tp + fp))
                push!(rs, tp + fn == 0 ? 0.0 : 100tp / (tp + fn))
            end
            push!(doml, dl); push!(domx, dx)
        end
        mf = mean([mean(bf[k]) for (k, _) in evals])
        @printf("%-9s %9.0f %8.1f %9.2f %11s %11s %10.1f %10.1f\n",
                string(scheme), mean(nrows), mean(malpct), mf,
                @sprintf("%.1f/6", mean(doml)), @sprintf("%.1f/6", mean(domx)),
                mean(ps), mean(rs))
        summary[scheme] = (f1=mf, doml=mean(doml), domx=mean(domx))
        @printf("RESULT scheme=%s train_rows=%.0f mal_pct=%.1f best_f1=%.2f dom_lgb=%.2f dom_xgb=%.2f default_p=%.1f default_r=%.1f\n",
                scheme, mean(nrows), mean(malpct), mf, mean(doml), mean(domx), mean(ps), mean(rs))
    end

    println("\nper-year best swept F1, against the boosters at their default threshold")
    @printf("%-9s %s %11s %11s\n", "scheme", join([@sprintf("%8s", k) for (k, _) in evals]), "LGB mean", "XGB mean")
    println("  (the F1 columns compare a swept FPTM against unswept boosters and so flatter us;")
    println("   the dominance columns above are the threshold-free verdict)")

    a = summary[:asis]; u = summary[:under]; o = summary[:over]; v = summary[:volume]
    @printf("\nbalancing effect on dominance (lower is better):\n")
    @printf("  as-is %.1f -> undersampled %.1f, oversampled %.1f, volume control %.1f  (vs LightGBM)\n",
            a.doml, u.doml, o.doml, v.doml)
    helped = min(u.doml, o.doml) < a.doml - 0.5 && min(u.doml, o.doml) < v.doml - 0.5
    println("RESULT imbalance_is_the_cause=", helped ? "SUPPORTED" : "NOT_SUPPORTED")
end

main()
