# QUESTION:  what does an FPTM actually cost to deploy, and does updating it continually retain
#            detection for FEWER LABELS than retraining does?
# SURPRISE:  yes, on both halves, and the first one is uncomfortable for this project's own framing.
#            The advertised FPTM footprint is ~50 KB, but that figure is the INCLUDE MASKS -- what
#            inference needs. A model that keeps learning must also carry its automaton states, one
#            counter per literal per clause, and those are the overwhelming majority of the bytes. So
#            the small-footprint claim and the online-learning claim are in tension: you get the 50 KB
#            only by giving up the ability to update, and this project has cited both.
#            The second half is the only defensible form of the online-learning claim. "Online learning
#            solves drift" is wrong -- applying a label is nearly free but PRODUCING one is not, and
#            the booster retrains in minutes. So the question is not speed, it is labels: per label
#            spent, does continuing to train an existing model beat starting over?
# ARMS:      four update strategies at each label budget, and the first two are the controls without
#            which the other two mean nothing:
#              1. FROZEN -- the 2013-14 model, no update at all. Establishes what the labels buy.
#              2. RETRAIN-RECENT -- a fresh model on the budget alone. The naive baseline, and the one
#                 that shows whether the old data is worth keeping.
#              3. CONTINUAL -- the 2013-14 model, training continued on the budget. The claim.
#              4. RETRAIN-ALL -- a fresh model on 2013-14 plus the budget. The honest upper bound: what
#                 you get if labels are cheap and compute is not. Without arm 4, arm 3 beating arm 2
#                 would look like a win when it might just be "more data helps".
#            Budgets 250 / 1000 / 4000 / 16000 labelled rows drawn from the test year itself.
# PASS/FAIL: not a gate. The label-efficiency claim SURVIVES if arm 3 reaches a given F1 at a smaller
#            budget than arms 2 and 4 do. It DIES if arm 4 dominates at every budget, because then the
#            right advice is "retrain on everything" and continual updating buys only wall clock, which
#            §7 of this project's design notes already rejects as a contribution.
#
#   julia --project=. -t 16 research/b4-footprint-labels/run.jl [nseeds]
#
# Labels are drawn at random from the year's training split with a per-seed RNG, not taken as the first
# N rows, because LAMDA's per-year files are ordered by class and a prefix would be all one label.
# Evaluation is always the year's held-out test split, which no arm trains on.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: clause_literals
using TMCore
using Random, Printf, Statistics

const TMX_DIR = joinpath("data", "lamda", "tmx")
const TRAIN_YEARS = (2013, 2014)
const TEST_YEARS = (2018, 2019, 2020, 2021, 2022)
const CLAUSES, T, S, L, LF = 20, 10, 100, 64, 10
const EPOCHS = 30
const BUDGETS = (250, 1000, 4000, 16000)
const TM_LAB_COMMIT = "43dba5f"

portion(year, which) = begin
    meta = read_meta(joinpath(TMX_DIR, "$year.meta.arrow"))
    idx = [i for i in eachindex(meta.file_split) if meta.file_split[i] == String(which)]
    xs, _ = read_inputs(joinpath(TMX_DIR, "$year.tmx"); rows=idx)
    (xs, Bool[Bool(meta.label[i]) for i in idx])
end

function load_pool(years)
    X = TMInput[]; Y = Bool[]
    for y in years
        xs, ys = portion(y, :train)
        append!(X, xs); append!(Y, ys)
    end
    return X, Y
end

function f1(m, X, Y)
    tp = 0; fp = 0; fn = 0
    for (x, y) in zip(X, Y)
        p = predict(m, x)
        if p && y
            tp += 1
        elseif p && !y
            fp += 1
        elseif !p && y
            fn += 1
        end
    end
    prec = tp + fp == 0 ? 0.0 : tp / (tp + fp)
    rec = tp + fn == 0 ? 0.0 : tp / (tp + fn)
    return prec + rec == 0 ? 0.0 : 200 * prec * rec / (prec + rec)
end

fresh(width) = TMClassifier([false, true], width;
                            clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)

function fit!(m, X, Y, seed; epochs=EPOCHS)
    for e in 1:epochs
        train!(m, X, Y; rng=MersenneTwister(1000seed + e), parallel=:none)
    end
    return m
end

# ---------------------------------------------------------------------------------------------
# Part A -- footprint, counted, not quoted

"""
Bytes an FPTM occupies, split by what the bytes are FOR.

  include masks   two bits per feature per clause (the literal and its negation), which is all that
                  inference reads -- this is the number the ~50 KB claim refers to.
  automaton state one counter per literal per clause. Required to CONTINUE training and carried by
                  any model that updates online. TMCore's states_num defaults to 256, so one byte.
"""
function footprint(m, width)
    nclauses = 0
    for ci in 1:2, bank in (m.positive[ci], m.negative[ci])
        nclauses += bank.nclauses
    end
    nliterals = 2 * width                       # feature present / feature absent
    masks = nclauses * nliterals / 8            # bits -> bytes
    states = nclauses * nliterals * 1           # one byte per automaton
    return (nclauses=nclauses, masks=masks, states=states)
end

function report_footprint(m, width)
    fp = footprint(m, width)
    println("PART A -- footprint, counted from the model and not quoted")
    @printf("  clauses total (both classes, both polarities)  %d\n", fp.nclauses)
    @printf("  literals per clause (2 x width)                %d\n", 2width)
    @printf("  include masks  -- what INFERENCE needs          %8.1f KB\n", fp.masks / 1024)
    @printf("  automaton state -- what UPDATING needs          %8.1f KB\n", fp.states / 1024)
    @printf("  ratio                                           %8.1fx\n", fp.states / fp.masks)
    @printf("RESULT footprint clauses=%d masks_kb=%.2f states_kb=%.2f ratio=%.1f\n",
            fp.nclauses, fp.masks / 1024, fp.states / 1024, fp.states / fp.masks)

    lgb = joinpath("data", "lamda", "lightgbm_2013_2014.pkl")
    if isfile(lgb)
        kb = filesize(lgb) / 1024
        @printf("  LightGBM on the same task, as pickled           %8.1f KB  (%.0fx the masks)\n",
                kb, kb / (fp.masks / 1024))
        @printf("RESULT footprint lightgbm_kb=%.1f\n", kb)
    end
    println("""
  The ~50 KB figure this project has cited is the include-mask row. A model that keeps
  learning carries the automaton row as well, so the small-footprint claim and the
  online-updating claim cannot both be made about the same artefact.
""")
end

# ---------------------------------------------------------------------------------------------

function main()
    nseeds = isempty(ARGS) ? 2 : parse(Int, ARGS[1])
    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    width = length(feats)
    Xbase, Ybase = load_pool(TRAIN_YEARS)

    @printf("LAMDA width %d  base train %d rows  %d seeds\n", width, length(Ybase), nseeds)
    @printf("clauses=%d T=%d S=%d L=%d LF=%d epochs=%d  ceiling=LiteralCapped  parallel=:none  tm-lab %s\n\n",
            CLAUSES, T, S, L, LF, EPOCHS, TM_LAB_COMMIT)
    flush(stdout)

    @printf("training the base 2013-14 model ... ")
    flush(stdout)
    t0 = time()
    base = fit!(fresh(width), Xbase, Ybase, 1)
    @printf("%.1fs\n\n", time() - t0)
    report_footprint(base, width)
    flush(stdout)

    println("PART B -- labels consumed per unit of retained detection")
    println("F1 on each year's held-out test split. 'frozen' spends no labels at all.\n")
    @printf("%-6s %8s %8s %10s %10s %10s\n",
            "year", "budget", "frozen", "retrain-r", "continual", "retrain-all")
    println("-"^58)

    for year in TEST_YEARS
        Xtr, Ytr = portion(year, :train)
        Xte, Yte = portion(year, :test)
        frozen_f1 = f1(base, Xte, Yte)

        for budget in BUDGETS
            budget <= length(Ytr) || continue
            rr = Float64[]; cc = Float64[]; ra = Float64[]
            for seed in 1:nseeds
                rng = MersenneTwister(77seed + year)
                sel = randperm(rng, length(Ytr))[1:budget]
                Xb, Yb = Xtr[sel], Ytr[sel]

                push!(rr, f1(fit!(fresh(width), Xb, Yb, seed), Xte, Yte))

                cont = deepcopy(base)                     # the claim: keep learning from where it is
                push!(cc, f1(fit!(cont, Xb, Yb, seed), Xte, Yte))

                allX = vcat(Xbase, Xb); allY = vcat(Ybase, Yb)
                push!(ra, f1(fit!(fresh(width), allX, allY, seed), Xte, Yte))
            end
            @printf("%-6d %8d %8.2f %10.2f %10.2f %10.2f\n",
                    year, budget, frozen_f1, mean(rr), mean(cc), mean(ra))
            @printf("RESULT year=%d budget=%d frozen=%.2f retrain_recent=%.2f continual=%.2f retrain_all=%.2f\n",
                    year, budget, frozen_f1, mean(rr), mean(cc), mean(ra))
            flush(stdout)
        end
        println()
    end

    println("-"^58)
    println("Read along a row: what each strategy gets for the SAME number of labels. The")
    println("label-efficiency claim needs 'continual' to reach a given F1 at a smaller budget than")
    println("both 'retrain-recent' and 'retrain-all'. If 'retrain-all' leads everywhere, the honest")
    println("advice is to retrain on everything and continual updating buys only wall clock, which is")
    println("not a contribution this project is willing to claim.")
end

main()
