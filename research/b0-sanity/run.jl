# QUESTION:  at the B1 gate configuration, what are this model's LF/included-literals ratio and its
#            sign composition? And does it train at all on real LAMDA data?
# SURPRISE:  yes, on both diagnostics — they are pre-registered predictions that can fail.
#            Prediction 2 says LAMDA lands near IMDb's 1.7% tolerance rather than MNIST's 21%, so the
#            clauses are near-strict conjunctions of thousands of literals and NOT human-readable.
#            Prediction 1 says the clauses are near-pure blacklists — literals requiring features to
#            be ABSENT — which would mean they give an analyst no indicators of compromise.
#            The detection number here is NOT the gate and must not be quoted as one.
# ARMS:      one model, two diagnostics. No comparison is being made, so no control is needed; the
#            dataset control for any later readability claim is a separate experiment.
# PASS/FAIL: PASS if training completes and both diagnostics are measured. The diagnostics have no
#            pass criterion — they are measurements whose values decide what may be claimed later.
#
#   julia --project=. research/b0-sanity/run.jl
#
# Configuration: 20 clauses per class across both polarities, T = 10 (= sqrt(CLAUSES/2 * LF)),
# S = 100, L = 64, LF = 10, LiteralCapped ceiling, parallel = :none. Serial because :classes caps at
# the class count (two) and :clauses is slower than serial at this clause count on this width.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: clause_literals
using TMCore
using Random, Printf, Statistics

const TMX_DIR = joinpath("data", "lamda", "tmx")
const TRAIN_YEARS = (2013, 2014)
const CLAUSES, T, S, L, LF = 20, 10, 100, 64, 10
const EPOCHS = 10
const PARALLEL = :none
const TM_LAB_COMMIT = "43dba5f"

"Rows of one year belonging to the released train or test portion."
function portion(year, which)
    meta = read_meta(joinpath(TMX_DIR, "$year.meta.arrow"))
    idx = [i for i in eachindex(meta.file_split) if meta.file_split[i] == which]
    inputs, _ = read_inputs(joinpath(TMX_DIR, "$year.tmx"); rows=idx)
    return inputs, Bool[meta.label[i] for i in idx]
end

function load(which)
    X = TMInput[]; Y = Bool[]
    for y in TRAIN_YEARS
        xs, ys = portion(y, which)
        append!(X, xs); append!(Y, ys)
    end
    return X, Y
end

function prf(pred, y)
    tp = count(pred .& y); fp = count(pred .& .!y)
    fn = count(.!pred .& y); tn = count(.!pred .& .!y)
    prec = tp + fp == 0 ? 0.0 : tp / (tp + fp)
    rec = tp + fn == 0 ? 0.0 : tp / (tp + fn)
    f1 = prec + rec == 0 ? 0.0 : 2prec * rec / (prec + rec)
    return (f1=100 * f1, precision=100 * prec, recall=100 * rec,
            fnr=100 * (1 - rec), fpr=100 * (fp / max(1, fp + tn)))
end

"""
The two diagnostics.

`LF`/included-literals is the readability ratio: on MNIST at 21% a fuzzy clause decomposes into a
strict core plus a tolerance tail that a human can read; at IMDb's 1.7% the core is thousands of
literals and extraction hands back something precise and useless. Sign composition is the other half
— a clause built only from negated literals says "none of these tokens present", which is a
blacklist and carries no indicator of compromise.
"""
function diagnostics(m)
    counts = Int[]
    pos_lits = 0
    neg_lits = 0
    per_clause_pos = Float64[]
    for bank in (m.positive[1], m.negative[1])
        for j in 1:bank.nclauses
            lits = clause_literals(bank, j)
            isempty(lits) && continue
            push!(counts, length(lits))
            p = count(l -> !l[2], lits)
            pos_lits += p
            neg_lits += length(lits) - p
            push!(per_clause_pos, p / length(lits))
        end
    end
    return counts, pos_lits, neg_lits, per_clause_pos
end

function main()
    @printf("tm-lab pin %s   threads %d   parallel %s   ceiling LiteralCapped\n",
            TM_LAB_COMMIT, Threads.nthreads(), PARALLEL)
    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    Xtr, Ytr = load("train")
    Xte, Yte = load("test")
    @printf("train %d (malware %d)   IID test %d (malware %d)   width %d\n",
            length(Ytr), count(Ytr), length(Yte), count(Yte), length(Xtr[1]))

    m = TMClassifier([false, true], length(feats);
                     clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    @printf("\nCLAUSES %d (per class, both polarities)  T %d  S %d  L %d  LF %d  epochs %d\n",
            CLAUSES, T, S, L, LF, EPOCHS)

    for e in 1:EPOCHS
        t = @elapsed train!(m, Xtr, Ytr; rng=MersenneTwister(e), parallel=PARALLEL)
        s = prf(predict(m, Xte), Yte)
        @printf("  epoch %2d  %5.1fs   IID F1 %6.2f  precision %6.2f  recall %6.2f  FNR %5.2f  FPR %5.2f\n",
                e, t, s.f1, s.precision, s.recall, s.fnr, s.fpr)
    end

    final = prf(predict(m, Xte), Yte)
    counts, pos_lits, neg_lits, per_clause_pos = diagnostics(m)
    tot = pos_lits + neg_lits
    med = isempty(counts) ? 0 : sort(counts)[cld(length(counts), 2)]

    println("\n=== diagnostic 1: readability ratio (prediction 2) ===")
    @printf("included literals per clause: min %d  median %d  max %d  (L = %d, so clauses run %.1fx over it)\n",
            minimum(counts), med, maximum(counts), L, maximum(counts) / L)
    @printf("LF / median included literals = %d / %d = %.2f%%\n", LF, med, 100 * LF / med)
    println("  reference points: MNIST 21% decomposes into readable rules; published IMDb 1.7% does not")

    println("\n=== diagnostic 2: sign composition (prediction 1) ===")
    @printf("literals: %d positive (feature present), %d negated (feature absent) — %.1f%% negated\n",
            pos_lits, neg_lits, 100 * neg_lits / tot)
    @printf("per-clause positive fraction: min %.3f  median %.3f  max %.3f\n",
            minimum(per_clause_pos), sort(per_clause_pos)[cld(length(per_clause_pos), 2)],
            maximum(per_clause_pos))
    @printf("clauses with NO positive literal (pure blacklist): %d of %d\n",
            count(iszero, per_clause_pos), length(per_clause_pos))

    @printf("\nRESULT iid_f1=%.2f fnr=%.2f fpr=%.2f median_literals=%d lf_ratio=%.4f neg_frac=%.4f pure_blacklist=%d/%d\n",
            final.f1, final.fnr, final.fpr, med, LF / med, neg_lits / tot,
            count(iszero, per_clause_pos), length(per_clause_pos))
    println("RESULT b0_sanity=PASS")
end

main()
