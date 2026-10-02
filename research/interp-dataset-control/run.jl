# QUESTION:  can the feature list the model's attribution picks out be reproduced WITHOUT the model,
#            from a chi-squared or frequency ranking of the raw training data?
# SURPRISE:  yes, whichever way it lands, because the equivalent control in the sibling algorithm
#            project RETRACTED its one interpretability positive: the readable words extracted from an
#            IMDb model turned out to come from the dataset, not from the model, and a plain
#            frequency ranking reproduced them. Nobody had checked. Here the prior is genuinely open --
#            this model's tolerance came out at 14.5% of the clause, the regime that decomposed into
#            genuine rules on MNIST, not IMDb's useless 1.7% -- so the control can pay off rather than
#            retract. If chi-squared reproduces the list, every readability claim in this project dies
#            and the contribution is the EXACTNESS of the attribution, which survives unreadable
#            clauses. If it does not, a readability claim becomes defensible for the first time.
# ARMS:      four rankings of the same 4,561 features on the same training rows, so the only thing
#            varying is what produced the ranking:
#              1. the MODEL, via exact Shapley attribution -- the thing under test.
#              2. chi-squared of feature against label -- the standard univariate selector, and the
#                 one that retracted the sibling result.
#              3. document-frequency difference, P(f|malware) - P(f|benign) -- the crudest possible
#                 dataset ranking, included because if even THAT reproduces the list the claim is dead
#                 in the simplest way.
#              4. RANDOM features at matched k -- the arm without which the others cannot be read.
#                 Top-100 of 4,561 overlaps a fixed 100-set by ~2.2 features by chance, and "the
#                 overlap is low" means nothing until that number is on the page.
#            A fifth ranking is reported alongside but is not an arm: inclusion frequency across
#            positive-polarity clauses, which is Blakely & Granmo's Global Feature Strength. It is
#            DATA-INDEPENDENT given a fixed model, so it cannot drift by construction, and it is here
#            to keep the contrast with a data-conditional measure legible and not to be tested.
# PASS/FAIL: not a gate, but it is decisive for what may be written. RETRACT every readability claim if
#            a dataset ranking recovers most of the model's top-k -- concretely, if overlap with
#            chi-squared or frequency reaches half the model's top-100. SURVIVES if dataset overlap
#            sits near the random baseline. The interesting middle is a partial overlap, which licenses
#            "the clauses select features a univariate ranking does not" only for the non-overlapping
#            part, and that part has to be shown, not asserted.
#
#   julia --project=. -t 16 research/interp-dataset-control/run.jl [nseeds]
#
# Prediction 1 of the project's pre-registered set is also settled here, because it is nearly free
# once the model exists: are the clauses BLACKLISTS -- cores that only require features to be ABSENT?
# On IMDb the published one-clause-per-class model was a pure blacklist, and a detector whose clauses
# encode only absence gives an analyst no indicators of compromise, which is a security claim rather
# than an aesthetic one. Sign composition is reported per polarity.

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
const KS = (20, 100, 500)
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

"""
Per-feature counts over the training rows: n11 = (feature present, malware), n10 = (present, benign).
One pass, so the two dataset rankings below share it and cannot disagree about the data.
"""
function feature_counts(X, Y, width)
    n11 = zeros(Int, width); n10 = zeros(Int, width)
    for r in eachindex(X)
        x = X[r]; mal = Y[r]
        @inbounds for i in 1:width
            if x[i]
                mal ? (n11[i] += 1) : (n10[i] += 1)
            end
        end
    end
    return n11, n10
end

"Chi-squared of a binary feature against a binary label, per feature."
function chi2_scores(n11, n10, npos, nneg, width)
    N = npos + nneg
    out = zeros(Float64, width)
    @inbounds for i in 1:width
        a = n11[i]; b = n10[i]            # present & malware, present & benign
        c = npos - a; d = nneg - b        # absent  & malware, absent  & benign
        r1 = a + b; r2 = c + d; c1 = a + c; c2 = b + d
        (r1 == 0 || r2 == 0 || c1 == 0 || c2 == 0) && continue
        num = Float64(a) * d - Float64(b) * c
        out[i] = N * num * num / (Float64(r1) * r2 * c1 * c2)
    end
    return out
end

"P(feature | malware) - P(feature | benign). The crudest dataset ranking."
function df_diff(n11, n10, npos, nneg, width)
    out = zeros(Float64, width)
    @inbounds for i in 1:width
        out[i] = abs(n11[i] / npos - n10[i] / nneg)
    end
    return out
end

"Inclusion frequency across positive-polarity clauses -- Blakely & Granmo's Global Feature Strength."
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
jacc(a, b) = (u = length(union(Set(a), Set(b))); u == 0 ? 0.0 : overlap(a, b) / u)

"Sign composition of one bank: fraction of included literals that are NEGATED (feature absent)."
function sign_composition(bank)
    pos = 0; neg = 0
    for j in 1:bank.nclauses
        for (_, isneg) in clause_literals(bank, j)
            isneg ? (neg += 1) : (pos += 1)
        end
    end
    tot = pos + neg
    return tot == 0 ? (NaN, 0) : (100neg / tot, tot)
end

function main()
    nseeds = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    width = length(feats)
    X, Y = load_train()
    npos = count(Y); nneg = length(Y) - npos

    @printf("LAMDA width %d  train %d rows (%d malware / %d benign)  %d seeds\n",
            width, length(Y), npos, nneg, nseeds)
    @printf("gate config: clauses=%d T=%d S=%d L=%d LF=%d epochs=%d  ceiling=LiteralCapped  tm-lab %s\n",
            CLAUSES, T, S, L, LF, EPOCHS, TM_LAB_COMMIT)
    flush(stdout)

    print("computing dataset rankings (one pass over the training rows) ... ")
    t0 = time()
    n11, n10 = feature_counts(X, Y, width)
    chi2 = chi2_scores(n11, n10, npos, nneg, width)
    dfd  = df_diff(n11, n10, npos, nneg, width)
    @printf("%.1fs\n\n", time() - t0)
    flush(stdout)

    # The explained and background rows: the tail of the training period, matching b2-drift's budget.
    bg = X[1:NBG]
    ex = X[(end - NEX + 1):end]

    rng_ctrl = MersenneTwister(4242)
    rows = []

    for seed in 1:nseeds
        m = TMClassifier([false, true], width;
                         clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
        for e in 1:EPOCHS
            train!(m, X, Y; rng=MersenneTwister(1000seed + e), parallel=:none)
        end

        shap = shapley_importance(m, ex, bg, width)
        incl = inclusion_frequency(m, width)
        nz = count(!iszero, shap)

        negpos, npos_lits = sign_composition(m.positive[CI])
        negneg, nneg_lits = sign_composition(m.negative[CI])

        @printf("seed %d: %d of %d features have nonzero exact attribution\n", seed, nz, width)
        @printf("  sign composition  positive bank %.1f%% negated (%d literals)  negative bank %.1f%% negated (%d)\n",
                negpos, npos_lits, negneg, nneg_lits)
        @printf("RESULT seed=%d nonzero_attrib=%d neg_frac_pos_bank=%.2f neg_frac_neg_bank=%.2f\n",
                seed, nz, negpos, negneg)

        for k in KS
            tm_top = topk(shap, k)
            ic_top = topk(incl, k)
            c_top  = topk(chi2, k)
            d_top  = topk(dfd, k)
            r_top  = randperm(rng_ctrl, width)[1:k]

            @printf("  k=%-4d  model∩chi2 %3d/%d (J %.3f)   model∩freq %3d/%d (J %.3f)   model∩random %3d/%d   model∩inclusion %3d/%d\n",
                    k,
                    overlap(tm_top, c_top), k, jacc(tm_top, c_top),
                    overlap(tm_top, d_top), k, jacc(tm_top, d_top),
                    overlap(tm_top, r_top), k,
                    overlap(tm_top, ic_top), k)
            @printf("RESULT seed=%d k=%d ov_chi2=%d ov_freq=%d ov_random=%d ov_inclusion=%d jac_chi2=%.4f jac_freq=%.4f\n",
                    seed, k, overlap(tm_top, c_top), overlap(tm_top, d_top),
                    overlap(tm_top, r_top), overlap(tm_top, ic_top),
                    jacc(tm_top, c_top), jacc(tm_top, d_top))
            push!(rows, (seed, k, overlap(tm_top, c_top), overlap(tm_top, d_top),
                         overlap(tm_top, r_top), overlap(tm_top, ic_top)))
        end
        flush(stdout)
    end

    println("\n", "-"^92)
    @printf("%-6s %14s %14s %14s %16s\n",
            "k", "∩ chi2", "∩ freq", "∩ random", "∩ inclusion")
    println("-"^92)
    for k in KS
        sel = [r for r in rows if r[2] == k]
        @printf("%-6d %9.1f /%3d %9.1f /%3d %9.1f /%3d %11.1f /%3d\n",
                k,
                mean(r[3] for r in sel), k, mean(r[4] for r in sel), k,
                mean(r[5] for r in sel), k, mean(r[6] for r in sel), k)
        @printf("RESULT mean k=%d chi2=%.2f freq=%.2f random=%.2f inclusion=%.2f of %d\n",
                k, mean(r[3] for r in sel), mean(r[4] for r in sel),
                mean(r[5] for r in sel), mean(r[6] for r in sel), k)
    end
    println("-"^92)
    println("RETRACT every readability claim if chi2 or freq recovers half or more of the model's top-k.")
    println("SURVIVES if they sit near the random column. The random column is what chance alone gives")
    println("and is the only reason the other columns can be read at all.")
end

main()
