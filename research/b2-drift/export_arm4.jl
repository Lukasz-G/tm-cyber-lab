# Exports one trained model, its rows, and its EXACT Shapley importances, so that arm 4 can run
# shap.KernelExplainer against the closed form on the identical model.
#
# Arm 4 is the one arm no published work can run: it needs a model whose exact Shapley values are
# computable, so that sampled and exact attribution can be compared with the model, the data and the
# attribution target all held fixed. Everything that varies between the two sides is then the
# estimator.
#
# The boundary is deliberately dumb -- clause literal lists and per-clause ceilings as JSON, plus the
# rows as a dense 0/1 matrix. Python re-implements the clause vote from that, and `arm4.py` checks its
# scores against the Julia scores exported here before it explains anything. If that check fails the
# comparison is meaningless, so it is a hard assertion in place of a warning.
#
#   julia --project=. -t 16 research/b2-drift/export_arm4.jl [year] [month]

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: shapley_mean, clause_literals
using TMCore
using Random, Printf, Statistics, JSON, Base.Threads

const TMX_DIR = joinpath("data", "lamda", "tmx")
const OUT = joinpath("data", "b2-arm4")
const CLAUSES, T, S, L, LF = 20, 10, 100, 64, 10
const EPOCHS = 30
const CI = 2
const NBG, NEX = 100, 100

function main()
    year  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2016
    month = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
    mkpath(OUT)

    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    width = length(feats)
    meta  = read_meta(joinpath(TMX_DIR, "$year.meta.arrow"))
    path  = joinpath(TMX_DIR, "$year.tmx")

    sel(split) = [i for i in eachindex(meta.month)
                  if !ismissing(meta.month[i]) && meta.month[i] == month &&
                     meta.file_split[i] == split]
    tr_idx, te_idx = sel("train"), sel("test")
    length(tr_idx) >= NBG || error("month has only $(length(tr_idx)) train rows")
    length(te_idx) >= NEX || error("month has only $(length(te_idx)) test rows")

    tr, _ = read_inputs(path; rows=tr_idx)
    te, _ = read_inputs(path; rows=te_idx)
    Ytr = Bool[Bool(meta.label[i]) for i in tr_idx]
    bg, ex = tr[1:NBG], te[1:NEX]

    @printf("%d-%02d  width %d  train %d  test %d\n", year, month, width, length(tr), length(te))

    m = TMClassifier([false, true], width;
                     clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    for e in 1:EPOCHS
        train!(m, tr, Ytr; rng=MersenneTwister(1000 + e), parallel=:none)
    end

    # Clauses of the explained class, both polarities, with the ceiling the model will actually use.
    clauses = Any[]
    for (bank, sgn) in ((m.positive[CI], 1), (m.negative[CI], -1))
        for j in 1:bank.nclauses
            lits = clause_literals(bank, j)
            c = Int(bank.count[j])
            push!(clauses, Dict("sign" => sgn,
                                "ceiling" => (c == 0 ? LF : min(c, LF)),
                                "features" => [l[1] for l in lits],
                                "negated" => [l[2] for l in lits]))
        end
    end

    # Exact Shapley importance, and the raw scores that let Python verify its own scoring first.
    imp = zeros(Float64, width)
    parts = [zeros(Float64, width) for _ in 1:nthreads()]
    @threads for r in eachindex(ex)
        phi = shapley_mean(m, CI, ex[r], bg)
        p = parts[threadid()]
        @inbounds for i in 1:width
            p[i] += abs(phi[i])
        end
    end
    imp .= reduce(+, parts) ./ length(ex)

    dense(rows) = [Int8[r[i] ? 1 : 0 for i in 1:width] for r in rows]

    open(joinpath(OUT, "model.json"), "w") do io
        JSON.print(io, Dict(
            "year" => year, "month" => month, "width" => width, "LF" => LF, "class_index" => CI,
            "clauses" => clauses,
            "score_ex" => [Float64(score(m, CI, x)) for x in ex],
            "score_bg" => [Float64(score(m, CI, x)) for x in bg],
            "exact_importance" => imp,
        ))
    end
    open(joinpath(OUT, "rows.json"), "w") do io
        JSON.print(io, Dict("background" => dense(bg), "explained" => dense(ex)))
    end

    @printf("wrote %s  (%d clauses, %d nonzero exact importances)\n",
            OUT, length(clauses), count(!iszero, imp))
end

main()
