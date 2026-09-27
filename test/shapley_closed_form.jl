# Check the closed-form Shapley values against brute-force enumeration over all 2^n coalitions.
#
#   julia --project=. test/shapley_closed_form.jl
#
# B2's whole claim is that these attributions are exact where a KernelExplainer's are sampled, so
# "exact" has to be demonstrated rather than asserted. Brute force is only tractable at ~14
# features, which is why this runs on small trained models rather than on LAMDA — but the formula
# has no width-dependent term, so a small model exercises every branch.
#
# Two properties are checked, and they fail differently:
#   * agreement with brute force, per feature
#   * efficiency, that the values sum to score(x) - score(background)
# Efficiency alone is not enough. An early version passed efficiency and still split the total
# wrongly between features, which is how the both-polarities case was found.

using TMCore
using Random
using Combinatorics: combinations

include(joinpath(@__DIR__, "..", "julia", "shapley.jl"))
using .Shapley: shapley

const N = 14   # 2^14 coalitions per feature is the practical brute-force ceiling

function toy_model(seed, LF, pol)
    rng = MersenneTwister(seed)
    X = [TMInput(Vector{Bool}(rand(rng, N) .< 0.4)) for _ in 1:400]
    Y = Bool[x[2] & !x[5] for x in X]
    m = TMClassifier([false, true], N; clauses_per_class=8, T=4, S=20, L=10, LF=LF, ceiling=pol)
    for e in 1:12
        train!(m, X, Y; rng=MersenneTwister(seed + e))
    end
    return m
end

hybrid(x, b, S) = TMInput(Bool[i in S ? x[i] : b[i] for i in eachindex(x)])

"Shapley by definition: every coalition, exact rational weights."
function brute_shapley(m, ci, x, b)
    n = length(x)
    fact = [factorial(big(k)) for k in 0:n]
    v = Dict{Set{Int},Int}()
    for k in 0:n, c in combinations(1:n, k)
        v[Set(c)] = score(m, ci, hybrid(x, b, Set(c)))
    end
    v[Set{Int}()] = score(m, ci, hybrid(x, b, Set{Int}()))
    phi = zeros(Rational{BigInt}, n)
    for i in 1:n
        acc = Rational{BigInt}(0)
        for k in 0:(n - 1), c in combinations(setdiff(1:n, i), k)
            S = Set(c)
            acc += Rational{BigInt}(fact[k + 1] * fact[n - k], fact[n + 1]) * (v[union(S, i)] - v[S])
        end
        phi[i] = acc
    end
    return Float64.(phi)
end

function main()
    worst, cases = 0.0, 0
    for seed in 1:4, LF in (2, 3, 6), pol in (LiteralCapped(), FlatLF())
        m = toy_model(seed, LF, pol)
        rng = MersenneTwister(100 + seed)
        for _ in 1:3
            x = Vector{Bool}(rand(rng, N) .< 0.4)
            b = Vector{Bool}(rand(rng, N) .< 0.4)
            for ci in 1:2
                bf = brute_shapley(m, ci, x, b)
                cf = shapley(m, ci, x, b)
                worst = max(worst, maximum(abs.(bf .- cf)))
                gap = score(m, ci, TMInput(x)) - score(m, ci, TMInput(b))
                abs(sum(cf) - gap) < 1e-6 ||
                    error("efficiency violated: closed form sums to $(sum(cf)), expected $gap")
                cases += 1
            end
        end
    end
    println("compared $cases (model, LF, ceiling, instance, background, class) cases at n=$N")
    println("max |closed form - brute force| = ", worst)
    worst < 1e-9 || error("closed form disagrees with brute force")
    println("EXACT MATCH")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
