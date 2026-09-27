# QUESTION:  the closed-form Shapley derivation used only one property of a fuzzy clause — that its
#            output depends on HOW MANY of its literals are unsatisfied, not which. Which model
#            classes does that cover?
# SURPRISE:  yes, and it decides the scope of the whole method contribution. If the derivation needs
#            the specific form max(0, ceiling - misses), the result is "exact Shapley for
#            Fuzzy-Pattern Tsetlin machines" — narrow. If it needs only count-symmetry, the same
#            closed form covers classical Tsetlin machines, m-of-n threshold ensembles and weighted
#            rule ensembles of the RuleFit kind, which is a far larger claim and one that reaches the
#            rule-based drift literature directly.
# ARMS:      four response functions over the same machinery, spanning the conjectured class:
#              (1) fuzzy clause      g(m) = max(0, ceiling - m)      [the known case, as a control]
#              (2) classical AND     g(m) = 1[m == 0]                [classical TM / a conjunction]
#              (3) m-of-n threshold  g(m) = 1[m <= t]
#              (4) arbitrary shape   g(m) = random per miss count     [count-symmetric but nothing else]
#            Arm (4) is the one that matters: if the formula holds for an arbitrary function of the
#            count, then count-symmetry is the whole requirement and nothing about fuzziness is load
#            bearing. Arm (1) is the control that catches a broken harness.
# PASS/FAIL: PASS if the generalised closed form matches brute-force enumeration over all 2^n
#            coalitions for every arm, to floating-point. FAIL on any arm bounds the claim to the arms
#            that passed, and that bound goes in the paper.
#
#   julia --project=. research/b2-generality/run.jl
#
# A NEGATIVE result here is also reported: ordered rule lists (RIPPER's "first matching rule wins")
# are deliberately NOT in the conjectured class, because the prediction is not a sum over rules and
# Shapley's linearity over components is what the construction rests on. Arm (5) checks that the
# formula indeed fails there, so the boundary of the claim is established by measurement rather than
# by assertion.

using Random, Printf
using Combinatorics: combinations

const N = 13   # 2^13 coalitions brute-forced per feature

# ---------------------------------------------------------------------------
# A generic count-symmetric rule ensemble: prediction is sum over rules of w * g(misses).
# ---------------------------------------------------------------------------
struct Rule
    lits::Vector{Tuple{Int,Bool}}   # (feature, is_negated)
    w::Float64
    g::Vector{Float64}              # g[m+1] = response when m literals are unsatisfied
end

satisfied(val::Bool, neg::Bool) = neg ? !val : val

function misses(r::Rule, x::AbstractVector{Bool})
    m = 0
    for (f, neg) in r.lits
        satisfied(x[f], neg) || (m += 1)
    end
    return m
end

value(rules::Vector{Rule}, x::AbstractVector{Bool}) =
    sum(r.w * r.g[misses(r, x) + 1] for r in rules)

"Ordered rule list: the first rule that fires decides. Deliberately NOT count-additive."
function value_ordered(rules::Vector{Rule}, x::AbstractVector{Bool})
    for r in rules
        misses(r, x) == 0 && return r.w
    end
    return 0.0
end

# ---------------------------------------------------------------------------
# Brute-force Shapley over every coalition, against whatever value function is given.
# ---------------------------------------------------------------------------
function brute(f, x::Vector{Bool}, b::Vector{Bool})
    n = length(x)
    fact = [factorial(big(k)) for k in 0:n]
    hybrid(S) = Bool[i in S ? x[i] : b[i] for i in 1:n]
    v = Dict{Set{Int},Float64}()
    for k in 0:n, c in combinations(1:n, k)
        v[Set(c)] = f(hybrid(Set(c)))
    end
    v[Set{Int}()] = f(hybrid(Set{Int}()))
    phi = zeros(Float64, n)
    for i in 1:n
        acc = 0.0
        for k in 0:(n - 1), c in combinations(setdiff(1:n, i), k)
            S = Set(c)
            w = Float64(Rational{BigInt}(fact[k + 1] * fact[n - k], fact[n + 1]))
            acc += w * (v[union(S, i)] - v[S])
        end
        phi[i] = acc
    end
    return phi
end

# ---------------------------------------------------------------------------
# The generalised closed form. Identical in structure to julia/shapley.jl, with the +-1 marginal
# replaced by g(M-1) - g(M) for a C member and g(M+1) - g(M) for a D member. Nothing else changes,
# which is precisely the claim under test.
# ---------------------------------------------------------------------------
function closed(rules::Vector{Rule}, x::AbstractVector{Bool}, b::AbstractVector{Bool}, n::Int)
    phi = zeros(Float64, n)
    for r in rules
        beta = 0
        cset = Set{Int}(); dset = Set{Int}()
        for (f, neg) in r.lits
            sx, sb = satisfied(x[f], neg), satisfied(b[f], neg)
            if sx && sb
                # constant satisfied
            elseif !sx && !sb
                beta += 1
            elseif sx && !sb
                push!(cset, f)
            else
                push!(dset, f)
            end
        end
        both = intersect(cset, dset)
        beta += length(both)          # one of the pair always misses
        C = setdiff(cset, both); D = setdiff(dset, both)
        g, d = length(C), length(D)
        tot = g + d
        tot == 0 && continue

        for (members, from_c) in ((C, true), (D, false))
            isempty(members) && continue
            nc = from_c ? g - 1 : g
            nd = from_c ? d : d - 1
            acc = 0.0
            for rr in 0:(tot - 1)
                denom = binomial(big(tot - 1), big(rr))
                for k in max(0, rr - nd):min(rr, nc)
                    M = beta + g + rr - 2k
                    delta = from_c ? (r.g[M] - r.g[M + 1]) : (r.g[M + 2] - r.g[M + 1])
                    iszero(delta) && continue
                    p = Float64(binomial(big(nc), big(k)) * binomial(big(nd), big(rr - k)) / denom)
                    acc += p * delta
                end
            end
            acc /= tot
            for f in members
                phi[f] += r.w * acc
            end
        end
    end
    return phi
end

# ---------------------------------------------------------------------------
function make_rules(rng, kind::Symbol, nrules::Int)
    rules = Rule[]
    for _ in 1:nrules
        nl = rand(rng, 2:6)
        feats = randperm(rng, N)[1:nl]
        lits = [(f, rand(rng, Bool)) for f in feats]
        maxm = nl
        gv = if kind === :fuzzy
            ceiling = min(nl, rand(rng, 2:4))
            Float64[max(0, ceiling - m) for m in 0:maxm]
        elseif kind === :conjunction
            Float64[m == 0 ? 1.0 : 0.0 for m in 0:maxm]
        elseif kind === :threshold
            t = rand(rng, 0:max(0, nl - 2))
            Float64[m <= t ? 1.0 : 0.0 for m in 0:maxm]
        else  # :arbitrary — count-symmetric and nothing else
            Float64[randn(rng) for _ in 0:maxm]
        end
        push!(rules, Rule(lits, rand(rng) * 2 - 1, gv))
    end
    return rules
end

function main()
    println("n = $N features, brute-forcing all $(2^N) coalitions per case\n")
    @printf("%-34s %14s %10s\n", "arm", "max |closed-bf|", "verdict")
    allpass = true
    for kind in (:fuzzy, :conjunction, :threshold, :arbitrary)
        worst = 0.0
        for trial in 1:6
            rng = MersenneTwister(100trial + Int(hash(kind) % 1000))
            rules = make_rules(rng, kind, rand(rng, 2:5))
            x = Vector{Bool}(rand(rng, N) .< 0.45)
            b = Vector{Bool}(rand(rng, N) .< 0.45)
            bf = brute(z -> value(rules, z), x, b)
            cf = closed(rules, x, b, N)
            worst = max(worst, maximum(abs.(bf .- cf)))
            gap = value(rules, x) - value(rules, b)
            abs(sum(cf) - gap) < 1e-8 ||
                error("$kind: efficiency violated, $(sum(cf)) vs $gap")
        end
        pass = worst < 1e-9
        allpass &= pass
        @printf("%-34s %14.2e %10s\n", "additive, g = $kind", worst, pass ? "MATCH" : "MISMATCH")
    end

    # The boundary: an ordered rule list is not a sum over rules, so linearity over components — the
    # thing the construction rests on — does not apply. The formula should FAIL here, and it is worth
    # showing that it does rather than claiming it would.
    worst_ord = 0.0
    for trial in 1:6
        rng = MersenneTwister(7000 + trial)
        rules = make_rules(rng, :conjunction, rand(rng, 2:5))
        x = Vector{Bool}(rand(rng, N) .< 0.45)
        b = Vector{Bool}(rand(rng, N) .< 0.45)
        bf = brute(z -> value_ordered(rules, z), x, b)
        cf = closed(rules, x, b, N)          # treats the list as additive, which it is not
        worst_ord = max(worst_ord, maximum(abs.(bf .- cf)))
    end
    @printf("%-34s %14.2e %10s\n", "ORDERED list (expected to fail)", worst_ord,
            worst_ord > 1e-6 ? "FAILS (as predicted)" : "unexpectedly matched")

    println()
    println("RESULT additive_count_symmetric=", allpass ? "EXACT" : "MISMATCH")
    @printf("RESULT ordered_list_error=%.3e\n", worst_ord)
    println("RESULT b2_generality=", allpass && worst_ord > 1e-6 ? "PASS" : "FAIL")
end

main()
