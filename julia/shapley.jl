# Exact Shapley attribution for a Fuzzy-Pattern Tsetlin Machine.
#
# This is the measurement B2 rests on, so the claim it makes is worth stating precisely: these are
# the *same* Shapley values a KernelExplainer estimates by sampling coalitions, computed in closed
# form with no sampling and no estimator variance. `test/shapley_closed_form.jl` checks that
# against brute-force enumeration over all 2^n coalitions on trained models.
#
# ---------------------------------------------------------------------------
# Derivation
#
# The value function is the usual interventional one: v(S) = f(z_S), where z_S takes the explained
# instance's value on features in S and a background instance's value elsewhere. Averaging over a
# background set is a separate outer sum, because Shapley is linear in v.
#
# A TM score is a signed sum of clause votes, and Shapley is linear in f, so it is enough to solve
# one clause and add up. A clause's vote is max(0, ceiling - misses), and misses counts unsatisfied
# literals. Classify each literal of the clause by how it behaves at x versus at the background b:
#
#   A  satisfied under both     — never a miss
#   B  unsatisfied under both   — always a miss                       (count into beta)
#   C  satisfied only under x   — a miss unless its feature is in S   (set C, size g)
#   D  satisfied only under b   — a miss when its feature IS in S     (set D, size d)
#
# so  misses(S) = beta + (g - |C n S|) + |D n S|.
#
# That depends only on the two *counts*, never on which literals. Three consequences:
#
#   1. Every feature outside C u D has Shapley value exactly zero for this clause. For sparse
#      Drebin data most literals sit on features that are zero in both x and b, so C u D is small
#      and the cost has nothing to do with the 4,561-feature width.
#   2. All features in C share one value, and all in D share another.
#   3. Adding a C feature lowers misses by one, which raises the vote by one exactly when
#      misses <= ceiling; adding a D feature raises misses by one, lowering the vote by one exactly
#      when misses <= ceiling - 1. The marginal contribution is therefore always 0 or +-1.
#
# Take a uniform random permutation of all features. Among the g + d relevant ones the relative
# order is uniform, so r = how many of them precede i is uniform on 0:(g+d-1), and given r the
# split k between C and D is hypergeometric. With misses = beta + g + r - 2k the whole Shapley
# value collapses to a double sum of O((g+d)^2) terms, independent of the feature count.
#
# One case needs care. A feature whose *both* polarities are included lands in C and in D at once;
# flipping it fixes one literal and breaks the other, so misses cannot see it and its value is
# zero. It is not free, though — exactly one of the pair is unsatisfied whatever the value, so it
# contributes a constant miss and must be folded into beta. Omitting that term breaks efficiency,
# which is how the bug was caught.
# ---------------------------------------------------------------------------

module Shapley

using TMCore: TMClassifier, ClauseBank, TMInput, CeilingPolicy, LiteralCapped, FlatLF

export clause_literals, shapley, shapley_mean

"Included literals of clause `j` as `(feature, is_negated)` pairs."
function clause_literals(b::ClauseBank, j::Integer)
    out = Tuple{Int,Bool}[]
    for n in 1:b.nchunks, (mask, neg) in ((b.included[n, j], false), (b.included_inv[n, j], true))
        iszero(mask) && continue
        for bit in 0:63
            f = (n - 1) * 64 + bit + 1
            f > b.width && break
            iszero(mask & (UInt64(1) << bit)) || push!(out, (f, neg))
        end
    end
    return out
end

ceiling_of(b::ClauseBank, j, LF, ::LiteralCapped) = (c = Int(b.count[j]); c == 0 ? LF : min(c, LF))
ceiling_of(::ClauseBank, ::Any, LF, ::FlatLF) = LF

"Shared Shapley value of a C-member (`from_c`) or a D-member, for one clause."
function _unit_value(beta::Int, g::Int, d::Int, ceil::Int, from_c::Bool)
    tot = g + d
    tot == 0 && return 0.0
    from_c ? (g == 0 && return 0.0) : (d == 0 && return 0.0)
    limit = from_c ? ceil : ceil - 1        # misses threshold for a +-1 marginal contribution
    nc = from_c ? g - 1 : g                 # other C members
    nd = from_c ? d : d - 1                 # other D members
    acc = 0.0
    for r in 0:(tot - 1)
        denom = binomial(big(tot - 1), big(r))
        s = 0.0
        for k in max(0, r - nd):min(r, nc)
            beta + g + r - 2k <= limit || continue
            s += Float64(binomial(big(nc), big(k)) * binomial(big(nd), big(r - k)) / denom)
        end
        acc += s
    end
    return acc / tot
end

"Accumulate one clause's exact Shapley values into `phi`."
function clause_shapley!(phi::Vector{Float64}, bank::ClauseBank, j::Integer,
                         x::AbstractVector{Bool}, b::AbstractVector{Bool},
                         LF::Integer, pol::CeilingPolicy, sign::Int)
    lits = clause_literals(bank, j)
    isempty(lits) && return nothing         # an empty clause votes its ceiling regardless of input
    sat(v, neg) = neg ? !v : v
    beta = 0
    cset = Set{Int}(); dset = Set{Int}()
    for (f, neg) in lits
        sx, sb = sat(x[f], neg), sat(b[f], neg)
        if sx && sb
            # A
        elseif !sx && !sb
            beta += 1
        elseif sx && !sb
            push!(cset, f)
        else
            push!(dset, f)
        end
    end
    both = intersect(cset, dset)
    beta += length(both)                    # one of the pair always misses; see the header
    C = setdiff(cset, both); D = setdiff(dset, both)
    g, d = length(C), length(D)
    g + d == 0 && return nothing            # this clause cannot react to this (x, b) pair

    ceil = ceiling_of(bank, j, LF, pol)
    if g > 0
        v = _unit_value(beta, g, d, ceil, true)
        for f in C
            phi[f] += sign * v
        end
    end
    if d > 0
        v = _unit_value(beta, g, d, ceil, false)
        for f in D
            phi[f] -= sign * v
        end
    end
    return nothing
end

"""
    shapley(model, ci, x, b) -> Vector{Float64}

Exact Shapley values of every feature for class index `ci`, explaining instance `x` against a
single background instance `b`. Sums to `score(m, ci, x) - score(m, ci, b)` by construction.
"""
function shapley(m::TMClassifier, ci::Integer, x::AbstractVector{Bool}, b::AbstractVector{Bool})
    phi = zeros(Float64, length(x))
    shapley!(phi, m, ci, x, b)
    return phi
end

function shapley!(phi::Vector{Float64}, m::TMClassifier, ci::Integer,
                  x::AbstractVector{Bool}, b::AbstractVector{Bool})
    LF, pol = m.params.LF, m.ceiling
    for (bank, sign) in ((m.positive[ci], 1), (m.negative[ci], -1))
        for j in 1:bank.nclauses
            clause_shapley!(phi, bank, j, x, b, LF, pol, sign)
        end
    end
    return phi
end

"""
    shapley_mean(model, ci, x, backgrounds) -> Vector{Float64}

Exact Shapley values against a background *set*, which is what `KernelExplainer` approximates when
it is handed 100 background rows. Linear in the background, so this is a plain mean — still exact.
"""
function shapley_mean(m::TMClassifier, ci::Integer, x::AbstractVector{Bool},
                      backgrounds::AbstractVector{<:AbstractVector{Bool}})
    phi = zeros(Float64, length(x))
    for b in backgrounds
        shapley!(phi, m, ci, x, b)
    end
    phi ./= length(backgrounds)
    return phi
end

end # module Shapley
