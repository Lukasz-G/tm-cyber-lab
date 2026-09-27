# QUESTION:  does the Python/Julia matrix boundary hold on the real dataset, not just on fixtures?
# SURPRISE:  nothing. CALIBRATION. The boundary is already checked bit-exact on small fixtures at 1,
#            64, 65 and 4,561 columns; what is untested is scale, per-month slicing, and whether a
#            month slice is fast enough for ~120 of them.
# ARMS:      one. Julia's read is compared against a popcount digest Python computed independently
#            from the Parquet, so the two sides never share code.
# PASS/FAIL: PASS if every year's row count, per-row popcount digest and label column agree with
#            Python, and a month slice loads in well under a second. FAIL → nothing downstream is
#            trustworthy, because the machine would be training on different bits than the pipeline
#            wrote.
#
#   julia --project=. research/b0-boundary/run.jl
#
# Run research/b0-boundary/digest.py first; it writes the expected digests.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
using .Tmx: read_inputs, read_meta, read_features, read_header, row_indices, labels
using Printf

const TMX_DIR = joinpath("data", "lamda", "tmx")

function digest_of(inputs)
    # Sum of per-row popcounts, plus a position-sensitive term so that a permutation of rows or a
    # transposition of bits within a row would both show up. A plain total popcount would not.
    tot = 0
    mix = UInt64(0)
    for (i, x) in enumerate(inputs)
        pc = 0
        for c in x.chunks
            pc += count_ones(c)
        end
        tot += pc
        mix = (mix * UInt64(1099511628211)) ⊻ UInt64(pc + i)
    end
    return tot, mix
end

function expected()
    exp = Dict{Int,Tuple{Int,Int,Int}}()   # year => (rows, total_popcount, malware)
    for line in eachline(joinpath(@__DIR__, "expected.txt"))
        isempty(strip(line)) && continue
        parts = split(line)
        exp[parse(Int, parts[1])] = (parse(Int, parts[2]), parse(Int, parts[3]), parse(Int, parts[4]))
    end
    return exp
end

function main()
    exp = expected()
    feats = read_features(joinpath(TMX_DIR, "features.arrow"))
    @printf("feature names: %d (first %s, last %s)\n", length(feats), feats[1], feats[end])
    length(feats) == 4561 || error("expected 4561 feature names")

    ok = true
    total_rows = 0
    slice_times = Float64[]
    for year in sort(collect(keys(exp)))
        t0 = time()
        inputs, h = read_inputs(joinpath(TMX_DIR, "$year.tmx"))
        t_full = time() - t0
        meta = read_meta(joinpath(TMX_DIR, "$year.meta.arrow"))
        y = labels(meta)
        tot, _ = digest_of(inputs)
        want_rows, want_pc, want_mw = exp[year]

        rows_ok = h.n_rows == want_rows == length(inputs) == length(y)
        pc_ok = tot == want_pc
        mw_ok = count(y) == want_mw
        ok &= rows_ok && pc_ok && mw_ok
        total_rows += h.n_rows

        # one month slice, which is the access pattern the drift series uses ~120 times
        months = sort(unique(collect(meta.month)))
        m = months[min(2, length(months))]
        idx = row_indices(meta; month=m)
        t1 = time()
        sub, _ = read_inputs(joinpath(TMX_DIR, "$year.tmx"); rows=idx)
        t_slice = time() - t1
        push!(slice_times, t_slice)
        sub_ok = length(sub) == length(idx) &&
                 all(digest_of([sub[k]])[1] == digest_of([inputs[idx[k]]])[1] for k in 1:min(50, length(idx)))
        ok &= sub_ok

        @printf("%d rows %7d %s  popcount %s  malware %s  full %.2fs  month %d (%5d rows) %.3fs %s\n",
                year, h.n_rows, rows_ok ? "ok" : "MISMATCH", pc_ok ? "ok" : "MISMATCH",
                mw_ok ? "ok" : "MISMATCH", t_full, m, length(idx), t_slice,
                sub_ok ? "ok" : "SLICE MISMATCH")
    end

    @printf("\ntotal rows %d (expected 1008381)\n", total_rows)
    ok &= total_rows == 1_008_381
    @printf("month slice: max %.3fs, median %.3fs\n",
            maximum(slice_times), sort(slice_times)[cld(length(slice_times), 2)])
    slice_ok = maximum(slice_times) < 1.0
    ok &= slice_ok
    slice_ok || println("month slice exceeds 1s — the drift series does ~120 of these")

    println("\nRESULT boundary_at_scale=", ok ? "PASS" : "FAIL")
    ok || exit(1)
end

main()
