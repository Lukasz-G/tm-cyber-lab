# Read the `.tmx` packed binary matrix -- the Python/Julia boundary.
#
# Format spec: docs/matrix-format.md. The on-disk layout is TMCore's `TMInput` layout, so a row
# becomes a `TMInput` by copying words, with no bit-level conversion.
#
# Python writes these files (python/tmcyber/tmx.py); Julia only reads them. A derived matrix --
# a resampled base rate, a month slice, a feature subset -- is produced on the Python side, so
# that one language owns the data pipeline and the other owns the machine.

module Tmx

using Arrow
using TMCore: TMInput

export TmxHeader, read_header, read_inputs, read_meta, read_features, nrows, ncols

const MAGIC = b"TMCYBERX"
const VERSION = UInt32(1)
const HEADER_BYTES = 64

struct TmxHeader
    version::UInt32
    flags::UInt32
    n_rows::Int
    n_cols::Int
end

nrows(h::TmxHeader) = h.n_rows
ncols(h::TmxHeader) = h.n_cols
words_per_row(h::TmxHeader) = cld(h.n_cols, 64)

function read_header(io::IO, path::AbstractString="<io>")
    magic = read(io, 8)
    magic == MAGIC || error("$path: bad magic $(String(copy(magic))), expected $(String(copy(MAGIC)))")
    version = read(io, UInt32)
    version == VERSION || error("$path: version $version, this reader speaks $VERSION")
    flags = read(io, UInt32)
    n_rows = Int(read(io, UInt64))
    n_cols = Int(read(io, UInt64))
    read(io, 32)  # reserved
    return TmxHeader(version, flags, n_rows, n_cols)
end

read_header(path::AbstractString) = open(io -> read_header(io, path), path)

"""
    read_inputs(path; rows = nothing) -> (Vector{TMInput}, TmxHeader)

Load a `.tmx` as `TMInput`s, ready for `TMCore.train!` and `predict`.

`rows` selects a subset by 1-based index, which is what the format is for: a month or a year is
read without materializing the rest. Indices must be sorted and unique; the caller gets them from
the metadata sidecar.
"""
function read_inputs(path::AbstractString;
                     rows::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    open(path, "r") do io
        h = read_header(io, path)
        wpr = words_per_row(h)
        if rows !== nothing
            issorted(rows) || error("rows must be sorted")
            allunique(rows) || error("rows must be unique")
            if !isempty(rows)
                (first(rows) >= 1 && last(rows) <= h.n_rows) ||
                    error("rows out of range 1:$(h.n_rows)")
            end
        end
        want = rows === nothing ? (1:h.n_rows) : rows
        base = position(io)
        out = Vector{TMInput}(undef, length(want))
        buf = Vector{UInt64}(undef, wpr)
        prev = 0
        for (k, r) in enumerate(want)
            r == prev + 1 || seek(io, base + (r - 1) * wpr * sizeof(UInt64))
            read!(io, buf)
            chunks = Memory{UInt64}(undef, wpr)
            copyto!(chunks, buf)
            out[k] = TMInput(chunks, h.n_cols)
            prev = r
        end
        return out, h
    end
end

"""
    read_meta(path) -> Arrow.Table

The row-metadata sidecar: `label`, `year`, `month`, `split`, `family`, `vt_detection`. One row per
matrix row, in matrix order.
"""
read_meta(path::AbstractString) = Arrow.Table(path)

"""
    read_features(path) -> Vector{String}

Feature names, one per matrix column, in column order. Needed to say what a clause literal means,
and therefore needed before any interpretability claim.
"""
read_features(path::AbstractString) = collect(String.(Arrow.Table(path).name))

"""
    row_indices(meta; year = nothing, month = nothing, split = nothing) -> Vector{Int}

Sorted 1-based row indices matching the given metadata filters, ready for `read_inputs`.

Pass `year` and `month` as integers or as collections. `split` is one of `"train"`, `"iid"`,
`"near"`, `"far"`, or a collection of them. A `far` slice should always be taken per year: the
2024-25 malware counts collapse to 794 and 23 samples against ~45,000 benign, which is AV label
lag and not drift, so a FAR mean across 2018-2025 is not a meaningful number.
"""
matches(_, ::Nothing) = true
matches(v, want::AbstractString) = v == want
matches(v, want::Number) = v == want
matches(v, want) = v in want   # any collection: vector, tuple, range, set

function row_indices(meta; year=nothing, month=nothing, split=nothing)
    ys, ms, ss = meta.year, meta.month, meta.split
    idx = Int[]
    for i in 1:length(ys)
        matches(ys[i], year) || continue
        matches(ms[i], month) || continue
        matches(ss[i], split) || continue
        push!(idx, i)
    end
    return idx
end

"""
    labels(meta, idx) -> Vector{Bool}

Labels for the rows `read_inputs` returned, in the same order. `true` is malware.
"""
labels(meta, idx::AbstractVector{<:Integer}) = Bool[meta.label[i] for i in idx]
labels(meta) = collect(Bool.(meta.label))

end # module Tmx
