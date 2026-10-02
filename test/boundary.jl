# Julia half of the cross-language boundary check. See test/boundary.py for the protocol.
#
#   julia --project=. test/boundary.jl <dir>
#
# Reads each fixture through the same code path training uses -- Tmx.read_inputs into TMInput --
# and writes back the set-bit indices of every row. Python compares. A disagreement here means the
# machine would train on different bits than the pipeline wrote, which is the kind of bug that
# produces a plausible wrong number, not an error.

include(joinpath(@__DIR__, "..", "julia", "tmx.jl"))

using .Tmx: read_inputs, read_meta, read_features

function row_line(x)
    idx = Int[]
    for i in eachindex(x)
        x[i] && push!(idx, i)
    end
    return join(idx, " ")
end

function main(dir::AbstractString)
    stems = split(read(joinpath(dir, "manifest.txt"), String))
    for stem in stems
        inputs, h = read_inputs(joinpath(dir, "$stem.tmx"))
        meta = read_meta(joinpath(dir, "$stem.meta.arrow"))
        feats = read_features(joinpath(dir, "$stem.features.arrow"))

        length(inputs) == h.n_rows || error("$stem: read $(length(inputs)) rows, header says $(h.n_rows)")
        length(meta.label) == h.n_rows || error("$stem: metadata rows disagree with the header")
        length(feats) == h.n_cols || error("$stem: $(length(feats)) feature names, $(h.n_cols) columns")
        all(length(x) == h.n_cols for x in inputs) || error("$stem: an input is not n_cols wide")

        # A subset read must agree with the full read, since every month slice depends on it.
        if h.n_rows >= 3
            want = [1, h.n_rows]
            sub, _ = read_inputs(joinpath(dir, "$stem.tmx"); rows=want)
            for (k, r) in enumerate(want)
                row_line(sub[k]) == row_line(inputs[r]) || error("$stem: subset read differs at row $r")
            end
        end

        open(joinpath(dir, "$stem.actual.txt"), "w") do io
            for x in inputs
                println(io, row_line(x))
            end
        end
        println("read $stem: $(h.n_rows) rows x $(h.n_cols) cols")
    end
    println("\nnext: python test/boundary.py verify $dir")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 1 || error("usage: julia --project=. test/boundary.jl <dir>")
    main(ARGS[1])
end
