# QUESTION:  measured the same way on four Android corpora, how much of a malware detector's month-to-month
#            explanation drift is data, how much is refitting, and how much is the sampled estimator? Corpora,
#            each with a training pool and a monthly series after it:
#              LAMDA     pool 2013-2014, series 2016-01 to 2022-12 (no 2015 in the release), 4,561 features
#              APIGraph  pool 2012,      series 2013-01 to 2018-12, 72 months,                1,159 features
#              AndroZoo  pool 2019,      series 2020-01 to 2021-12, 24 months, 16,978 features at 39% density
#              Hypercube pool 2021,      series 2022-01 to 2023-12, 24 months, 4,087 features at 3.8% density
#            (APIGraph and AndroZoo as released with Chen et al. 2023, data/apigraph/tmx/; LAMDA from
#            data/lamda/tmx/; Hypercube, Chow et al. 2025, Google Play, Drebin features after
#            VarianceThreshold(0.001) fitted on the 2021 pool, data/hypercube/tmx/). LAMDA is measured here
#            under the same construction as the other three, and the paper reports these numbers.
# SURPRISE:  predicted before the run: (P1) on every corpus the sampled estimator at nsamples=100 is dominated
#            by variance, not bias (F2 does not fire); (P2) arm 2s minus arm 2 is at least 0.30 on every
#            corpus, so on the identical model most of the sampled drift is the estimator; (P3) the corrected
#            overlap of arm 2 exceeds that of arm 3 on every corpus. Outcomes that would change the paper:
#            explanation drift is real on some corpus (F1), which becomes that corpus's headline; or the
#            estimator is biased there (F2), which withdraws "variance, not bias" for that corpus.
# ARMS:      every arm on every corpus, under every condition; the comparison between arms is the result.
#              2  fixed model trained on the pool, exact attributions per month, consecutive months: data
#                 drift. Trained at 2 pool seeds; every falsifier must hold for both.
#              2r fixed model, same month, two DISJOINT stratified draws, background redrawn as well: the
#                 row-draw floor of arm 2.
#              2s fixed model, KernelExplainer at nsamples=100 with l1_reg = num_features(10) passed explicitly
#                 (the released call passes none; that is its default under shap 0.49.1, pinned, the version
#                 b2-drift ran); consecutive months: sampled drift on the identical model; 2s - 2 = estimator.
#              3  model refitted per month on the month's train half, 3 refit seeds; 3 - 2 is refitting.
#              1  seed floor of arm 3: same month, two refit seeds.
#              1n, 3n  arms 1 and 3 refitted on a common 1,000 rows at 10% malware on every corpus.
#              5  arm 2 over the month's whole test half, subsampled to 10% malware.
#              4  on the identical fixed model, three months per corpus (first, middle and last of the series),
#                 5 sampled runs each: 4a sampled at 100 vs exact; 4b sampled at 100 vs a second sampled run;
#                 4c sampled at 1,000 vs exact (10x the published budget on every corpus; the fraction of
#                 shap's default 2*M_eff+2048 is recorded); 4f as 4a with l1_reg=False. Bias is tested in
#                 attribution space, not on top-k sets: per explained row, bias^2 = |mean of 5 runs - exact|^2
#                 minus (across-run variance)/5, and the bias share is bias^2 / (bias^2 + variance).
#              0  the released LAMDA protocol here: MLP refitted per month on the month's training set of each
#                 condition, KernelExplainer as in 2s, 3 runs; consecutive months, and same month across runs
#                 (0w). Model class differs from the TM arms as well as estimator: 0 - 2s is reported, not
#                 decomposed. Python, no S.
#            CONDITIONS (the same text as research/threshold-corpora where the rule is shared), for every
#            trained model: the fixed FPTM of arms 2, 2r, 5, 2s, 4; the refitted FPTM of 1, 3, 1n, 3n; the MLP
#            of arm 0. No class weights, sample weights or loss weights in any model, the MLP included.
#              B0  the training split cut to exactly 1:9: k = min(M, G div 9) malware rows and 9k goodware
#                  rows, drawn at random without replacement. The split is the pool for the fixed models, the
#                  month's train half for a refit and the arm-0 MLP, and the common 1,000 rows for 1n/3n.
#                  LAMDA 2016-03: a train half of 11,582 malware and 2,238 goodware gives B0 = 248 and 2,232.
#              U   B0 with the majority class undersampled at random, without replacement, to the minority
#                  count: 50/50, size 2 x minority.
#              O   B0 with the minority class oversampled to the majority count: each minority row repeated
#                  floor(R) times (R = majority/minority), plus round((R - floor R) x minority) further
#                  minority rows drawn without replacement: 50/50, size 2 x majority. At exactly 10% that is
#                  x9.
#            Majority and minority are decided per training set, not assumed to be goodware and malware.
#            Seed s draws the U and O resample s and trains model s. Explained and background rows stay at
#            10% and are the same draws in every condition. Under U the models of arms 1, 1n and 0w differ in
#            their resample as well as their seed, so there they are a seed-and-resample floor.
#            Decomposition, per corpus and condition: exact data drift = arm 2; refitting = arm 3 - arm 2;
#            estimator on the identical model = arm 2s - arm 2; the released protocol = arm 0, beside them.
#            P is the planted positive control (PASS/FAIL). Configurations, construction: DESIGN below.
# PASS/FAIL: not a gate; a per-corpus measurement, every outcome reported, every claim scoped to its corpus,
#            configuration and training condition. Every training condition is reported and none is chosen
#            after the run; the decomposition and F1-F3 below are evaluated per condition. Metrics beside
#            every Jaccard: the supports, the chance level E for two random
#            top-k sets from them, and the corrected overlap c = (|intersection| - E) / (k - E), 1 = identical,
#            0 = chance. Statistics: pairs only between calendar-consecutive months (a gap breaks the chain);
#            95% intervals by a moving-block bootstrap over pairs within a year, block length 3, 2,000
#            resamples; a year decides only with at least 6 usable pairs; a condition holds for a corpus when
#            it holds in a majority of its deciding years, for both pool seeds; "within 0.05" means the
#            interval of the difference lies inside [-0.05, 0.05]. Fixed now, at k=100:
#              F1 explanation drift is real there: c(arm 2) <= 0.30 and c(arm 2r) - c(arm 2) >= 0.30, both by
#                 the interval bound.
#              F2 the estimator is biased there: the lower interval bound of the bias share (rows resampled)
#                 exceeds 0.25 at nsamples=100 with l1_reg as in 2s. "Variance, not bias" is then
#                 withdrawn for that corpus; 4f says whether l1_reg causes it.
#              F3 the budget is not the cause there: 4c within 0.05 of 4a.
#              Unresolved at 20 clauses per class there: c(arm 1) within 0.05 of both c(arm 2) and c(arm 3).
#                 Kept in the table as a scoped negative on how far standing claim 3 reaches.
#            Positive control P (null player), criteria fixed before it runs (2026-11-13; it replaces a plant
#            in the explained rows only, whose answer was not guaranteed): on the fixed model of pool seed
#            101, in every configuration and condition, at the first arm-4 month, the 20 features of highest
#            exact importance are set to 0 in BOTH the explained and the background rows; no model is
#            trained. Each planted feature then equals every background row, so it is never a player.
#              (a) the maximum over explained rows of |phi_i| for the 20 planted features is exactly 0;
#              (b) whenever the planted support is at least 100, J(P) at k=100 is at least 1/3: at most 80
#                  of the original top-100 can remain, so J = 1 - |A n B| / (200 - |A n B|) >= 1/3.
#            J(P) minus the arm-2r floor of the same model is reported beside them. A miss on (a) or (b)
#            is a failure and is reported as one. Negative controls: arm 2r (row draws), arm 1 (seeds).
#            EXPLORATORY, no pass/fail (written before launch, 2026-11-13): the drift-vs-lag curve. For arm 2
#            (exact) and arm 2s (KernelExplainer), on every corpus, configuration, condition and pool seed,
#            the Jaccard distance and the corrected overlap c at k=100 between month t and month t+L, for L in
#            {1, 2, 3, 6, 12} calendar months, wherever both months exist; pairs may span a calendar gap, and
#            the number of missing months in between is recorded. Per L: the mean and a moving-block bootstrap
#            95% interval over the pairs in calendar order (block 3, 2,000 resamples, one seed per test), with
#            the arm-2r same-month floor beside it. Its purpose is to tell drift that accumulates with lag from
#            a flat curve at the noise floor (Webb et al. 2016, DMKD 30(4)). Computed by decide.py from the
#            saved attributions, after the run; it decides nothing.
#            No F1 score or threshold is computed in this run.
#
# DESIGN:    configurations: S = 100, and S matched to LAMDA's s = width/S = 45.61 (APIGraph 25, AndroZoo 372,
#            Hypercube 89, LAMDA 100), so TMCore's integer s = round(width/S) is 46 on all four corpora; every
#            Julia arm in both. Width is not varied: shap's effective feature count M_eff (features that vary
#            in the background) is recorded per model-month, and no claim about width is made.
#            Construction, identical on all corpora:
#              pool: 30,000 rows at 10% malware, stratified, fixed seed, from the corpus's pool period; it is
#                the fixed models' B0 set;
#              month: rows (LAMDA: release train and test portions merged) split in half by a fixed seed,
#                stratified by label;
#              draws: background and explained rows 100 each, at 10% malware, from the train and test halves;
#              skip: a month that cannot supply its draws (10 malware and 90 goodware in each half; 20 and 180
#                for arm 2r; 100 and 900 for 1n/3n) is skipped for that arm and counted; every other month
#                yields a B0 set, and with it U and O, for every refit;
#              guard: each model-month records its AUC on the test half and the support m of its exact
#                attribution; a model-month with AUC < 0.6 is reported and excluded from deciding; a pair
#                with m < k on either side is excluded at that k.
#
# Model, unchanged from b2-drift: flat FPTM, 20 clauses per class, T 10, L 64, LF 10, 30 epochs,
# LiteralCapped(), training parallel mode :none, attribution threaded bit-identically, tm-lab 43dba5f,
# importance = mean of abs(phi) over explained rows. Every results line carries corpus, configuration,
# training condition, N, malware share, repeat factor, pool seed, refit seed, tm-lab commit, ceiling policy
# and parallel mode. The tm-lab checkout is checked against 43dba5f at start-up.
# Resumable: the units are the pool models (configuration, condition, pool seed) and the month units
# (configuration, condition, month). Each output is written to a temporary file, renamed, and followed by
# a .sha256 sidecar; a unit whose outputs all verify is loaded, not recomputed. Before it starts, the run
# waits until available memory covers 1.25 x the measured peak of its job class.
# Run order, each corpus complete before the next: LAMDA, APIGraph, AndroZoo, Hypercube; Python arms after
# the Julia arms of the same corpus.
#
#   julia --project=. -t 16 research/b2-corpora/run.jl <lamda|apigraph|androzoo|hypercube> [nmonths] [nseeds]

# This script runs the Julia arms (2, 2r, 3, 1, 3n, 1n, 5) and exports, for arms.py, the row draws of
# every month, each condition's training rows, the fixed models, and the exact per-row attributions of
# the three arm-4 months. arms.py runs arms 2s, 4 and 0; decide.py applies the statistics and the
# conditions above.
#
# Arguments: nmonths limits the series to its first n calendar months and sends every output to smoke/
# (and the exports to data/b2-corpora-smoke/). nseeds sets the refit seeds (default 3); the pool seeds
# are min(nseeds, 2). Training parallel mode is :none for every model; independent models (the refit
# seeds and conditions of one month) train concurrently on separate threads, which leaves each model's
# arithmetic and random stream unchanged. Seed s (pool seeds 101, 102; refit seeds 1-3; 1n/3n seeds
# 11-13) trains model s in every condition and draws that model's U and O resample, so B0, U and O
# differ only in their rows.

include(joinpath(@__DIR__, "..", "..", "julia", "tmx.jl"))
include(joinpath(@__DIR__, "..", "..", "julia", "shapley.jl"))
using .Tmx: read_inputs, read_meta, read_features
using .Shapley: shapley_mean, clause_literals
using TMCore
using Random, Printf, Statistics, JSON, SHA, Dates, Base.Threads

const CLAUSES, T, L, LF = 20, 10, 64, 10
const EPOCHS = 30
const PARALLEL = :none
const CI = 2                         # class index of `true` (malware) in [false, true]
const NBG, NEX = 100, 100
const MAL_BG, MAL_EX = 10, 10        # 10% malware in every draw
const POOL_N, POOL_MAL = 30_000, 3_000
const SMALL_N, SMALL_MAL = 1_000, 100
const KS = (100, 1000)
const AUC_MIN = 0.6
const TINY = 1e-12                   # an importance at or below this counts as zero for the support m
const TM_LAB_COMMIT = "43dba5f"
const CEILING = "LiteralCapped"
const SEED_POOL_ROWS = 20_261_112
const SEED_MONTH = 7_000_000
const SEED_RESAMPLE = 9_000_000
const NPOOL_MAX = 2
pool_seed(p) = 100 + p               # training seeds: pool models 101, 102; refits 1..3; 1n/3n 11..13
small_seed(s) = 10 + s

"The training conditions, in the order they are reported."
const CONDS = ("B0", "U", "O")
const PLANTED_TOP = 20                # positive control P: features set to 0 in explained and background rows

"The random stream that draws seed s's U or O resample of one month's B0 set (y = m = 0 for the pool)."
resample_rng(seed, y=0, m=0) = MersenneTwister(SEED_RESAMPLE + 100_000seed + 100y + m)

"The tm-lab checkout TMCore was loaded from must be the pinned commit, with no modified tracked file."
function check_tm_lab()
    dir = normpath(joinpath(dirname(pathof(TMCore)), "..", "..", ".."))
    head = try
        readchomp(`git -C $dir rev-parse HEAD`)
    catch
        ""
    end
    startswith(head, TM_LAB_COMMIT) ||
        error("tm-lab at $dir is $(isempty(head) ? "not a git checkout" : head), expected $TM_LAB_COMMIT")
    dirty = readchomp(`git -C $dir status --porcelain --untracked-files=no`)
    isempty(dirty) || error("tm-lab at $dir has modified tracked files:\n$dirty")
    return head
end

struct Corpus
    name::String
    dir::String
    s_matched::Int
    months::Vector{Tuple{Int,Int}}
    pool_stems::Vector{String}
    yearfiles::Bool                  # LAMDA: one file per year with a month column
end

const CORPORA = Dict(
    "lamda" => Corpus("lamda", joinpath("data", "lamda", "tmx"), 100,
                      [(y, m) for y in 2016:2022 for m in 1:12], ["2013", "2014"], true),
    "apigraph" => Corpus("apigraph", joinpath("data", "apigraph", "tmx", "apigraph"), 25,
                         [(y, m) for y in 2013:2018 for m in 1:12], ["train"], false),
    "androzoo" => Corpus("androzoo", joinpath("data", "apigraph", "tmx", "androzoo"), 372,
                         [(y, m) for y in 2020:2021 for m in 1:12], ["train"], false),
    "hypercube" => Corpus("hypercube", joinpath("data", "hypercube", "tmx"), 89,
                          [(y, m) for y in 2022:2023 for m in 1:12], ["train"], false),
)

# ---------------------------------------------------------------------------------------------
# rows

const META = Dict{String,Any}()
meta_of(c::Corpus, stem) = get!(() -> read_meta(joinpath(c.dir, "$stem.meta.arrow")), META,
                                joinpath(c.dir, stem))
islabel(meta, i) = Bool(meta.label[i])

"Malware and goodware row indices of one calendar month, and the file they are in."
function month_rows(c::Corpus, y, m)
    stem = c.yearfiles ? string(y) : @sprintf("%d-%02d", y, m)
    isfile(joinpath(c.dir, "$stem.tmx")) || return stem, Int[], Int[]
    meta = meta_of(c, stem)
    idx = c.yearfiles ?
          [i for i in eachindex(meta.label)
           if isequal(meta.month[i], m) && isequal(meta.year[i], y)] :
          collect(eachindex(meta.label))
    return stem, [i for i in idx if islabel(meta, i)], [i for i in idx if !islabel(meta, i)]
end

struct Month
    year::Int
    month::Int
    stem::String
    tr_mal::Vector{Int}              # train half, shuffled by the month's seed
    tr_good::Vector{Int}
    te_mal::Vector{Int}              # test half
    te_good::Vector{Int}
    base::Bool                       # 10 / 90 in each half: arms 2, 3, 1, 5, 2s, 0, 4
    twice::Bool                      # 20 / 180 in each half: arm 2r
    small::Bool                      # base, and 100 / 900 in the train half: arms 1n, 3n
end

mlabel(M::Month) = @sprintf("%d-%02d", M.year, M.month)
have(a, b, nm, ng) = length(a) >= nm && length(b) >= ng

function build_month(c::Corpus, y, m)
    stem, mal, good = month_rows(c, y, m)
    rng = MersenneTwister(SEED_MONTH + 100y + m)
    shuffle!(rng, mal)
    shuffle!(rng, good)
    hm, hg = length(mal) ÷ 2, length(good) ÷ 2
    trm, tem = mal[1:hm], mal[(hm + 1):end]
    trg, teg = good[1:hg], good[(hg + 1):end]
    base = have(trm, trg, MAL_BG, NBG - MAL_BG) && have(tem, teg, MAL_EX, NEX - MAL_EX)
    twice = have(trm, trg, 2MAL_BG, 2(NBG - MAL_BG)) && have(tem, teg, 2MAL_EX, 2(NEX - MAL_EX))
    small = base && have(trm, trg, SMALL_MAL, SMALL_N - SMALL_MAL)
    return Month(y, m, stem, trm, trg, tem, teg, base, twice, small)
end

# The halves are random permutations, so a prefix is a random draw. Draw j = 1 is the main draw used by
# every arm; draw j = 2 is disjoint from it and used only by arm 2r.
bg_rows(M, j) = vcat(M.tr_mal[((j - 1) * MAL_BG + 1):(j * MAL_BG)],
                     M.tr_good[((j - 1) * (NBG - MAL_BG) + 1):(j * (NBG - MAL_BG))])
ex_rows(M, j) = vcat(M.te_mal[((j - 1) * MAL_EX + 1):(j * MAL_EX)],
                     M.te_good[((j - 1) * (NEX - MAL_EX) + 1):(j * (NEX - MAL_EX))])
function full_rows(M)            # the whole test half, subsampled to exactly 10% malware
    nm = min(length(M.te_mal), length(M.te_good) ÷ 9)
    return vcat(M.te_mal[1:nm], M.te_good[1:(9nm)])
end

small_rows(M) = (M.tr_mal[1:SMALL_MAL], M.tr_good[1:(SMALL_N - SMALL_MAL)])

"""
B0 of a train half: cut to 10% malware by undersampling whichever class is in excess, malware =
min(M, floor(G/9)) and goodware = 9 x that. The lists are already in random order, so the cut takes
prefixes.
"""
function b0_cut(mal::Vector, good::Vector)
    k = min(length(mal), length(good) ÷ 9)
    return mal[1:k], good[1:(9k)]
end

"""
One condition's training rows from a B0 set. Majority and minority are decided per set. U keeps the
minority and draws as many majority rows at random without replacement. O keeps the majority and repeats
every minority row floor(R) times, R = majority / minority, then adds round((R - floor R) x minority)
minority rows drawn without replacement. Returns the rows, the repeat factor and the extra row count.
"""
function resample(cond::String, mal::Vector{T}, good::Vector{T}, rng) where {T}
    cond == "B0" && return (mal=mal, good=good, rep=1, extra=0, minority=:none)
    malmin = length(mal) <= length(good)
    mn, mj = malmin ? (mal, good) : (good, mal)
    if cond == "U"
        mj2 = mj[sort(randperm(rng, length(mj))[1:length(mn)])]
        mn2, rep, extra = mn, 1, 0
    elseif cond == "O"
        R = length(mj) / length(mn)
        rep = floor(Int, R)
        extra = round(Int, (R - rep) * length(mn))
        mn2 = vcat(repeat(mn, rep), mn[sort(randperm(rng, length(mn))[1:extra])])
        mj2 = mj
    else
        error("unknown condition $cond")
    end
    who = malmin ? :malware : :goodware
    return malmin ? (mal=mn2, good=mj2, rep=rep, extra=extra, minority=who) :
                    (mal=mj2, good=mn2, rep=rep, extra=extra, minority=who)
end

ninfo(r) = (N=length(r.mal) + length(r.good), share=length(r.mal) / (length(r.mal) + length(r.good)),
            rep=r.rep, extra=r.extra)

"Every row of the month, read once; `rowsof(rows)` maps file row indices to inputs."
function load_month(c::Corpus, M::Month)
    all = sort(vcat(M.tr_mal, M.tr_good, M.te_mal, M.te_good))
    X, _ = read_inputs(joinpath(c.dir, "$(M.stem).tmx"); rows=all)
    pos = Dict(r => k for (k, r) in enumerate(all))
    meta = meta_of(c, M.stem)
    rowsof(rows) = TMInput[X[pos[r]] for r in rows]
    lab(rows) = Bool[islabel(meta, r) for r in rows]
    return rowsof, lab
end

"The 30,000-row pool as malware and goodware inputs, each in the pool's random order."
function load_pool(c::Corpus)
    refs = Tuple{String,Int,Bool}[]
    for stem in c.pool_stems
        meta = meta_of(c, stem)
        for i in eachindex(meta.label)
            push!(refs, (stem, i, islabel(meta, i)))
        end
    end
    rng = MersenneTwister(SEED_POOL_ROWS)
    mal = shuffle(rng, [r for r in refs if r[3]])
    good = shuffle(rng, [r for r in refs if !r[3]])
    have(mal, good, POOL_MAL, POOL_N - POOL_MAL) ||
        error("$(c.name): pool has $(length(mal)) malware and $(length(good)) goodware")
    mal, good = mal[1:POOL_MAL], good[1:(POOL_N - POOL_MAL)]
    D = Dict{Tuple{String,Int},TMInput}()
    for stem in c.pool_stems
        rows = sort([r[2] for r in vcat(mal, good) if r[1] == stem])
        isempty(rows) && continue
        xs, _ = read_inputs(joinpath(c.dir, "$stem.tmx"); rows=rows)
        for (k, r) in enumerate(rows)
            D[(stem, r)] = xs[k]
        end
    end
    return TMInput[D[(r[1], r[2])] for r in mal], TMInput[D[(r[1], r[2])] for r in good]
end

# ---------------------------------------------------------------------------------------------
# model, attribution, guards

function train_model(X, Y, width, S, seed)
    m = TMClassifier([false, true], width; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    m.ceiling isa TMCore.LiteralCapped || error("ceiling policy is not LiteralCapped")
    for e in 1:EPOCHS
        train!(m, X, Y; rng=MersenneTwister(1000seed + e), parallel=PARALLEL)
    end
    return m
end

train_rows(r, width, S, seed) =
    train_model(vcat(r.mal, r.good), vcat(fill(true, length(r.mal)), fill(false, length(r.good))),
                width, S, seed)

margin(m, x) = Float64(score(m, 2, x) - score(m, 1, x))

"Rank AUC with midranks for ties; malware is the positive class."
function auc_of(s::Vector{Float64}, y::Vector{Bool})
    n = length(s)
    o = sortperm(s)
    r = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && s[o[j + 1]] == s[o[i]]
            j += 1
        end
        for t in i:j
            r[o[t]] = (i + j) / 2
        end
        i = j + 1
    end
    np = count(y)
    nn = n - np
    (np == 0 || nn == 0) && return NaN
    return (sum(r[y]) - np * (np + 1) / 2) / (np * nn)
end

auc(m, X, Y) = auc_of([margin(m, x) for x in X], Y)

"""
Importance = mean over explained rows of abs(exact phi) against the background set. Rows are cut into
blocks whose size depends only on the row count; each block sums its rows in order and the blocks are
added in order, so the result is bit-identical at any thread count. With `keep`, the signed per-row
phi is returned as a width x rows matrix.
"""
function attribute(model, ex::Vector{TMInput}, bg::Vector{TMInput}, width; keep::Bool=false)
    n = length(ex)
    B = n <= 2000 ? 4 : 64
    nb = cld(n, B)
    parts = [zeros(Float64, width) for _ in 1:nb]
    rows = keep ? Matrix{Float64}(undef, width, n) : Matrix{Float64}(undef, 0, 0)
    @threads :dynamic for b in 1:nb
        p = parts[b]
        for r in ((b - 1) * B + 1):min(b * B, n)
            phi = shapley_mean(model, CI, ex[r], bg)
            @inbounds for i in 1:width
                p[i] += abs(phi[i])
            end
            keep && (rows[:, r] .= phi)
        end
    end
    tot = zeros(Float64, width)
    for b in 1:nb
        tot .+= parts[b]
    end
    return tot ./ n, rows
end

"The same quantity on one thread, for the cost comparison against KernelExplainer on the same rows."
function attribute_serial(model, ex::Vector{TMInput}, bg::Vector{TMInput}, width)
    tot = zeros(Float64, width)
    for x in ex
        phi = shapley_mean(model, CI, x, bg)
        tot .+= abs.(phi)
    end
    return tot ./ length(ex)
end

support(v) = count(>(TINY), v)

"""
Features that vary in the background (M_bg), and shap's effective count M_eff: the mean over explained
rows of the features on which the row differs from at least one background row, which is the M that
KernelExplainer regresses on and puts into its default budget 2*M + 2048.
"""
function meff(bg::Vector{TMInput}, ex::Vector{TMInput}, width)
    nw = length(bg[1].chunks)
    OR = zeros(UInt64, nw)
    AND = fill(typemax(UInt64), nw)
    for b in bg, w in 1:nw
        OR[w] |= b.chunks[w]
        AND[w] &= b.chunks[w]
    end
    r = width % 64
    valid = [w < nw || r == 0 ? typemax(UInt64) : (UInt64(1) << r) - 1 for w in 1:nw]
    AND .&= valid
    mbg = sum(count_ones(OR[w] & ~AND[w]) for w in 1:nw)
    mshap = mean(sum(count_ones((OR[w] & ~AND[w]) | (AND[w] & ~x.chunks[w]) |
                                (~OR[w] & x.chunks[w] & valid[w])) for w in 1:nw) for x in ex)
    return mbg, mshap
end

function median_literals(m)
    lits = Int[]
    for bank in (m.positive[CI], m.negative[CI]), j in 1:bank.nclauses
        n = length(clause_literals(bank, j))
        n > 0 && push!(lits, n)
    end
    return isempty(lits) ? 0.0 : median(lits)
end

"The explained class's clauses, with the ceiling the model uses, for the Python scorer in arms.py."
function clause_export(m)
    out = Any[]
    for (bank, sgn) in ((m.positive[CI], 1), (m.negative[CI], -1)), j in 1:bank.nclauses
        lits = clause_literals(bank, j)
        c = Int(bank.count[j])
        push!(out, Dict("sign" => sgn, "ceiling" => (c == 0 ? LF : min(c, LF)),
                        "features" => [l[1] for l in lits], "negated" => [l[2] for l in lits]))
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# pair metrics

topk(imp, k) = partialsortperm(imp, 1:min(k, length(imp)); rev=true)

"Kendall tau-b over the union's importance values, then (1 - tau)/2, as the reference computes it."
function kendall_dist(a, b, imp_a, imp_b)
    u = collect(union(Set(a), Set(b)))
    n = length(u)
    n < 2 && return 1.0
    xa = [imp_a[i] for i in u]
    xb = [imp_b[i] for i in u]
    C = 0; D = 0; ta = 0; tb = 0
    for i in 1:(n - 1), j in (i + 1):n
        da = xa[i] - xa[j]
        db = xb[i] - xb[j]
        if da == 0 && db == 0
            ta += 1; tb += 1
        elseif da == 0
            ta += 1
        elseif db == 0
            tb += 1
        else
            sign(da) == sign(db) ? (C += 1) : (D += 1)
        end
    end
    n0 = n * (n - 1) ÷ 2
    den = sqrt(max(0.0, n0 - ta) * max(0.0, n0 - tb))
    den == 0 && return 1.0
    return (1 - (C - D) / den) / 2
end

"""
Jaccard distance of the two top-k sets and its chance correction. For top-k sets drawn uniformly from
supports A and B, a feature in both supports is in both sets with probability (k/|A|)(k/|B|), so the
expected intersection is E = k^2 |A n B| / (|A| |B|), and c = (|intersection| - E) / (k - E).
"""
function pairstats(a::Vector{Float64}, b::Vector{Float64}, k::Int)
    sa = a .> TINY
    sb = b .> TINY
    ma, mb, mab = count(sa), count(sb), count(sa .& sb)
    ta, tb = topk(a, k), topk(b, k)
    inter = length(intersect(Set(ta), Set(tb)))
    jac = 1 - inter / length(union(Set(ta), Set(tb)))
    E = (ma > 0 && mb > 0) ? k^2 * mab / (ma * mb) : NaN
    c = (isnan(E) || k - E < 1e-9) ? NaN : (inter - E) / (k - E)
    return (jaccard=jac, kendall=kendall_dist(ta, tb, a, b), inter=inter, ma=ma, mb=mb, mab=mab,
            E=E, c=c, support_ok=(ma >= k && mb >= k))
end

# ---------------------------------------------------------------------------------------------

"Wall-clock seconds per step, configuration and condition."
mutable struct Timer
    t::Dict{Tuple{String,String,String},Float64}
    n::Dict{Tuple{String,String,String},Int}
end
Timer() = Timer(Dict{Tuple{String,String,String},Float64}(), Dict{Tuple{String,String,String},Int}())
function tick!(tm::Timer, key, cfg, cond, dt)
    k = (key, cfg, cond)
    tm.t[k] = get(tm.t, k, 0.0) + dt
    tm.n[k] = get(tm.n, k, 0) + 1
end

const PROV = "tm_lab=$TM_LAB_COMMIT ceiling=$CEILING parallel=$PARALLEL"

# ---------------------------------------------------------------------------------------------
# resumable outputs and the memory check

"Peak resident memory, in MB, of one whole run.jl process per corpus, measured on the smoke of
2026-11-13 (16 threads, 3 months) and doubled for the full series. The value used is the larger of this
and every peak recorded since in peak-ram.jsonl beside the exports."
const PEAK_MB_DEFAULT = Dict("julia-lamda" => 1800.0, "julia-apigraph" => 1700.0,
                             "julia-androzoo" => 2500.0, "julia-hypercube" => 1800.0)
const MEM_FACTOR = 1.25
const MEM_RECHECK_S = 30

sha_hex(path) = bytes2hex(open(sha256, path))
sidecar(path) = path * ".sha256"

"Write `path` through a temporary file and a rename, then its .sha256 sidecar the same way."
function write_atomic(f::Function, path::AbstractString)
    mkpath(dirname(path))
    tmp = path * ".tmp"
    open(f, tmp, "w")
    mv(tmp, path; force=true)
    open(io -> println(io, sha_hex(path), "  ", basename(path)), sidecar(path) * ".tmp", "w")
    mv(sidecar(path) * ".tmp", sidecar(path); force=true)
    return path
end

"True when `path` and its sidecar exist and the sidecar's hash is the file's hash."
function verified(path::AbstractString)
    (isfile(path) && isfile(sidecar(path))) || return false
    want = first(split(readline(sidecar(path))))
    return want == sha_hex(path)
end

function mem_available_mb()
    if Sys.islinux() && isfile("/proc/meminfo")
        for l in eachline("/proc/meminfo")
            startswith(l, "MemAvailable:") && return parse(Int, split(l)[2]) / 1024
        end
    end
    return Sys.free_memory() / 2^20
end

function peak_mb(class::String, dirs)
    best = get(PEAK_MB_DEFAULT, class, 0.0)
    for d in dirs
        f = joinpath(d, "peak-ram.jsonl")
        isfile(f) || continue
        for l in eachline(f)
            r = JSON.parse(l)
            r["class"] == class && (best = max(best, Float64(r["peak_mb"])))
        end
    end
    return best
end

"Wait until available memory covers MEM_FACTOR x the class's measured peak."
function wait_for_memory(class::String, need::Float64)
    while true
        avail = mem_available_mb()
        avail >= MEM_FACTOR * need && return avail
        @printf("memory: %s needs %.0f MB x %.2f, %.0f MB available; re-checking in %d s\n",
                class, need, MEM_FACTOR, avail, MEM_RECHECK_S)
        flush(stdout)
        sleep(MEM_RECHECK_S)
    end
end

function record_peak(class::String, dir::AbstractString, smoke::Bool)
    mkpath(dir)
    open(joinpath(dir, "peak-ram.jsonl"), "a") do io
        JSON.print(io, Dict("class" => class, "peak_mb" => Sys.maxrss() / 2^20, "smoke" => smoke,
                            "threads" => nthreads(), "host" => gethostname(), "date" => string(now())))
        println(io)
    end
end

"Rows with the features `top` set to 0 (positive control P: applied to explained and background rows)."
function zeroed(rows::Vector{TMInput}, top)
    out = TMInput[]
    for x in rows
        b = collect(Bool, x)
        b[top] .= false
        push!(out, TMInput(b))
    end
    return out
end

function main()
    isempty(ARGS) && error("usage: run.jl <lamda|apigraph|androzoo|hypercube> [nmonths] [nseeds]")
    head = check_tm_lab()
    c = CORPORA[ARGS[1]]
    smoke = length(ARGS) >= 2
    nmonths = smoke ? parse(Int, ARGS[2]) : typemax(Int)
    nrefit = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
    npool = min(nrefit, NPOOL_MAX)
    outdir = smoke ? joinpath(@__DIR__, "smoke") : @__DIR__
    datadir = joinpath("data", smoke ? "b2-corpora-smoke" : "b2-corpora", c.name)
    peakdirs = [joinpath("data", "b2-corpora"), joinpath("data", "b2-corpora-smoke")]
    class = "julia-$(c.name)"
    need = peak_mb(class, peakdirs)
    avail = wait_for_memory(class, need)
    mkpath(outdir)
    mkpath(datadir)

    width = length(read_features(joinpath(c.dir, "features.arrow")))
    Ss = unique([c.s_matched, 100])                       # matched first; LAMDA has one
    cfgname(S) = "S$S"
    matched(S) = S == c.s_matched

    months = [build_month(c, y, m) for (y, m) in c.months[1:min(nmonths, length(c.months))]]
    usable = [i for i in eachindex(months) if months[i].base]
    arm4 = isempty(usable) ? Int[] :
           unique([usable[1], usable[cld(length(usable), 2)], usable[end]])
    planted = isempty(arm4) ? 0 : arm4[1]

    @printf("b2-corpora  corpus %s  width %d  months %s..%s (%d calendar, %d usable)\n",
            c.name, width, mlabel(months[1]), mlabel(months[end]), length(months), length(usable))
    @printf("configurations %s (matched S=%d, integer s=round(width/S)=%d)  pool seeds %d  refit seeds %d\n",
            join(cfgname.(Ss), ", "), c.s_matched, round(Int, width / c.s_matched), npool, nrefit)
    @printf("training conditions %s  tm-lab checkout %s\n", join(CONDS, ", "), head)
    @printf("model: clauses=%d T=%d L=%d LF=%d epochs=%d  %s  attribution threads=%d\n",
            CLAUSES, T, L, LF, EPOCHS, PROV, nthreads())
    @printf("memory: class %s peak %.0f MB, %.0f MB available at start\n", class, need, avail)
    @printf("skips: base %d, 2r %d, 1n/3n %d of %d months; arm-4 months %s; planted control at %s\n\n",
            count(M -> !M.base, months), count(M -> !M.twice, months),
            count(M -> !M.small, months), length(months), join(mlabel.(months[arm4]), ", "),
            planted == 0 ? "-" : mlabel(months[planted]))
    flush(stdout)

    tm = Timer()
    nunits = Dict("pool computed" => 0, "pool loaded" => 0, "month computed" => 0, "month loaded" => 0)

    # --- pool models: every condition, configuration and pool seed; a verified unit is loaded ---
    pooldir(S, n, p) = joinpath(datadir, cfgname(S), n, "pool$p")
    pool_done(S, n, p) = verified(joinpath(pooldir(S, n, p), "model.json")) &&
                         verified(joinpath(pooldir(S, n, p), "model.tm"))
    fixed = Dict{Tuple{Int,String,Int},Any}()
    pinfo = Dict{Tuple{String,Int},Any}()
    todo_pool = [(S, n, p) for S in Ss for n in CONDS for p in 1:npool if !pool_done(S, n, p)]
    Pmal, Pgood = load_pool(c)                             # the pool is the fixed models' B0 set
    @printf("pool: %d malware, %d goodware; pool units to compute %d of %d\n", length(Pmal), length(Pgood),
            length(todo_pool), length(Ss) * length(CONDS) * npool)
    prow = Dict((n, p) => resample(n, Pmal, Pgood, resample_rng(pool_seed(p))) for n in CONDS, p in 1:npool)
    for (key, r) in prow
        pinfo[key] = ninfo(r)
    end
    check_rows = Pmal[1:20]
    tasks = Dict{Tuple{Int,String,Int},Task}()
    for (S, n, p) in todo_pool
        r = prow[(n, p)]
        tasks[(S, n, p)] = Threads.@spawn begin
            t0 = time(); m = train_rows(r, width, S, pool_seed(p)); (m, time() - t0)
        end
    end
    for S in Ss, n in CONDS, p in 1:npool
        d = pooldir(S, n, p)
        if haskey(tasks, (S, n, p))
            m, dt = fetch(tasks[(S, n, p)])
            tick!(tm, "train_pool", cfgname(S), n, dt)
            info = pinfo[(n, p)]
            write_atomic(io -> save_model(io, m; include_states=true), joinpath(d, "model.tm"))
            write_atomic(joinpath(d, "model.json")) do io
                JSON.print(io, Dict("corpus" => c.name, "cfg" => cfgname(S), "S" => S,
                                    "matched" => matched(S), "condition" => n, "N" => info.N,
                                    "share" => info.share, "repeat" => info.rep, "extra" => info.extra,
                                    "pool_seed" => pool_seed(p), "width" => width, "LF" => LF,
                                    "class_index" => CI, "tm_lab" => head, "train_seconds" => dt,
                                    "check_scores" => [Float64(score(m, CI, x)) for x in check_rows],
                                    "clauses" => clause_export(m)))
            end
            nunits["pool computed"] += 1
            @printf("  %s %-2s pool seed %d: computed, %d malware + %d goodware rows (repeat %d), %.1fs\n",
                    cfgname(S), n, pool_seed(p), round(Int, info.N * info.share),
                    round(Int, info.N * (1 - info.share)), info.rep, dt)
        else
            m = open(load_model, joinpath(d, "model.tm"))
            mj = JSON.parsefile(joinpath(d, "model.json"))
            [Float64(score(m, CI, x)) for x in check_rows] == Float64.(mj["check_scores"]) ||
                error("$(cfgname(S)) $n pool $p: the loaded model does not reproduce its recorded scores")
            tick!(tm, "train_pool", cfgname(S), n, Float64(mj["train_seconds"]))
            nunits["pool loaded"] += 1
            @printf("  %s %-2s pool seed %d: loaded (verified)\n", cfgname(S), n, pool_seed(p))
        end
        fixed[(S, n, p)] = m
    end
    Pmal = Pgood = prow = nothing
    println()
    flush(stdout)

    # --- month units (configuration, condition, month) ---
    K(S, cond, arm, seed, i) = (S, cond, arm, seed, i)
    imp = Dict{Tuple{Int,String,String,Int,Int},Vector{Float64}}()
    mrows = Any[]
    monthjson = Any[]
    unitdir(S, n) = joinpath(datadir, "units", cfgname(S), n)
    unit_files(S, n, M, i) = vcat([joinpath(unitdir(S, n), "$(mlabel(M)).json"),
                                   joinpath(unitdir(S, n), "$(mlabel(M)).imp.bin")],
                                  [joinpath(pooldir(S, n, p), "month-$(mlabel(M)).json") for p in 1:npool],
                                  i in arm4 ? [joinpath(pooldir(S, n, p), "phi-$(mlabel(M)).bin")
                                               for p in 1:npool] : String[])
    unit_done(S, n, M, i) = all(verified, unit_files(S, n, M, i))

    for (i, M) in enumerate(months)
        rec = Dict{String,Any}("label" => mlabel(M), "year" => M.year, "month" => M.month,
                               "stem" => M.stem, "base" => M.base, "twice" => M.twice,
                               "small" => M.small, "arm4" => (i in arm4), "planted" => (i == planted),
                               "n" => [length(M.tr_mal), length(M.tr_good),
                                       length(M.te_mal), length(M.te_good)])
        if !M.base
            push!(monthjson, rec)
            @printf("  %s  skipped: train half %d/%d, test half %d/%d (malware/goodware)\n",
                    mlabel(M), length(M.tr_mal), length(M.tr_good), length(M.te_mal), length(M.te_good))
            continue
        end
        t0 = time()
        meta = meta_of(c, M.stem)
        lab(rows) = Bool[islabel(meta, r) for r in rows]
        terows = vcat(M.te_mal, M.te_good)
        rec["bg"] = bg_rows(M, 1); rec["ex"] = ex_rows(M, 1)
        rec["bg_label"] = lab(bg_rows(M, 1)); rec["ex_label"] = lab(ex_rows(M, 1))
        rec["test"] = terows; rec["test_label"] = lab(terows)

        # Training rows, as file row indices: the month's B0 cut, and per seed its U and O resample.
        # The arm-0 MLP of run r uses seed r + 1's rows, so arms.py receives them as exported.
        bmal, bgood = b0_cut(M.tr_mal, M.tr_good)
        smal, sgood = M.small ? small_rows(M) : (Int[], Int[])
        rrows = Dict{Tuple{String,Int},Any}()
        srows = Dict{Tuple{String,Int},Any}()
        rec["train"] = Dict{String,Any}("B0" => Dict("mal" => bmal, "good" => bgood))
        for n in CONDS, s in 1:nrefit
            rrows[(n, s)] = resample(n, bmal, bgood, resample_rng(s, M.year, M.month))
            M.small && (srows[(n, s)] = resample(n, smal, sgood, resample_rng(small_seed(s), M.year, M.month)))
            r = rrows[(n, s)]
            if n == "U"
                rec["train"]["U/$s"] = Dict("minority" => string(r.minority),
                                            "majority_rows" => (r.minority == :malware ? r.good : r.mal))
            elseif n == "O"
                mn2, nmn = r.minority == :malware ? (r.mal, length(bmal)) : (r.good, length(bgood))
                rec["train"]["O/$s"] = Dict("minority" => string(r.minority), "repeat" => r.rep,
                                            "extra_rows" => mn2[(r.rep * nmn + 1):end])
            end
        end

        todo = [(S, n) for S in Ss for n in CONDS if !unit_done(S, n, M, i)]
        if !isempty(todo)
            rowsof, _ = load_month(c, M)
            bg, ex = rowsof(bg_rows(M, 1)), rowsof(ex_rows(M, 1))
            Xte = rowsof(terows)
            Yte = rec["test_label"]
            mbg, mshap = meff(bg, ex, width)
            if M.twice
                bg2, ex2 = rowsof(bg_rows(M, 2)), rowsof(ex_rows(M, 2))
            end
            ex5 = rowsof(full_rows(M))

            # Refits of the units still to compute train concurrently, and all of them finish before any
            # attribution starts, so the attribution timings are not shared with training.
            rtasks = Dict{Tuple{String,Int,String,Int},Task}()
            for (S, n) in todo, s in 1:nrefit
                r = rrows[(n, s)]
                Xr = rowsof(vcat(r.mal, r.good))
                Yr = vcat(fill(true, length(r.mal)), fill(false, length(r.good)))
                rtasks[("3", S, n, s)] = Threads.@spawn begin
                    t1 = time(); m = train_model(Xr, Yr, width, S, s); (m, time() - t1)
                end
                if M.small
                    q = srows[(n, s)]
                    Xs = rowsof(vcat(q.mal, q.good))
                    Ys = vcat(fill(true, length(q.mal)), fill(false, length(q.good)))
                    rtasks[("3n", S, n, s)] = Threads.@spawn begin
                        t1 = time(); m = train_model(Xs, Ys, width, S, small_seed(s)); (m, time() - t1)
                    end
                end
            end
            refit = Dict(key => fetch(t) for (key, t) in rtasks)

            for (S, n) in todo
                ut = Dict{String,Vector{Float64}}()            # step => [total seconds, count]
                tk(step, dt) = (v = get!(ut, step, [0.0, 0.0]); v[1] += dt; v[2] += 1)
                vecs = Dict{String,Vector{Float64}}()
                urows = Any[]
                for s in 1:nrefit
                    tk("train_refit", refit[("3", S, n, s)][2])
                    M.small && tk("train_refit_1000", refit[("3n", S, n, s)][2])
                end
                for p in 1:npool
                    m = fixed[(S, n, p)]
                    a = auc(m, Xte, Yte)
                    t1 = time()
                    v, rows = attribute(m, ex, bg, width; keep=(i in arm4))
                    dt2 = time() - t1
                    tk("attr_2", dt2)
                    vecs["2:$p"] = v
                    push!(urows, Dict("model" => "pool", "seed" => pool_seed(p), "auc" => a, "m" => support(v),
                                      "lits" => median_literals(m), "secs" => dt2,
                                      "info" => Dict(pairs(pinfo[(n, p)]))))
                    if M.twice
                        t1 = time(); vecs["2r:$p"], _ = attribute(m, ex2, bg2, width); tk("attr_2r", time() - t1)
                    end
                    t1 = time(); vecs["5:$p"], _ = attribute(m, ex5, bg, width); tk("attr_5", time() - t1)
                    # What arms.py needs on this model-month: the guard, the scores for its port check,
                    # and the exact attribution's wall clock on the rows that arm 2s explains.
                    d = pooldir(S, n, p)
                    mj = Dict{String,Any}("auc" => a, "m" => support(v), "M_eff" => mshap,
                                          "score_bg" => [Float64(score(m, CI, x)) for x in bg],
                                          "score_ex" => [Float64(score(m, CI, x)) for x in ex],
                                          "seconds_exact" => dt2, "threads_exact" => nthreads())
                    if i in arm4
                        t1 = time()
                        vs = attribute_serial(m, ex, bg, width)
                        mj["seconds_exact_1thread"] = time() - t1
                        tk("attr_2_1thread", mj["seconds_exact_1thread"])
                        maximum(abs.(vs .- v)) < 1e-12 || error("serial and threaded attribution differ")
                        mj["exact_importance"] = v
                        write_atomic(io -> write(io, rows), joinpath(d, "phi-$(mlabel(M)).bin"))
                    end
                    write_atomic(io -> JSON.print(io, mj), joinpath(d, "month-$(mlabel(M)).json"))
                end
                for (arm, kind, sd) in (("3", "refit", identity), ("3n", "refit1000", small_seed)), s in 1:nrefit
                    haskey(refit, (arm, S, n, s)) || continue
                    m, _ = refit[(arm, S, n, s)]
                    t1 = time()
                    v, _ = attribute(m, ex, bg, width)
                    dt = time() - t1
                    tk("attr_" * arm, dt)
                    vecs["$arm:$s"] = v
                    info = arm == "3" ? ninfo(rrows[(n, s)]) : ninfo(srows[(n, s)])
                    push!(urows, Dict("model" => kind, "seed" => sd(s), "auc" => auc(m, Xte, Yte),
                                      "m" => support(v), "lits" => median_literals(m), "secs" => dt,
                                      "info" => Dict(pairs(info))))
                end
                # the unit: its vectors, then its record, which is written last and marks it complete
                keys_ = sort(collect(keys(vecs)))
                write_atomic(joinpath(unitdir(S, n), "$(mlabel(M)).imp.bin")) do io
                    for k in keys_
                        write(io, vecs[k])
                    end
                end
                write_atomic(joinpath(unitdir(S, n), "$(mlabel(M)).json")) do io
                    JSON.print(io, Dict("cfg" => cfgname(S), "condition" => n, "month" => mlabel(M),
                                        "width" => width, "keys" => keys_, "rows" => urows, "timing" => ut,
                                        "M_bg" => mbg, "M_eff" => mshap, "tm_lab" => head))
                end
                nunits["month computed"] += 1
            end
        end

        # every unit of this month, computed just now or loaded, from its verified files
        for S in Ss, n in CONDS
            unit_done(S, n, M, i) || error("$(cfgname(S)) $n $(mlabel(M)): unit outputs do not verify")
            u = JSON.parsefile(joinpath(unitdir(S, n), "$(mlabel(M)).json"))
            (S, n) in todo || (nunits["month loaded"] += 1)
            raw = reinterpret(Float64, read(joinpath(unitdir(S, n), "$(mlabel(M)).imp.bin")))
            for (j, k) in enumerate(u["keys"])
                arm, seed = split(k, ":")
                arm == "P" && continue      # the withdrawn plant of earlier units; P is its own unit now
                imp[K(S, n, String(arm), parse(Int, seed), i)] = collect(raw[((j - 1) * width + 1):(j * width)])
            end
            for r in u["rows"]
                info = (N=Int(r["info"]["N"]), share=Float64(r["info"]["share"]), rep=Int(r["info"]["rep"]),
                        extra=Int(r["info"]["extra"]))
                push!(mrows, (cfg=cfgname(S), S=S, cond=n, info=info, model=r["model"], seed=Int(r["seed"]),
                              month=mlabel(M), auc=Float64(r["auc"]), m=Int(r["m"]), lits=Float64(r["lits"]),
                              M_bg=Int(u["M_bg"]), M_eff=Float64(u["M_eff"]), secs=Float64(r["secs"])))
            end
            for (step, v) in u["timing"]
                for _ in 1:Int(v[2])
                    tick!(tm, step, cfgname(S), n, v[1] / v[2])
                end
            end
            rec["M_bg"] = u["M_bg"]; rec["M_eff"] = u["M_eff"]
        end
        push!(monthjson, rec)
        @printf("  %s  halves %d/%d + %d/%d  B0 refit %d/%d  units computed %d, loaded %d  %.1fs\n",
                mlabel(M), length(M.tr_mal), length(M.tr_good), length(M.te_mal), length(M.te_good),
                length(bmal), length(bgood), length(todo), length(Ss) * length(CONDS) - length(todo),
                time() - t0)
        flush(stdout)
        i % 3 == 0 && GC.gc()
    end

    # --- positive control P (null player): its own unit per configuration and condition, from the
    # loaded or just-computed fixed model of pool seed 101 and the unplanted exact importance; no training ---
    plant = Dict{Tuple{Int,String},Any}()
    if planted > 0
        M = months[planted]
        pfiles(S, n) = (joinpath(unitdir(S, n), "P-$(mlabel(M)).json"), joinpath(unitdir(S, n), "P-$(mlabel(M)).imp.bin"))
        ptodo = [(S, n) for S in Ss for n in CONDS if !all(verified, pfiles(S, n))]
        if !isempty(ptodo)
            rowsof, _ = load_month(c, M)
            bg, ex = rowsof(bg_rows(M, 1)), rowsof(ex_rows(M, 1))
        end
        for S in Ss, n in CONDS
            fj, fb = pfiles(S, n)
            if (S, n) in ptodo
                v = imp[K(S, n, "2", 1, planted)]
                top = partialsortperm(v, 1:PLANTED_TOP; rev=true)
                t1 = time()
                vP, rowsP = attribute(fixed[(S, n, 1)], zeroed(ex, top), zeroed(bg, top), width; keep=true)
                dt = time() - t1
                maxabs = maximum(abs.(rowsP[top, :]))
                write_atomic(io -> write(io, vP), fb)
                write_atomic(fj) do io
                    JSON.print(io, Dict("cfg" => cfgname(S), "condition" => n, "month" => mlabel(M),
                                        "pool_seed" => pool_seed(1), "planted" => collect(top),
                                        "max_abs_phi_planted" => maxabs, "support" => support(vP),
                                        "support_unplanted" => support(v), "seconds" => dt, "tm_lab" => head))
                end
            end
            u = JSON.parsefile(fj)
            imp[K(S, n, "P", 1, planted)] = collect(reinterpret(Float64, read(fb)))
            plant[(S, n)] = u
            tick!(tm, "attr_planted", cfgname(S), n, Float64(u["seconds"]))
        end
        write_atomic(joinpath(outdir, "$(c.name)-planted.csv")) do io
            println(io, "corpus,cfg,S,matched,condition,pool_seed,month,max_abs_phi_planted,support_planted,support_unplanted,planted_features,tm_lab,ceiling,parallel")
            for S in Ss, n in CONDS
                u = plant[(S, n)]
                @printf(io, "%s,%s,%d,%s,%s,%d,%s,%.17g,%d,%d,%s,%s,%s,%s\n", c.name, cfgname(S), S, matched(S), n,
                        pool_seed(1), mlabel(M), u["max_abs_phi_planted"], u["support"], u["support_unplanted"],
                        join(u["planted"], ";"), TM_LAB_COMMIT, CEILING, PARALLEL)
            end
        end
        @printf("positive control P at %s: %d units computed, %d loaded; max |phi| of planted features %s\n",
                mlabel(M), length(ptodo), length(Ss) * length(CONDS) - length(ptodo),
                join(sort(unique([string(u["max_abs_phi_planted"]) for u in values(plant)])), ", "))
    end

    write_atomic(joinpath(datadir, "months.json")) do io
        JSON.print(io, Dict("corpus" => c.name, "width" => width, "dir" => c.dir,
                            "s_matched" => c.s_matched, "configs" => cfgname.(Ss),
                            "conditions" => collect(CONDS), "tm_lab" => head,
                            "npool" => npool, "nrefit" => nrefit, "months" => monthjson))
    end

    # --- model-month table ---
    auc_of_model = Dict{Tuple{String,String,String,Int,String},Float64}()
    write_atomic(joinpath(outdir, "$(c.name)-months.csv")) do io
        println(io, "corpus,cfg,S,matched,condition,N,share,repeat,extra,model,seed,month,auc,guard_ok,m,median_literals,M_bg,M_eff,seconds_attr,threads,tm_lab,ceiling,parallel")
        for r in mrows
            auc_of_model[(r.cfg, r.cond, r.model, r.seed, r.month)] = r.auc
            @printf(io, "%s,%s,%d,%s,%s,%d,%.4f,%d,%d,%s,%d,%s,%.6f,%s,%d,%.1f,%d,%.1f,%.4f,%d,%s,%s,%s\n",
                    c.name, r.cfg, r.S, matched(r.S), r.cond, r.info.N, r.info.share, r.info.rep,
                    r.info.extra, r.model, r.seed, r.month, r.auc, r.auc >= AUC_MIN, r.m, r.lits, r.M_bg,
                    r.M_eff, r.secs, nthreads(), TM_LAB_COMMIT, CEILING, PARALLEL)
        end
    end
    guard(cfg, cond, model, seed, i) =
        get(auc_of_model, (cfg, cond, model, seed, mlabel(months[i])), NaN) >= AUC_MIN
    Nof = Dict((r.cfg, r.cond, r.model, r.seed, r.month) => r.info for r in mrows)
    ninf(cfg, cond, kind, seed, i) = Nof[(cfg, cond, kind, seed, mlabel(months[i]))]

    # --- pairs ---
    # anchor = the later month of a consecutive pair, or the month itself for a same-month arm
    summary = Dict{Tuple{String,String,String,Int,Int},Vector{Any}}()
    has(key) = haskey(imp, key)
    write_atomic(joinpath(outdir, "$(c.name)-pairs.csv")) do io
        println(io, "corpus,cfg,S,matched,condition,share,repeat,N_a,N_b,arm,pool_seed,refit_seed,k,from,to,anchor,jaccard,kendall,inter,m_a,m_b,m_ab,E,c,guard_ok,support_ok,tm_lab,ceiling,parallel")
        function emit(S, cond, ia_info, ib_info, arm, ps, rs, k, ia, ib, va, vb, g)
            st = pairstats(va, vb, k)
            @printf(io, "%s,%s,%d,%s,%s,%.4f,%d,%d,%d,%s,%s,%s,%d,%s,%s,%s,%.6f,%.6f,%d,%d,%d,%d,%.6f,%.6f,%s,%s,%s,%s,%s\n",
                    c.name, cfgname(S), S, matched(S), cond, ib_info.share, ib_info.rep, ia_info.N, ib_info.N,
                    arm, ps, rs, k, mlabel(months[ia]), mlabel(months[ib]), mlabel(months[ib]), st.jaccard,
                    st.kendall, st.inter, st.ma, st.mb, st.mab, st.E, st.c, g, st.support_ok, TM_LAB_COMMIT,
                    CEILING, PARALLEL)
            if g && st.support_ok
                push!(get!(summary, (cfgname(S), cond, arm, isempty(ps) ? 0 : parse(Int, ps), k), Any[]),
                      (st.jaccard, st.c, ia_info, ib_info, rs))
            end
        end
        for S in Ss, n in CONDS, k in KS, i in eachindex(months)
            cf = cfgname(S)
            for p in 1:npool
                ps = string(pool_seed(p))
                has(K(S, n, "2", p, i)) || continue
                pinf = ninf(cf, n, "pool", pool_seed(p), i)
                gp(j) = guard(cf, n, "pool", pool_seed(p), j)
                if i > 1 && has(K(S, n, "2", p, i - 1))
                    emit(S, n, pinf, pinf, "2", ps, "", k, i - 1, i, imp[K(S, n, "2", p, i - 1)],
                         imp[K(S, n, "2", p, i)], gp(i - 1) && gp(i))
                end
                if i > 1 && has(K(S, n, "5", p, i - 1)) && has(K(S, n, "5", p, i))
                    emit(S, n, pinf, pinf, "5", ps, "", k, i - 1, i, imp[K(S, n, "5", p, i - 1)],
                         imp[K(S, n, "5", p, i)], gp(i - 1) && gp(i))
                end
                if has(K(S, n, "2r", p, i))
                    emit(S, n, pinf, pinf, "2r", ps, "", k, i, i, imp[K(S, n, "2", p, i)],
                         imp[K(S, n, "2r", p, i)], gp(i))
                end
                if has(K(S, n, "P", p, i))
                    emit(S, n, pinf, pinf, "P", ps, "", k, i, i, imp[K(S, n, "2", p, i)],
                         imp[K(S, n, "P", p, i)], gp(i))
                end
            end
            for (arm, farm, kind, sd) in (("3", "1", "refit", identity), ("3n", "1n", "refit1000", small_seed))
                for s in 1:nrefit
                    gs(j) = guard(cf, n, kind, sd(s), j)
                    if i > 1 && has(K(S, n, arm, s, i - 1)) && has(K(S, n, arm, s, i))
                        emit(S, n, ninf(cf, n, kind, sd(s), i - 1), ninf(cf, n, kind, sd(s), i), arm, "",
                             string(sd(s)), k, i - 1, i, imp[K(S, n, arm, s, i - 1)], imp[K(S, n, arm, s, i)],
                             gs(i - 1) && gs(i))
                    end
                end
                for s in 1:(nrefit - 1), s2 in (s + 1):nrefit
                    has(K(S, n, arm, s, i)) && has(K(S, n, arm, s2, i)) || continue
                    emit(S, n, ninf(cf, n, kind, sd(s), i), ninf(cf, n, kind, sd(s2), i), farm, "",
                         "$(sd(s))-$(sd(s2))", k, i, i, imp[K(S, n, arm, s, i)], imp[K(S, n, arm, s2, i)],
                         guard(cf, n, kind, sd(s), i) && guard(cf, n, kind, sd(s2), i))
                end
            end
        end
    end

    # --- timing per model-month, configuration and condition ---
    write_atomic(joinpath(outdir, "$(c.name)-timing.csv")) do io
        println(io, "corpus,step,cfg,condition,count,total_s,mean_s,threads,maxrss_mb,tm_lab,ceiling,parallel")
        for key in sort(collect(keys(tm.t)))
            @printf(io, "%s,%s,%s,%s,%d,%.3f,%.4f,%d,%.0f,%s,%s,%s\n", c.name, key[1], key[2], key[3], tm.n[key],
                    tm.t[key], tm.t[key] / tm.n[key], nthreads(), Sys.maxrss() / 2^20, TM_LAB_COMMIT,
                    CEILING, PARALLEL)
        end
    end

    println("\n", "-"^110)
    @printf("%-6s %-3s %-4s %-5s %6s %9s %9s %5s  %s\n", "cfg", "cnd", "arm", "pool", "k", "jaccard", "c", "n",
            "N range, malware share, repeat")
    println("-"^110)
    for key in sort(collect(keys(summary)))
        cf, cond, arm, ps, k = key
        v = summary[key]
        js = [x[1] for x in v]
        cs = [x[2] for x in v]
        Ns = vcat([x[3].N for x in v], [x[4].N for x in v])
        sh = vcat([x[3].share for x in v], [x[4].share for x in v])
        rp = vcat([x[3].rep for x in v], [x[4].rep for x in v])
        rseeds = join(sort(unique([string(x[5]) for x in v if !isempty(x[5])])), ";")
        @printf("%-6s %-3s %-4s %-5s %6d %9.3f %9.3f %5d  N %d-%d share %.3f-%.3f repeat %d-%d\n", cf, cond, arm,
                ps == 0 ? "-" : string(ps), k, mean(js), mean(cs), length(v), minimum(Ns), maximum(Ns),
                minimum(sh), maximum(sh), minimum(rp), maximum(rp))
        @printf("RESULT corpus=%s cfg=%s condition=%s arm=%s pool_seed=%s refit_seeds=%s k=%d jaccard=%.4f c=%.4f n=%d N=%d-%d share=%.3f-%.3f repeat=%d-%d %s\n",
                c.name, cf, cond, arm, ps == 0 ? "-" : string(ps), isempty(rseeds) ? "-" : rseeds,
                k, mean(js), mean(cs), length(v), minimum(Ns), maximum(Ns), minimum(sh), maximum(sh),
                minimum(rp), maximum(rp), PROV)
    end
    println("\nseconds per model-month (wall clock, ", nthreads(), " threads; loaded units carry their recorded times):")
    for key in sort(collect(keys(tm.t)))
        @printf("  %-18s %-5s %-3s %8.3f s  (n=%d)\n", key[1], key[2], key[3], tm.t[key] / tm.n[key], tm.n[key])
        @printf("RESULT corpus=%s timing=%s cfg=%s condition=%s mean_s=%.4f n=%d threads=%d %s\n",
                c.name, key[1], key[2], key[3], tm.t[key] / tm.n[key], tm.n[key], nthreads(), PROV)
    end
    @printf("units: pool computed %d, pool loaded %d, month computed %d, month loaded %d\n",
            nunits["pool computed"], nunits["pool loaded"], nunits["month computed"], nunits["month loaded"])
    @printf("RESULT corpus=%s units_pool_computed=%d units_pool_loaded=%d units_month_computed=%d units_month_loaded=%d %s\n",
            c.name, nunits["pool computed"], nunits["pool loaded"], nunits["month computed"],
            nunits["month loaded"], PROV)
    record_peak(class, datadir |> dirname, smoke)
    @printf("RESULT corpus=%s peak_rss_mb=%.0f threads=%d %s\n", c.name, Sys.maxrss() / 2^20, nthreads(), PROV)
    println("\nmeans above are over pairs with guard and support both satisfied; intervals, the positive")
    println("control and the conditions F1-F3 and P1-P3 are computed by decide.py")
end

main()
