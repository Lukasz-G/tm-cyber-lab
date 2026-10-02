# Bring the tm-lab packages into vendor/ so this project's Julia environment resolves.
#
#   julia --project=. julia/bootstrap.jl
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'
#
# tm-lab is cloned at a pinned commit, not vendored into this repository's history:
# experiments stay reproducible from a fresh checkout without carrying another project's
# source, and the pin makes "which version of the machine produced this number" answerable.
#
# Set TM_LAB_PATH to a local working copy to develop against uncommitted tm-lab changes.
# Anything measured that way is not reproducible from this repo alone -- say so in the
# experiment's README.

const TM_LAB_URL = "https://github.com/Lukasz-G/tm-lab.git"

# Pinned 2026-09-18 at 43dba5f (threaded inference, `parallel = :classes | :clauses`, and the
# width-aware gate that keeps a wrong mode from costing anything). Bump deliberately, and re-run any
# experiment whose number depends on the machine's behaviour: tm-lab's internals have moved before,
# and the parallel keyword itself changed shape twice in one day before settling here.
const TM_LAB_COMMIT = "43dba5fc6a88a6ec06d7fd30ad78a8caa9cbd7c9"

const VENDOR = joinpath(@__DIR__, "..", "vendor")
const CHECKOUT = joinpath(VENDOR, "tm-lab")

run_git(args...) = run(Cmd(["git", args...]))

function current_commit(dir)
    try
        return strip(read(Cmd(["git", "-C", dir, "rev-parse", "HEAD"]), String))
    catch
        return ""
    end
end

"""
Point `vendor/tm-lab` at a local working copy.

`Project.toml`'s `[sources]` are committed and must not be rewritten per machine, so the override
works by making `vendor/tm-lab` a link and not by editing the environment. A Windows directory
junction needs no privileges, unlike a directory symlink.
"""
function link_local(local_path::AbstractString)
    mkpath(VENDOR)
    if islink(CHECKOUT)
        rm(CHECKOUT; force=true)
    elseif isdir(joinpath(CHECKOUT, ".git"))
        # The pinned clone is gitignored and re-creatable in seconds, so replacing it is cheap --
        # but only if nobody has edited it, because that edit would be the only copy.
        dirty = strip(read(Cmd(["git", "-C", CHECKOUT, "status", "--porcelain"]), String))
        isempty(dirty) ||
            error("$CHECKOUT has uncommitted changes; save them before pointing at TM_LAB_PATH:\n$dirty")
        rm(CHECKOUT; recursive=true, force=true)
    elseif ispath(CHECKOUT)
        error("$CHECKOUT exists and is neither a link nor a git checkout; remove it by hand")
    end
    if Sys.iswindows()
        run(Cmd(["cmd", "/c", "mklink", "/J", CHECKOUT, abspath(local_path)]))
    else
        symlink(abspath(local_path), CHECKOUT; dir_target=true)
    end
    isfile(joinpath(CHECKOUT, "packages", "TMCore", "Project.toml")) ||
        error("link created but $CHECKOUT/packages/TMCore is not there — is TM_LAB_PATH a tm-lab root?")
    return CHECKOUT
end

function bootstrap()
    local_path = get(ENV, "TM_LAB_PATH", "")
    if !isempty(local_path)
        isdir(local_path) || error("TM_LAB_PATH=$local_path is not a directory")
        link_local(local_path)
        @info "vendor/tm-lab now links to a working copy" path = local_path commit = current_commit(local_path)
        @warn "a working copy may hold uncommitted changes, so numbers measured this way are not " *
              "reproducible from this repository alone — say so in the experiment's README"
        println("\nnext: julia --project=. -e 'using Pkg; Pkg.instantiate()'")
        return CHECKOUT
    end

    mkpath(VENDOR)
    if islink(CHECKOUT)
        @info "removing a TM_LAB_PATH link to restore the pinned clone"
        rm(CHECKOUT; force=true)
    end
    if !isdir(joinpath(CHECKOUT, ".git"))
        @info "cloning tm-lab" url = TM_LAB_URL into = CHECKOUT
        run_git("clone", "--quiet", TM_LAB_URL, CHECKOUT)
    end

    if current_commit(CHECKOUT) != TM_LAB_COMMIT
        run_git("-C", CHECKOUT, "fetch", "--quiet", "origin")
        run_git("-C", CHECKOUT, "checkout", "--quiet", TM_LAB_COMMIT)
    end

    got = current_commit(CHECKOUT)
    got == TM_LAB_COMMIT || error("checkout is at $got, expected $TM_LAB_COMMIT")

    for pkg in ("TMCore", "TMBoolean")
        p = joinpath(CHECKOUT, "packages", pkg, "Project.toml")
        isfile(p) || error("$pkg not found at $p -- has tm-lab's layout changed?")
    end

    @info "tm-lab ready" commit = TM_LAB_COMMIT path = CHECKOUT
    println("\nnext: julia --project=. -e 'using Pkg; Pkg.instantiate()'")
    return CHECKOUT
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && bootstrap()
