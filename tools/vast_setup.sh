#!/usr/bin/env bash
# Bootstrap TM-Cyber on a fresh vast.ai CPU instance. Idempotent; safe to re-run, and usable as
# the instance onstart script.
#
#   curl -sL https://raw.githubusercontent.com/<owner>/TM-Cyber/main/tools/vast_setup.sh | bash
#
# Why a script rather than a template: the rented image has no Julia, and the version matters --
# a model serialized under one Julia can fail to load under another, and the Tsetlin stack is
# pinned by commit, so pinning the runtime too is the cheap half of reproducibility.
#
# What this does NOT do: fetch datasets. LAMDA is large and licensed by its authors; the matrices
# this project trains on are derived locally and shipped as .tmx (see docs/matrix-format.md).
# Put the packed matrix on the box yourself, and prefer one Google Drive fetch over scp for
# anything above a few tens of megabytes.
set -euo pipefail

REPO="${TMCYBER_REPO:-https://github.com/Lukasz-G/TM-Cyber.git}"
JULIA_VERSION="${JULIA_VERSION:-1.11.9}"
WORK="${WORK:-/workspace}"

JULIA_DIR="$WORK/julia-$JULIA_VERSION"
JULIA_TARBALL="julia-$JULIA_VERSION-linux-x86_64.tar.gz"
JULIA_URL="https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/$JULIA_TARBALL"

need() { command -v "$1" >/dev/null 2>&1; }

echo "=== packages ==="
if ! need git || ! need curl || ! need python3; then
    apt-get update -qq
    apt-get install -y -qq git curl ca-certificates python3 python3-venv
fi

echo "=== julia $JULIA_VERSION ==="
mkdir -p "$WORK"
if [ ! -x "$JULIA_DIR/bin/julia" ]; then
    curl -fsSL "$JULIA_URL" -o "$WORK/$JULIA_TARBALL"
    tar -xzf "$WORK/$JULIA_TARBALL" -C "$WORK"
    rm -f "$WORK/$JULIA_TARBALL"
fi
export PATH="$JULIA_DIR/bin:$PATH"
julia --version

echo "=== repo ==="
cd "$WORK"
if [ ! -d tm-cyber/.git ]; then
    git clone --quiet "$REPO" tm-cyber
fi
cd tm-cyber
git pull --ff-only --quiet

echo "=== tsetlin stack (pinned commit) ==="
julia --project=. julia/bootstrap.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Precompile ONCE, before anything fans out. Dozens of Julia processes starting together all try
# to precompile the same packages into the same depot and serialise on its locks, which turns a
# 20-second cost into minutes of a box doing nothing. tools/fanout.sh assumes this already ran.
echo "=== precompile ==="
julia --project=. -e 'using TMCore, TMBoolean, Arrow; println("precompiled")'

echo "=== self check: the Python/Julia boundary ==="
python3 -m venv "$WORK/venv" 2>/dev/null || true
# shellcheck disable=SC1091
. "$WORK/venv/bin/activate"
pip install -q numpy pyarrow
FX="$WORK/boundary_fx"
python test/boundary.py write "$FX"
julia --project=. test/boundary.jl "$FX"
python test/boundary.py verify "$FX"

cat <<MSG

setup complete.
  julia      $JULIA_DIR/bin/julia
  repo       $WORK/tm-cyber
  venv       $WORK/venv
  cores      $(nproc) ($(nproc --all) total)
  memory     $(free -g | awk '/^Mem:/ {print $2}') GB

next: put a .tmx matrix under $WORK/tm-cyber/data/, then
      setsid nohup bash tools/fanout.sh <results.log> <parallelism> "name|command" ... \\
             >/dev/null 2>&1 </dev/null &
MSG
