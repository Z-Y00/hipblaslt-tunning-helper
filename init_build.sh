#!/usr/bin/env bash
set -euo pipefail

# Build TensileLite client, install Origami, and cache git deps.
# Run once after cloning / updating submodules.
#
# Prefers the build-tree Tensile (tmp_rebuild) so that tuning uses the
# same Tensile version as TensileCreateLibrary during hipBLASLt build.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GIT_CACHE="$SCRIPT_DIR/.git-cache"

# ── Cache git repos used by cmake FetchContent ───────────────────────────
# Creates local bare mirrors and configures git to use them instead of
# hitting the network.  Survives container restarts if /workspace persists.
setup_git_cache() {
  echo "=== Setting up git mirror cache ==="
  mkdir -p "$GIT_CACHE"

  local repos=(
    "https://github.com/fmtlib/fmt.git"
    "https://github.com/gabime/spdlog.git"
    "https://github.com/catchorg/Catch2.git"
    "https://github.com/jbeder/yaml-cpp.git"
    "https://github.com/ridiculousfish/libdivide.git"
  )

  for url in "${repos[@]}"; do
    local name=$(basename "$url")       # e.g. fmt.git
    local mirror="$GIT_CACHE/$name"
    if [ -d "$mirror" ]; then
      echo "  $name: updating existing mirror"
      git -C "$mirror" fetch --all -q 2>/dev/null || true
    else
      echo "  $name: creating bare mirror"
      git clone --bare "$url" "$mirror"
    fi
    # redirect future clones to local mirror
    git config --global url."file://$mirror".insteadOf "$url"
  done
  echo "  git cache ready at $GIT_CACHE"
}

setup_git_cache

# ── Detect Tensile directory ─────────────────────────────────────────────
_BUILD_TENSILE="$SCRIPT_DIR/tmp_rebuild/rocm-libraries/projects/hipblaslt/tensilelite"
_LOCAL_TENSILE="$SCRIPT_DIR/hipblaslt/tensilelite"

if [ -d "$_BUILD_TENSILE/Tensile" ]; then
  TENSILE_DIR="$_BUILD_TENSILE"
  echo "Using build-tree Tensile: $TENSILE_DIR"
else
  TENSILE_DIR="$_LOCAL_TENSILE"
  echo "Using local submodule Tensile: $TENSILE_DIR"
fi

# ── Apply Cosmo patches to Tensile ────────────────────────────────────────
PATCH="$SCRIPT_DIR/patches/0001-cosmo-tensile-tuning-fixes.rocm-libraries.patch"
if [ -f "$PATCH" ] && [ "$TENSILE_DIR" = "$_BUILD_TENSILE" ]; then
  MONO_REPO="$SCRIPT_DIR/tmp_rebuild/rocm-libraries"
  if cd "$MONO_REPO" && git apply --check "$PATCH" 2>/dev/null; then
    echo "=== Applying Cosmo Tensile patches ==="
    git apply "$PATCH"
    echo "  Patch applied successfully"
  else
    echo "  Cosmo patches already applied or not needed (skipping)"
  fi
  cd "$SCRIPT_DIR"
fi

apt update && apt install -y rocm-llvm-dev

rm -rf "$TENSILE_DIR/build_tmp/"
export LLVM_DIR=/opt/rocm/llvm

echo "=== Building TensileLite client ($TENSILE_DIR) ==="
cd "$TENSILE_DIR"
pip3 install invoke
invoke build-client
cd "$SCRIPT_DIR"

echo "=== Installing Origami ==="
_ORIGAMI_BUILD="$SCRIPT_DIR/tmp_rebuild/rocm-libraries/shared/origami/python"
_ORIGAMI_SUB="$SCRIPT_DIR/origami/shared/origami/python"
if [ -d "$_ORIGAMI_BUILD" ]; then
  pip3 install "$_ORIGAMI_BUILD"
elif [ -d "$_ORIGAMI_SUB" ]; then
  pip3 install "$_ORIGAMI_SUB"
else
  echo "  WARNING: origami not found (run rebuild_hipblaslt.sh first or init origami submodule)"
fi

echo "=== Done ==="
