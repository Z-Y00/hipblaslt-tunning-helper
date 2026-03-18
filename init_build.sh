#!/usr/bin/env bash
set -euo pipefail

# Clone rocm-libraries (sparse), apply patches, build TensileLite client,
# install Origami, and cache git deps.
#
# Run once after cloning the helper repo.  Subsequent runs are idempotent
# (skips clone if tmp_rebuild/ exists, skips patch if already applied).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Source repo (must match reference/hipblaslt.Dockerfile) ───────────────
ROCM_LIBS_REPO="${ROCM_LIBS_REPO:-https://github.com/ROCm/rocm-libraries.git}"
ROCM_LIBS_COMMIT="${ROCM_LIBS_COMMIT:-b3db63927d3df09ec2f93d46e733d7a0ab51b87b}"

MONO_REPO="$SCRIPT_DIR/tmp_rebuild/rocm-libraries"

GIT_CACHE="$SCRIPT_DIR/.git-cache"

# ── Cache git repos used by cmake FetchContent ───────────────────────────
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
    local name=$(basename "$url")
    local mirror="$GIT_CACHE/$name"
    if [ -d "$mirror" ]; then
      echo "  $name: updating existing mirror"
      git -C "$mirror" fetch --all -q 2>/dev/null || true
    else
      echo "  $name: creating bare mirror"
      git clone --bare "$url" "$mirror"
    fi
    git config --global url."file://$mirror".insteadOf "$url"
  done
  echo "  git cache ready at $GIT_CACHE"
}

# ── Clone rocm-libraries with sparse checkout ────────────────────────────
# Only fetches projects/hipblaslt/ and shared/origami/ (~500 MB vs ~10 GB full).
clone_rocm_libraries() {
  echo "=== Cloning rocm-libraries (sparse checkout) ==="
  echo "  Repo:   $ROCM_LIBS_REPO"
  echo "  Commit: $ROCM_LIBS_COMMIT"

  if [ -d "$MONO_REPO/.git" ]; then
    echo "  Already cloned at $MONO_REPO, resetting to $ROCM_LIBS_COMMIT"
    cd "$MONO_REPO"
    git sparse-checkout set projects/hipblaslt shared/origami
    git clean -fd
    git reset --hard
    git checkout "$ROCM_LIBS_COMMIT" 2>/dev/null || git checkout -B build "$ROCM_LIBS_COMMIT"
    cd "$SCRIPT_DIR"
    return
  fi

  mkdir -p "$SCRIPT_DIR/tmp_rebuild"
  git clone --filter=blob:none --no-checkout "$ROCM_LIBS_REPO" "$MONO_REPO"
  cd "$MONO_REPO"
  git sparse-checkout init --cone
  git sparse-checkout set projects/hipblaslt shared/origami
  git checkout "$ROCM_LIBS_COMMIT" 2>/dev/null || git checkout -B build "$ROCM_LIBS_COMMIT"
  cd "$SCRIPT_DIR"

  echo "  Sparse checkout complete ($(du -sh "$MONO_REPO" | awk '{print $1}'))"
}

# ── Apply Cosmo patches ──────────────────────────────────────────────────
apply_patches() {
  local patch="$SCRIPT_DIR/patches/0001-cosmo-tensile-tuning-fixes.rocm-libraries.patch"
  if [ ! -f "$patch" ]; then
    echo "  No patches found, skipping"
    return
  fi

  cd "$MONO_REPO"
  if git apply --check "$patch" 2>/dev/null; then
    echo "=== Applying Cosmo Tensile patches ==="
    git apply "$patch"
    echo "  Patch applied successfully"
  else
    echo "  Cosmo patches already applied or not needed (skipping)"
  fi
  cd "$SCRIPT_DIR"
}

# ── Main ─────────────────────────────────────────────────────────────────
setup_git_cache
clone_rocm_libraries
apply_patches

# Detect Tensile directory
_BUILD_TENSILE="$MONO_REPO/projects/hipblaslt/tensilelite"
_LOCAL_TENSILE="$SCRIPT_DIR/hipblaslt/tensilelite"

if [ -d "$_BUILD_TENSILE/Tensile" ]; then
  TENSILE_DIR="$_BUILD_TENSILE"
  echo "Using build-tree Tensile: $TENSILE_DIR"
else
  TENSILE_DIR="$_LOCAL_TENSILE"
  echo "Using local submodule Tensile: $TENSILE_DIR"
fi

apt update && apt install -y rocm-llvm-dev

rm -rf "$TENSILE_DIR/build_tmp/"
export LLVM_DIR=/opt/rocm/llvm

echo "=== Building TensileLite client ($TENSILE_DIR) ==="
cd "$TENSILE_DIR"
pip3 install invoke
invoke build-client
cd "$SCRIPT_DIR"

echo "=== Building API bench driver ==="
if [ -f "$SCRIPT_DIR/test_hipblaslt_api.cpp" ]; then
  hipcc "$SCRIPT_DIR/test_hipblaslt_api.cpp" -lhipblaslt -o "$SCRIPT_DIR/test_hipblaslt_api"
  echo "  Built: $SCRIPT_DIR/test_hipblaslt_api"
else
  echo "  WARNING: test_hipblaslt_api.cpp not found, skipping"
fi

echo "=== Installing Origami ==="
_ORIGAMI_BUILD="$MONO_REPO/shared/origami/python"
_ORIGAMI_SUB="$SCRIPT_DIR/origami/shared/origami/python"
if [ -d "$_ORIGAMI_BUILD" ]; then
  pip3 install "$_ORIGAMI_BUILD"
elif [ -d "$_ORIGAMI_SUB" ]; then
  pip3 install "$_ORIGAMI_SUB"
else
  echo "  WARNING: origami not found"
fi

echo "=== Done ==="
