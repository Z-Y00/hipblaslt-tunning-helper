#!/usr/bin/env bash
# Rebuild hipBLASLt with tuned Tensile kernels merged in.
#
# Clones rocm-libraries into a temp directory (matching the Docker image build),
# merges tuned 3_LibraryLogic YAML files, builds, packages, and installs.
#
# Workflow:
#   1. Clone rocm-libraries (same repo/commit as Docker image)
#   2. Collect 3_LibraryLogic YAML files from tunning_results/
#   3. Merge tuned logic into the cloned hipblaslt Equality logic
#   4. Build hipBLASLt via cmake (same command as Dockerfile)
#   5. Package + install via dpkg
#
# Usage:
#   ./rebuild_hipblaslt.sh                     # full pipeline
#   ./rebuild_hipblaslt.sh --merge-only        # only clone+merge, skip build
#   ./rebuild_hipblaslt.sh --skip-install      # build but don't dpkg -i
#   ./rebuild_hipblaslt.sh --no-gate           # merge all shapes (skip T/B gate)
#   ./rebuild_hipblaslt.sh --gate-threshold 1.05
#   ./rebuild_hipblaslt.sh --keep-source       # don't delete cloned source after build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Source repo (must match reference/hipblaslt.Dockerfile) ───────────────
HIPBLASLT_REPO="${HIPBLASLT_REPO:-https://github.com/ROCm/rocm-libraries.git}"
HIPBLASLT_BRANCH="${HIPBLASLT_BRANCH:-b3db63927d3df09ec2f93d46e733d7a0ab51b87b}"

# ── Paths ─────────────────────────────────────────────────────────────────
TENSILE_BIN="$SCRIPT_DIR/hipblaslt/tensilelite/Tensile/bin"
ROCISA_LIB="$SCRIPT_DIR/hipblaslt/tensilelite/build_tmp/tensilelite/rocisa/lib"
RESULTS_DIR="$SCRIPT_DIR/tunning_results"

ROCM_PATH="${ROCM_PATH:-$(readlink -f /opt/rocm)}"
GPU_ARCH="${GPU_ARCH:-gfx942;gfx950}"

TORCH_LIB="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || echo "")"
TORCH_TENSILE="${TORCH_LIB:+$TORCH_LIB/hipblaslt/library}"

# Tensile Python package: TENSILE_BIN is <tensilelite>/Tensile/bin,
# so two levels up gives the <tensilelite> dir containing the Tensile package.
TENSILE_PYPATH="$(cd "$(dirname "$TENSILE_BIN")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$ROCISA_LIB:$TENSILE_PYPATH"

# ── Parse args ────────────────────────────────────────────────────────────
MERGE_ONLY=0
SKIP_INSTALL=0
KEEP_SOURCE=0
GATE_THRESHOLD="1.05"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --merge-only)       MERGE_ONLY=1; shift ;;
    --skip-install)     SKIP_INSTALL=1; shift ;;
    --keep-source)      KEEP_SOURCE=1; shift ;;
    --gate-threshold)   shift; GATE_THRESHOLD="${1:-1.05}"; shift ;;
    --gate-threshold=*) GATE_THRESHOLD="${1#*=}"; shift ;;
    --no-gate)          GATE_THRESHOLD="0"; shift ;;
    -h|--help)
      sed -n '2,/^set/{ /^#/s/^# \?//p }' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

log()  { echo -e "\n\033[1;36m════ $* ════\033[0m"; }
warn() { echo -e "\033[1;33m  WARN: $*\033[0m"; }

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Clone rocm-libraries into a temp directory
# ══════════════════════════════════════════════════════════════════════════
clone_source() {
  log "Step 1: Clone hipBLASLt source"
  echo "  Repo:   $HIPBLASLT_REPO"
  echo "  Commit: $HIPBLASLT_BRANCH"

  mkdir -p $SCRIPT_DIR/tmp_rebuild
  BUILD_TMP=$SCRIPT_DIR/tmp_rebuild
  echo "  Clone target: $BUILD_TMP"

  git clone "$HIPBLASLT_REPO" "$BUILD_TMP/rocm-libraries" || echo "folder exists, assume already cloned"
  cd "$BUILD_TMP/rocm-libraries/projects/hipblaslt"
  git clean -f # Lorri: since we are repeatly reuse the same folder
  git reset --hard
  git checkout "$HIPBLASLT_BRANCH"

  HIPBLASLT_SRC="$BUILD_TMP/rocm-libraries/projects/hipblaslt"
  BASE_LOGIC="$HIPBLASLT_SRC/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/Equality"

  cd "$SCRIPT_DIR"
}

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Collect + gate tuned logic YAML files
# ══════════════════════════════════════════════════════════════════════════
_get_approved_shapes() {
  local csv="$1" threshold="$2"
  python3 - "$csv" "$threshold" <<'PYEOF'
import csv, sys, os
csv_path, threshold = sys.argv[1], float(sys.argv[2])
if threshold <= 0:
    print("__ALL__"); sys.exit(0)
if not os.path.isfile(csv_path):
    print("__ALL__", file=sys.stderr)
    print(f"  [gate] No report at {csv_path}, allowing all shapes", file=sys.stderr)
    print("__ALL__"); sys.exit(0)
approved, gated = [], []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        t = float(row.get("tensile_tflops") or 0)
        a = float(row.get("api_tflops") or 0)
        b = float(row.get("bench_tflops") or 0)
        trans = row.get("trans", "TNN")
        shape_id = (f"{row['model']}_{row['layer']}_mbs{row['mbs']}"
                    f"_{row.get('phase','fwd')}_{trans}"
                    f"_M{row['M']}_N{row['N']}_K{row['K']}")
        # Prefer api_tflops (same methodology as Tensile) for gating;
        # fall back to bench_tflops if api is unavailable
        if a > 0:
            ratio = t / a
            tag = "T/A"
        elif b > 0:
            ratio = t / b
            tag = "T/B"
        else:
            ratio = 0
            tag = "N/A"
        if ratio >= threshold:
            approved.append(shape_id)
        else:
            gated.append((shape_id, ratio, tag))
if gated:
    print(f"  [gate] {len(gated)} shape(s) gated (ratio < {threshold:.2f}):", file=sys.stderr)
    for sid, r, tag in gated:
        print(f"         {sid}  {tag}={r:.2%}", file=sys.stderr)
if approved:
    print(f"  [gate] {len(approved)} shape(s) approved", file=sys.stderr)
for s in approved:
    print(s)
PYEOF
}

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Merge tuned logic into cloned source
# ══════════════════════════════════════════════════════════════════════════
merge_tuned_logic() {
  log "Step 2: Collect + merge tuned 3_LibraryLogic (gate=${GATE_THRESHOLD})"

  # Multiple shapes with the same transpose produce the same YAML filename
  # (e.g. all fwd TN shapes → gfx950_Cijk_Alik_Bljk_BBS_BH_..._SAV_UserArgs.yaml).
  # A flat copy would overwrite earlier files. Instead, merge each shape's
  # output one at a time into a combined directory, then merge that into stock.
  local combined_dir inc_dir
  combined_dir=$(mktemp -d /tmp/hipblaslt_combined_logic.XXXXXX)
  inc_dir=$(mktemp -d /tmp/hipblaslt_inc_logic.XXXXXX)
  local count=0 gated=0

  for search_root in "$RESULTS_DIR"/bf16 "$RESULTS_DIR"/f8 "$RESULTS_DIR"/f8b8; do
    [ -d "$search_root" ] || continue

    local approved_list
    approved_list=$(_get_approved_shapes "$search_root/comparison_report.csv" "$GATE_THRESHOLD")

    for case_dir in "$search_root"/*/; do
      [ -d "${case_dir}3_LibraryLogic" ] || continue
      local folder_name
      folder_name=$(basename "$case_dir")

      if [ "$approved_list" != "__ALL__" ]; then
        if ! echo "$approved_list" | grep -qxF "$folder_name"; then
          gated=$((gated + 1))
          continue
        fi
      fi

      for f in "${case_dir}3_LibraryLogic/"*.yaml; do
        [ -f "$f" ] || continue
        local base
        base=$(basename "$f")
        # Rename to stock naming convention
        if [ -f "$combined_dir/$base" ]; then
          # Same filename already in combined → merge incrementally
          rm -f "$inc_dir"/*.yaml
          cp "$f" "$inc_dir/$base"
          python3 "$TENSILE_BIN/TensileMergeLibrary" \
            --force_merge true -v 0 \
            "$combined_dir" "$inc_dir" "$combined_dir" 2>&1 | \
            grep -E "size|kernel|added|written" || true
        else
          cp "$f" "$combined_dir/$base"
        fi
        count=$((count + 1))
      done
    done
  done

  echo "  Collected: $count logic file(s) from $((count + gated)) shape(s), gated: $gated"
  echo "  Combined into $(ls "$combined_dir"/*.yaml 2>/dev/null | wc -l) unique problem type(s)"

  if [ "$count" -eq 0 ]; then
    warn "No approved 3_LibraryLogic files found. Run tuning first."
    rm -rf "$combined_dir" "$inc_dir"
    return 1
  fi

  echo "  Base logic:  $BASE_LOGIC"
  # Lorri: no eff , efficiency is needed for merger to decide keep old or replace new
  python3 "$TENSILE_BIN/TensileMergeLibrary" \
    --no_eff \
    "$BASE_LOGIC" \
    "$combined_dir" \
    "$BASE_LOGIC"

  echo "  Merge complete."
  rm -rf "$combined_dir" "$inc_dir"
}

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Build hipBLASLt (same cmake as Dockerfile)
# ══════════════════════════════════════════════════════════════════════════

# QUARANTINE_PATTERNS=(HSS SS_BSS)

# quarantine_broken_logic() {
#   local qdir="$BASE_LOGIC/.quarantine"
#   mkdir -p "$qdir"
#   local moved=0
#   for pat in "${QUARANTINE_PATTERNS[@]}"; do
#     for f in "$BASE_LOGIC"/*"$pat"*.yaml; do
#       [ -f "$f" ] || continue
#       mv "$f" "$qdir/"
#       moved=$((moved + 1))
#     done
#   done
#   [ "$moved" -gt 0 ] && echo "  Quarantined $moved broken logic file(s) (VGPR overflow)"
# }

# restore_quarantined_logic() {
#   local qdir="$BASE_LOGIC/.quarantine"
#   [ -d "$qdir" ] || return 0
#   local restored=0
#   for f in "$qdir"/*.yaml; do
#     [ -f "$f" ] || continue
#     mv "$f" "$BASE_LOGIC/"
#     restored=$((restored + 1))
#   done
#   rmdir "$qdir" 2>/dev/null || true
#   [ "$restored" -gt 0 ] && echo "  Restored $restored quarantined logic file(s)"
# }

build_hipblaslt() {
  log "Step 3: Build hipBLASLt (GPU_TARGETS=${GPU_ARCH})"

  rm -rf "$HIPBLASLT_SRC/build/"

  cd "$HIPBLASLT_SRC"

  python3 -m pip install -q -r tensilelite/requirements.txt 2>/dev/null || true

  echo "  Installing AOCL (BLAS dependency)..."
  if [ ! -f /opt/AMD/aocl/aocl-linux-gcc-4.2.0/lib/libblis.a ]; then
    wget -nv https://download.amd.com/developer/eula/aocl/aocl-4-2/aocl-linux-gcc-4.2.0_1_amd64.deb -O /tmp/aocl.deb
    apt install -y /tmp/aocl.deb 2>/dev/null || dpkg -i /tmp/aocl.deb 2>/dev/null || true
    rm -f /tmp/aocl.deb
  else
    echo "  AOCL already installed"
  fi

  echo "  Configuring..."
  cmake --preset rocm-7.0.0 \
    -DGPU_TARGETS="${GPU_ARCH}" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH}/lib/llvm;${ROCM_PATH}" \
    -DCMAKE_INSTALL_PREFIX="${ROCM_PATH}" \
    -DCMAKE_PACKAGING_INSTALL_PREFIX="${ROCM_PATH}" \
    -DROCM_PATH="${ROCM_PATH}" \
    -B build -S .

  echo "  Building + packaging..."
  cmake --build build --target package --parallel

  # restore_quarantined_logic
  # trap - EXIT

  cd "$SCRIPT_DIR"
}

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Install via dpkg
# ══════════════════════════════════════════════════════════════════════════
install_hipblaslt() {
  log "Step 4: Install hipBLASLt"

  local old_ver
  old_ver=$(dpkg -l 2>/dev/null | awk '$2=="hipblaslt" { print $3 }')

  echo "  Removing old hipblaslt packages..."
  dpkg --remove --force-depends hipblaslt hipblaslt-dev 2>/dev/null || true

  echo "  Installing new packages..."
  dpkg -i "$HIPBLASLT_SRC/build/"*.deb

  echo "  Installed library:"
  ls -lh "${ROCM_PATH}/lib/libhipblaslt"* 2>/dev/null || true
  echo "  Tensile library files: $(ls "${ROCM_PATH}/lib/hipblaslt/library/" 2>/dev/null | wc -l) in ${ROCM_PATH}/lib/hipblaslt/library/"

  # PyTorch compat symlink (before pytorch supports hipblaslt 1.x.x)
  ln -sf "${ROCM_PATH}/lib/libhipblaslt.so" "${ROCM_PATH}/lib/libhipblaslt.so.0"
  echo "  Created libhipblaslt.so.0 symlink (PyTorch compat)"

  # Fix apt version tracking (same as Dockerfile)
  local new_ver
  new_ver=$(dpkg -l 2>/dev/null | awk '$2=="hipblaslt" { print $3 }')
  if [ -n "$old_ver" ] && [ -n "$new_ver" ] && [ "$old_ver" != "$new_ver" ]; then
    sed -i -e "s/hipblaslt (= ${old_ver})/hipblaslt (= ${new_ver})/g" /var/lib/dpkg/status 2>/dev/null || true
    sed -i -e "s/hipblaslt-dev (= ${old_ver})/hipblaslt-dev (= ${new_ver})/g" /var/lib/dpkg/status 2>/dev/null || true
    echo "  Updated dpkg status: ${old_ver} → ${new_ver}"
  fi
}



# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
echo ">>  Rebuild hipBLASLt with tuned Tensile kernels"
echo ">>  $(date)"
echo ">>  ROCM_PATH=$ROCM_PATH  GPU_ARCH=$GPU_ARCH"
echo ">>  Repo: $HIPBLASLT_REPO"
echo ">>  Commit: $HIPBLASLT_BRANCH"

OVERALL_START=$SECONDS

clone_source
merge_tuned_logic || { warn "Merge failed or no files to merge"; [ "$MERGE_ONLY" -eq 1 ] && exit 1; }

if [ "$MERGE_ONLY" -eq 1 ]; then
  echo ""; log "Merge-only mode complete"
  echo "  Source with merged logic at: $HIPBLASLT_SRC"
  exit 0
fi

build_hipblaslt

if [ "$SKIP_INSTALL" -eq 0 ]; then
  install_hipblaslt
fi


ELAPSED=$(( SECONDS - OVERALL_START ))
log "Done in ${ELAPSED}s"
echo ""
echo "  Verify tuned kernels in your own code:"
echo "    export TENSILE_DB=0x8080 
export HIPBLASLT_LOG_MASK=32 
export HIPBLASLT_LOG_FILE=./trace_%i.log" # see https://rocm.docs.amd.com/_/downloads/Tensile/en/docs-6.3.0/pdf/#:~:text=Advanced%20Micro%20Devices%2C%20Inc.%20Tensile%20is%20a,(%20Advanced%20Micro%20Devices%2C%20Inc%20)%20GPU.
echo "     python3 your_benchmark.py"
echo ""
