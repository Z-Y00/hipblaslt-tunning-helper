#!/usr/bin/env bash
# Build hipBLASLt from source with tuned Tensile kernels and install.
#
# End-to-end workflow:
#   1. Collect 3_LibraryLogic YAML files from tunning_results/
#   2. Merge tuned logic into the hipblaslt base Equality logic
#   3. Build + package + install hipblaslt via install.sh (deb)
#   4. Verify with hipblaslt-bench (kernel dispatch check)
#   5. Verify with turbo PyTorch GEMM (end-to-end check)
#
# Prerequisites:
#   - Submodules initialized (run init_build.sh first)
#   - Tuning completed via run_shapes.py --run
#   - ROCm 7.x installed at /opt/rocm with amdclang++
#
# Usage:
#   ./build_hipblaslt.sh                    # full pipeline
#   ./build_hipblaslt.sh --merge-only       # only merge logic, skip build
#   ./build_hipblaslt.sh --build-only       # skip merge, only build + install
#   ./build_hipblaslt.sh --skip-install     # build but don't package/install
#   ./build_hipblaslt.sh --skip-verify      # skip post-install verification
#   ./build_hipblaslt.sh --gate-threshold 1.05  # only merge if T/B >= 1.05
#   ./build_hipblaslt.sh --no-gate          # merge all, skip regression gate
#
# Verify tuned kernels at runtime:
#   TENSILE_DB=0xFFFF python3 your_benchmark.py
#   HIPBLASLT_LOG_MASK=128 HIPBLASLT_LOG_FILE=./trace_%i.log python3 your_benchmark.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Paths ────────────────────────────────────────────────────────────────
HIPBLASLT_DIR="$SCRIPT_DIR/hipblaslt"
TENSILE_DIR="$HIPBLASLT_DIR/tensilelite"
TENSILE_BIN="$TENSILE_DIR/Tensile/bin"
ROCISA_LIB="$TENSILE_DIR/build_tmp/tensilelite/rocisa/lib"
RESULTS_DIR="$SCRIPT_DIR/tunning_results"
BASE_LOGIC="$HIPBLASLT_DIR/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/Equality"
INSTALL_PREFIX="/opt/rocm"
GPU_ARCH="gfx950"

export PYTHONPATH="${PYTHONPATH:-}:$ROCISA_LIB"

# ── Parse args ───────────────────────────────────────────────────────────
MERGE_ONLY=0
BUILD_ONLY=0
SKIP_INSTALL=0
SKIP_VERIFY=0
GATE_THRESHOLD="1.0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --merge-only)       MERGE_ONLY=1; shift ;;
    --build-only)       BUILD_ONLY=1; shift ;;
    --skip-install)     SKIP_INSTALL=1; shift ;;
    --skip-verify)      SKIP_VERIFY=1; shift ;;
    --gate-threshold)   shift; GATE_THRESHOLD="${1:-1.0}"; shift ;;
    --gate-threshold=*) GATE_THRESHOLD="${1#*=}"; shift ;;
    --no-gate)          GATE_THRESHOLD="0"; shift ;;
    -h|--help)
      sed -n '2,/^set/{ /^#/s/^# \?//p }' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; shift ;;
  esac
done

log() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }

# =====================================================================
# Steps 1-2: Collect and merge tuned logic
# =====================================================================
_get_approved_shapes() {
  local csv="$1" threshold="$2"
  python3 - "$csv" "$threshold" <<'PYEOF'
import csv, sys, os
csv_path, threshold = sys.argv[1], float(sys.argv[2])
if threshold <= 0:
    print("__ALL__")
    sys.exit(0)
if not os.path.isfile(csv_path):
    print("__ALL__", file=sys.stderr)
    print(f"  [gate] No report at {csv_path}, allowing all shapes", file=sys.stderr)
    print("__ALL__")
    sys.exit(0)
approved, gated = [], []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        t = float(row.get("tensile_tflops") or 0)
        b = float(row.get("bench_tflops") or 0)
        trans = row.get("trans", "TNN")
        shape_id = (f"{row['model']}_{row['layer']}_mbs{row['mbs']}"
                    f"_{row.get('phase','fwd')}_{trans}"
                    f"_M{row['M']}_N{row['N']}_K{row['K']}")
        ratio = t / b if b > 0 else 0
        if ratio >= threshold:
            approved.append(shape_id)
        else:
            gated.append((shape_id, ratio))
if gated:
    print(f"  [gate] {len(gated)} shape(s) gated (T/B < {threshold:.2f}):", file=sys.stderr)
    for sid, r in gated:
        print(f"         {sid}  T/B={r:.2%}", file=sys.stderr)
if approved:
    print(f"  [gate] {len(approved)} shape(s) approved", file=sys.stderr)
for s in approved:
    print(s)
PYEOF
}

merge_tuned_logic() {
  log "Step 1/5: Collect tuned 3_LibraryLogic files (gate threshold=${GATE_THRESHOLD})"

  local merged_dir
  merged_dir=$(mktemp -d /tmp/hipblaslt_merged_logic.XXXXXX)
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
        cp "$f" "$merged_dir/"
        count=$((count + 1))
      done
    done
  done

  echo "  Collected: $count logic file(s), gated: $gated shape(s)"

  if [ "$count" -eq 0 ]; then
    echo ""
    echo "  WARNING: No approved 3_LibraryLogic files found."
    echo "  Run tuning first: python3 run_shapes.py --run"
    rm -rf "$merged_dir"
    return 1
  fi

  log "Step 2/5: Merge tuned logic into base library"
  echo "  Base:  $BASE_LOGIC"
  echo "  Tuned: $merged_dir ($count files)"

  python3 "$TENSILE_BIN/TensileMergeLibrary" \
    --no_eff \
    "$BASE_LOGIC" \
    "$merged_dir" \
    "$BASE_LOGIC"

  echo "  Merge complete."
  rm -rf "$merged_dir"
}

# =====================================================================
# Step 3: Build + package + install hipblaslt via install.sh
# =====================================================================

# Known-broken logic files that cause VGPR overflow during asm codegen.
# These are FP16/FP32 problem types that exceed the 256-VGPR limit on gfx950.
# We temporarily hide them so TensileCreateLibrary can succeed.
QUARANTINE_DIR="$BASE_LOGIC/.quarantine"
QUARANTINE_PATTERNS=(HSS SS_BSS)

quarantine_broken_logic() {
  mkdir -p "$QUARANTINE_DIR"
  local moved=0
  for pat in "${QUARANTINE_PATTERNS[@]}"; do
    for f in "$BASE_LOGIC"/*"$pat"*.yaml; do
      [ -f "$f" ] || continue
      mv "$f" "$QUARANTINE_DIR/"
      moved=$((moved + 1))
    done
  done
  if [ "$moved" -gt 0 ]; then
    echo "  Quarantined $moved broken logic file(s) (VGPR overflow)"
  fi
}

restore_quarantined_logic() {
  if [ -d "$QUARANTINE_DIR" ]; then
    local restored=0
    for f in "$QUARANTINE_DIR"/*.yaml; do
      [ -f "$f" ] || continue
      mv "$f" "$BASE_LOGIC/"
      restored=$((restored + 1))
    done
    rmdir "$QUARANTINE_DIR" 2>/dev/null || true
    if [ "$restored" -gt 0 ]; then
      echo "  Restored $restored quarantined logic file(s)"
    fi
  fi
}

build_and_install_hipblaslt() {
  log "Step 3/5: Build & install hipBLASLt via install.sh (deb package)"
  echo "  Source:  $HIPBLASLT_DIR"
  echo "  Arch:    $GPU_ARCH"
  echo "  Filter:  gfx950/Equality/*"

  quarantine_broken_logic
  trap 'restore_quarantined_logic' EXIT

  cd "$HIPBLASLT_DIR"

  local install_args=(
    -i                                          # build + package + dpkg -i
    -a "$GPU_ARCH"                              # GPU architecture
    --skip_rocroller                            # no rocRoller backend
    --logic-yaml-filter "gfx950/Equality/*"     # only compile gfx950 Equality logic
  )

  if [ "$SKIP_INSTALL" -eq 1 ]; then
    # drop -i so it only builds, no package/install
    install_args=( -a "$GPU_ARCH" --skip_rocroller --logic-yaml-filter "gfx950/Equality/*" )
  fi

  echo "  Running: ./install.sh ${install_args[*]}"
  echo ""
  bash ./install.sh "${install_args[@]}"

  restore_quarantined_logic
  trap - EXIT

  echo ""
  echo "  Installed library:"
  ls -lh "$INSTALL_PREFIX/lib/libhipblaslt"* 2>/dev/null || true
  echo ""
  echo "  Installed Tensile library files:"
  ls "$INSTALL_PREFIX/lib/hipblaslt/library/" 2>/dev/null | wc -l
  echo "  files in $INSTALL_PREFIX/lib/hipblaslt/library/"
}

# =====================================================================
# Step 4: Verify via hipblaslt-bench
# =====================================================================
verify_tuned_kernel() {
  log "Step 4/5: Verify tuned kernels are dispatched (hipblaslt-bench)"

  local bench="$INSTALL_PREFIX/bin/hipblaslt-bench"
  if [ ! -x "$bench" ]; then
    bench="/opt/rocm/bin/hipblaslt-bench"
  fi
  if [ ! -x "$bench" ]; then
    echo "  hipblaslt-bench not found, skipping verification."
    echo "  You can verify manually with:"
    echo "    TENSILE_DB=0xFFFF hipblaslt-bench -m 4096 -n 12288 -k 4096 --precision bf16_r --compute_type f32_r --transA T --transB N -i 10 -j 5 --rotating 512 --print_kernel_info"
    return 0
  fi

  echo "  Using: $bench"
  echo "  Test shape: M=4096 N=12288 K=4096 (BF16, TransA=T, TransB=N)"
  echo ""

  set +e
  local output
  output=$(TENSILE_DB=0xFFFF "$bench" \
    -m 4096 -n 12288 -k 4096 \
    --precision bf16_r --compute_type f32_r \
    --transA T --transB N \
    -i 10 -j 5 --rotating 512 \
    --print_kernel_info 2>&1)
  local rc=$?
  set -e

  if [ "$rc" -ne 0 ]; then
    echo "  hipblaslt-bench failed (exit $rc):"
    echo "$output" | tail -10
    return 0
  fi

  local gflops solution_name kernel_name
  gflops=$(echo "$output" | grep "hipblaslt-Gflops" -A1 | tail -1 | awk -F, '{print $(NF-2)}')
  solution_name=$(echo "$output" | grep -- "--Solution name:" | head -1 | sed 's/.*--Solution name:\s*//')
  kernel_name=$(echo "$output" | grep -- "--kernel name:" | head -1 | sed 's/.*--kernel name:\s*//')

  echo "  GFLOPS:        ${gflops:-N/A}"
  echo "  Solution name: ${solution_name:-N/A}"
  echo "  Kernel name:   ${kernel_name:-N/A}"
  echo ""
  echo "  $output" | grep -E "transA|hipblaslt-Gflops" | head -2
}

# =====================================================================
# Step 5: Verify via turbo PyTorch GEMM
# =====================================================================
TURBO_DIR="$SCRIPT_DIR/turbo"

verify_turbo_pytorch() {
  log "Step 5/5: Verify tuned kernel via turbo PyTorch GEMM"

  if [ ! -d "$TURBO_DIR" ]; then
    echo "  turbo/ submodule not found, skipping PyTorch verification."
    return 0
  fi

  set +e
  python3 -c "import primus_turbo" 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "  primus_turbo not importable, skipping PyTorch verification."
    set -e
    return 0
  fi
  set -e

  echo "  Running BF16 GEMM: M=4096, N=12288, K=4096 (trans_b=True)"
  echo "  This exercises the same hipblaslt dispatch path as LLM inference."
  echo ""

  set +e
  local output
  output=$(TENSILE_DB=0xFFFF python3 - 2>&1 <<'PYEOF'
import torch
import torch.utils.benchmark as benchmark
import sys, os

M, N, K = 4096, 12288, 4096
dtype = torch.bfloat16
device = "cuda"

try:
    import primus_turbo.pytorch as turbo
    use_turbo = True
except ImportError:
    use_turbo = False

a = torch.randn(M, K, dtype=dtype, device=device)
b = torch.randn(N, K, dtype=dtype, device=device)

if use_turbo:
    gemm_fn = lambda: turbo.ops.gemm(a, b, trans_b=True)
    label = "turbo.ops.gemm"
else:
    b_t = b.T.contiguous().T
    gemm_fn = lambda: torch.mm(a, b_t)
    label = "torch.mm"

# warmup
for _ in range(20):
    gemm_fn()
torch.cuda.synchronize()

# timed run (this also prints TENSILE_DB kernel names to stderr)
timer = benchmark.Timer(stmt="fn()", globals={"fn": gemm_fn})
measurement = timer.timeit(50)
mean_ms = measurement.mean * 1e3
tflops = (2 * M * N * K) / (mean_ms * 1e-3) / 1e12

print(f"TURBO_RESULT: engine={label} M={M} N={N} K={K} mean_ms={mean_ms:.3f} tflops={tflops:.2f}")
PYEOF
  )
  local rc=$?
  set -e

  if [ "$rc" -ne 0 ]; then
    echo "  PyTorch GEMM failed (exit $rc):"
    echo "$output" | tail -15
    return 0
  fi

  local result_line kernel_lines
  result_line=$(echo "$output" | grep "TURBO_RESULT:" | head -1)
  kernel_lines=$(echo "$output" | grep "Running kernel:" | head -3)

  if [ -n "$result_line" ]; then
    echo "  $result_line"
  fi
  echo ""

  if [ -n "$kernel_lines" ]; then
    echo "  Kernels dispatched by hipblaslt (via TENSILE_DB):"
    echo "$kernel_lines" | head -3 | while read -r line; do
      echo "    $line"
    done
  else
    echo "  No TENSILE_DB kernel output captured."
    echo "  (Kernel names may only appear on first dispatch; try running manually)"
  fi
}

# =====================================================================
# Main
# =====================================================================
echo "================================================================"
echo "  hipBLASLt Build with Tuned Tensile Kernels"
echo "  $(date)"
echo "================================================================"

if [ "$BUILD_ONLY" -eq 0 ]; then
  merge_tuned_logic || { echo "Merge failed, continuing to build anyway..."; }
fi

if [ "$MERGE_ONLY" -eq 1 ]; then
  echo ""
  echo "=== Merge-only mode, stopping here ==="
  exit 0
fi

build_and_install_hipblaslt

if [ "$SKIP_VERIFY" -eq 0 ]; then
  verify_tuned_kernel
  verify_turbo_pytorch
fi

echo ""
log "Done"
echo ""
echo "To verify tuned kernels in your own code:"
echo "  TENSILE_DB=0xFFFF python3 your_benchmark.py"
echo ""
echo "  # Extended profile (solution + kernel names to log file):"
echo "  HIPBLASLT_LOG_MASK=128 HIPBLASLT_LOG_FILE=./trace_%i.log python3 your_benchmark.py"
