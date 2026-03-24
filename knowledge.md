# hipBLASLt Tuning Knowledge Base

## Re-running a Specific Tensile Solution from Tuning Results

After a Tensile tuning run completes (via `run_shapes.py`), you can re-benchmark
individual solutions from the result library without re-running the full tuning.

### Locate the Client and Config

Each tuned shape produces a directory under `tunning_results/<dtype>/<shape_id>/`.
The key files are:

```
tunning_results/bf16/<shape_id>/
├── 1_BenchmarkProblems/
│   └── Cijk_..._00/
│       ├── 00_Final/
│       │   └── source/
│       │       ├── ClientParameters.ini   # <-- benchmark config
│       │       └── library/
│       │           └── TensileLibrary_gfx950.co  # <-- compiled kernels
│       └── Data/
│           └── 00_Final.csv
└── 2_BenchmarkData/
    └── Cijk_..._CSVWinner.csv   # <-- winner summary
```

The Tensile client binary is at:
```
tmp_rebuild/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/tensilelite/client/tensilelite-client
```

### Find the Winner Solution Index

```bash
python3 -c "
import csv
with open('tunning_results/bf16/<shape_id>/2_BenchmarkData/Cijk_*_CSVWinner.csv') as f:
    for i, row in enumerate(csv.DictReader(f)):
        wg = float(row.get(' WinnerGFlops') or row.get('WinnerGFlops') or 0)
        wi = row.get(' WinnerIdx') or row.get('WinnerIdx') or '?'
        wn = (row.get(' WinnerName') or row.get('WinnerName') or '').strip()
        print(f'Row {i}: idx={wi}  {wg/1e6:.1f} TFLOPS  {wn[-80:]}')
"
```

### Re-benchmark a Specific Solution

Use `--solution-start-idx` and `--num-solutions 1` to run only the desired solution:

```bash
CLIENT="tmp_rebuild/rocm-libraries/projects/hipblaslt/tensilelite/build_tmp/tensilelite/client/tensilelite-client"
CONFIG="tunning_results/bf16/<shape_id>/1_BenchmarkProblems/Cijk_..._00/00_Final/source/ClientParameters.ini"

# Run only solution 911 (example), 10 benchmark iterations
HIP_VISIBLE_DEVICES=4 $CLIENT \
    --config-file "$CONFIG" \
    --solution-start-idx 911 \
    --num-solutions 1 \
    --num-benchmarks 10
```

### Run the Stock Library Baseline

Use `--best-solution` to benchmark the library's pre-selected best solution (solution 0):

```bash
HIP_VISIBLE_DEVICES=4 $CLIENT \
    --config-file "$CONFIG" \
    --best-solution \
    --num-benchmarks 10
```

### Control Enqueues-per-Sync

By default, the config uses `num-enqueues-per-sync=50` (50 kernel launches batched
before measuring). You can override:

```bash
# Batch mode (original tuning behavior)
--num-enqueues-per-sync 50 --num-syncs-per-benchmark 1

# Per-call mode (closer to hipblaslt-bench behavior)
--num-enqueues-per-sync 1 --num-syncs-per-benchmark 50
```

**Finding:** Both modes produce nearly identical results (~556 µs for this kernel).
The ~12% gap between Tensile client and hipblaslt-bench is NOT caused by batching.

### Compare with hipblaslt-bench

```bash
HIP_VISIBLE_DEVICES=4 /opt/rocm/bin/hipblaslt-bench \
    -m 14336 -n 8192 -k 4096 \
    --precision bf16_r --compute_type f32_r \
    --transA T --transB N \
    -i 50 -j 30 \
    --rotating 0 \
    --use_gpu_timer \
    --print_kernel_info
```

### Output Format

The Tensile client outputs CSV rows with these fields:
```
run, problem-progress, solution-progress, operation, problem-sizes, bias-type,
factor-dim, activation-type, solution, validation, time-us, gflops, ...
```

Extract time and TFLOPS from `PASSED` lines:
```bash
... | grep PASSED | python3 -c "
import sys
for line in sys.stdin:
    parts = line.split('PASSED,')
    after = parts[1].split(',')
    time_us, gflops = float(after[0]), float(after[1])
    print(f'  {time_us:.1f} us  {gflops/1e6:.1f} TFLOPS')
"
```

## Tensile Client Timing Internals (Source-Verified)

Source: `hipblaslt/tensilelite/client/src/BenchmarkTimer.cpp` and `client/main.cpp`

### Benchmark Parameters (from ClientParameters.ini)

```
num-enqueues-per-sync=50     # 50 kernel launches per sync batch
num-syncs-per-benchmark=1    # 1 batch per benchmark run
use-gpu-timer=True           # GPU event timing (not host clock)
num-warmups=30               # warmups before timed run
icache-flush-args=False      # no icache flush (flushTimeUs=0)
rotating-buffer-size=0       # Template default; overridden per-shape by run_shapes.py
```

### Exact Timing Flow

1. **`preEnqueues()`**: Creates ONE event pair, records start:
   ```cpp
   hipEventCreate(&start);
   hipEventCreate(&stop);
   hipEventRecord(start, stream);
   ```

2. **Inner loop** (50 iterations): Direct kernel launch, NO per-kernel events:
   ```cpp
   for (int j = 0; j < 50; j++) {
       adapter.launchKernels(kernels[kIdx], stream, nullptr, nullptr);
       // → calls hipExtModuleLaunchKernel() directly
       // → nullptr events = NO hipEventRecord per kernel
   }
   ```

3. **`postEnqueues()`**: Records stop + synchronizes:
   ```cpp
   hipEventRecord(stop, stream);
   hipEventSynchronize(stop);
   ```

4. **`validateEnqueues()`**: Reads the single event pair:
   ```cpp
   hipEventElapsedTime(&eventMs, start, stop);
   // totalTime = eventMs (includes all 50 kernel executions + inter-kernel gaps)
   ```

5. **`postSolution()`**: Computes average:
   ```cpp
   timePerEnqueue_us = totalTime_us / numEnqueues - flushTimeUs;
   // flushTimeUs=0 when icache-flush=False
   gflops = flopCount / timePerEnqueue_us / 1000.0;
   ```

### Key Properties

- **Single event pair** around all 50 enqueues (NOT per-call events)
- **Zero per-kernel overhead**: `nullptr` events, no validation between calls
- **Direct `hipExtModuleLaunchKernel`**: Bypasses hipBLASLt dispatch layer entirely
- **Average (total/N)**: NOT best-of-N or median — just total time / 50
- **Inter-kernel gaps included**: The single event pair captures wall-clock GPU time
  including any micro-gaps between consecutive kernel executions

### Rotating Buffer Behavior (Source-Verified)

**CORRECTION**: Earlier notes stated `rotating-buffer-size=0` in all tuning configs.
This was true for the *template default*, but `run_shapes.py` **overrides** the YAML
`RotatingBufferSize` per-shape via `compute_rotating_buffer_mb()`, so the generated
`ClientParameters.ini` gets a non-zero value for shapes that fit in cache.

The Tensile client **does** cycle through different buffer pointers during the timed
loop.  Source: `client/main.cpp` lines 1076–1203:

1. `prepareRotatingGPUOutput()` creates N copies of input buffers (A, B, C, D, etc.)
   in separate GPU memory regions, where N = `min(maxEnqueues, ceil(rotBufSize/tensorSetSize)) - 1`.
2. `solution->solve(*problem, *inputArr[r], ...)` is called for each copy,
   producing `kernels[r]` with **different buffer pointers baked in**.
3. The timed inner loop uses `kIdx = ((i * enq) + j) % kernels.size()`, so each
   of the 50 enqueues hits a **different memory address set** → L2 cache is cold.

When `RotatingBufferSize=0` (or tensor_set >= 2×LLC), `inputArr` has only 1 entry,
`kIdx` is always 0, and all enqueues hit the **same pointers** → warm L2.

### `compute_rotating_buffer_mb()` Logic (run_shapes.py)

```
LLC = 256 MB (gfx950)
tensor_set = (M*K + K*N) * elem_bytes + 2 * M*N * 2   # A + B + C + D
if tensor_set >= 512 MB:   return 0                     # too big, skip rotation
else: return min(max(tensor_set * 5, 512 MB), 8192 MB)  # ≥5 rotations or ≥2×LLC
```

Note: the formula counts only A, B, C, D — it does **not** include bias or
scaleAlphaVec.  This means the actual memory footprint per rotation may be slightly
larger than estimated.

### Three Benchmark Drivers and Their Cache Behavior

| Driver          | Rotating buffer                  | Cache state     | Purpose                          |
|-----------------|----------------------------------|-----------------|----------------------------------|
| Tensile client  | `compute_rotating_buffer_mb()`   | Cold L2 (if >0) | Tune under realistic conditions  |
| hipblaslt-bench | Same `compute_rotating_buffer_mb()` | Cold L2 (if >0) | Fair comparison, same methodology |
| API bench       | Same `compute_rotating_buffer_mb()` | Cold L2 (if >0) | Verify hipblasLtMatmul matches Tensile |

The API bench (`test_hipblaslt_api`) accepts `--rotating <MB>` and cycles through
multiple buffer copies, just like the Tensile client.  `run_shapes.py` automatically
passes the same `compute_rotating_buffer_mb()` value to all three drivers.

### Why hipBLASLt API Calls Measure Differently

A C++ driver using `hipblasLtMatmul()` with per-call events will report ~10-15%
higher TFLOPS than Tensile because:

1. **Best-of-N vs Average**: Taking best of N individual measurements always gives
   lower time than total/N average
2. **Per-call events exclude inter-kernel gaps**: Individual event pairs measure
   only kernel execution time, while Tensile's single pair includes dispatch gaps

To reproduce Tensile's numbers exactly: use ONE event pair around all N enqueues,
divide total elapsed time by N, use `hipExtModuleLaunchKernel` directly (not
`hipblasLtMatmul`), and match the same rotating-buffer-size.

## Measurement Bias: Tensile Client vs hipblaslt-bench

### Key Finding

There is a consistent ~12% measurement gap between the Tensile client and
hipblaslt-bench when benchmarking the **same kernel on the same idle GPU**:

| Tool                         | Time (µs) | TFLOPS | Notes                    |
|------------------------------|----------:|-------:|--------------------------|
| Tensile client (tuning .co)  |     556   |  1730  | JIT-compiled kernel      |
| hipblaslt-bench (installed)  |     624   |  1538  | AOT-compiled, via lib    |

### Root Cause

The gap is **not** caused by:
- Batching (50 vs 1 enqueue/sync) — tested, identical results
- GPU contention — tested on idle GPU
- Additional kernel launches per iteration — confirmed via `rocprofv3`

The gap **is** caused by:
- Different compiled kernel binaries (the installed .co has differently parameterized
  variants even for the same tile shape: different NT, NEPBS, SPO, SSO, SKXCCM values)
- Tensile uses direct `hipExtModuleLaunchKernel` while hipblaslt-bench uses the
  `hipblasLtMatmul` API with its dispatch overhead

### Implication for Gate Logic

The gate in `rebuild_hipblaslt.sh` compares `tensile_tflops / bench_tflops`.
Due to the ~12% tool bias, this ratio is always inflated — a kernel with ZERO
actual improvement will show T/B ≈ 1.12.

## T/A Ratio (Tensile vs API bench) — What It Tells You

The T/A ratio = `Tensile TFLOPS / API TFLOPS`.  It compares the Tensile client's
reported winner performance against `test_hipblaslt_api` which calls `hipblasLtMatmul`
with Tensile-aligned timing (single event pair, total/N average, random init, and
the **same** rotating buffer size as Tensile).

Both drivers now use the same cache conditions, so T/A should converge:

- **T/A ≈ 96–105%**: Tensile and API agree under identical cache conditions.
  These are trustworthy results.
- **T/A >> 110%**: Suspect.  Tensile reports much higher than API can reproduce.
  Common causes:
  1. hipBLASLt heuristic selects a **different kernel** than Tensile's tuned winner
  2. The tuned kernel's AOT-compiled variant behaves differently than Tensile's JIT
  3. GPU contention during the API bench run

### Patterns from BF16 Tuning (Llama-2-7B + Llama-2-70B, gfx950)

From the `.report.md` files (see warning below):

- **75% of shapes** have T/A 85–110% — trustworthy
- **25% of shapes** have T/A > 110% — hipBLASLt heuristic picks a worse kernel

**After re-verification** (single-GPU, idle, matching rotating buffers), 44 of 57
"suspect" shapes collapsed to healthy 85–105% T/A — the original high ratios were
logging artifacts from parallel interleaving (see warning below).

Only **4 `attn_k` shapes remain mildly elevated (111–119%)**:
  - These are small/skinny shapes with K=1024 or M=1024.
  - Tensile's tuned kernel genuinely outperforms the hipBLASLt heuristic's pick.
  - This is **expected** and exactly what tuning is for — the whole point is to find
    better kernels for shapes where the library's heuristic falls short.
  - hipblaslt-bench confirms the same gap, so it is a library kernel selection issue,
    not a measurement error.

### WARNING: Do NOT parse T/A ratios from the parallel tuning log

When `run_shapes.py` runs with `--parallel N` (N > 1), the log output from
multiple threads is **interleaved**.  The `>> Tensile=X | API=Y` summary lines
in the log can be **misattributed** — a summary line may appear under a different
shape's header because thread output is interleaved by the print lock.

**Always use the `.report.md` files** in `tunning_results/bf16/<shape_id>.report.md`
for per-shape results.  These are written atomically per thread and contain the
correct Tensile, API, and Bench TFLOPS for each shape.

Cross-GPU contention (Tensile running on other GPUs) does **not** affect
single-GPU GEMM performance on MI300.  Each GCD has independent HBM and compute.
Bad T/A numbers in the log are from misattributed interleaved output, not from
contention.

## GPU Contention Warning

Always verify GPUs are idle before benchmarking:

```bash
rocm-smi --showuse
```

GPUs at 99% usage will produce dramatically worse results (e.g., 1026 TFLOPS
instead of 1730 TFLOPS for the same kernel). Use `HIP_VISIBLE_DEVICES=<idle_gpu>`
to target an idle GPU.

## Verifying Which Kernel hipblaslt-bench Dispatches

Use `rocprofv3 --kernel-trace` to see the exact kernel binary dispatched:

```bash
HIP_VISIBLE_DEVICES=4 rocprofv3 --kernel-trace -d /tmp/bench_trace -- \
    /opt/rocm/bin/hipblaslt-bench -m 14336 -n 8192 -k 4096 \
    --precision bf16_r --compute_type f32_r --transA T --transB N -i 4 -j 2
```

Then parse the SQLite database:
```python
import sqlite3, os
db_dir = "/tmp/bench_trace"
for root, dirs, files in os.walk(db_dir):
    for f in files:
        if f.endswith(".db"):
            db_path = os.path.join(root, f)
            break

conn = sqlite3.connect(db_path)
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
# Find the guid from table names (format: rocpd_kernel_dispatch_<guid>)
for t in tables:
    if "kernel_dispatch" in t[0]:
        dispatch_table = t[0]
        guid = t[0].replace("rocpd_kernel_dispatch_", "")
    if "kernel_symbol" in t[0]:
        symbol_table = t[0]

knames = {}
for row in conn.execute(f"SELECT id, kernel_name FROM {symbol_table}"):
    knames[row[0]] = row[1]

for kid, start, end in conn.execute(
    f"SELECT kernel_id, start, end FROM {dispatch_table} ORDER BY start"
):
    name = knames.get(kid, "?")
    dur_us = (end - start) / 1000
    tag = "GEMM" if "Cijk" in name else "fill" if "fill" in name else "other"
    print(f"{tag:6s} dur={dur_us:8.1f} us  {name[:120]}")
```

## Inspecting Kernel Symbols in Bundled .co Files

The installed `.co` files are bundled code objects. To inspect kernel symbols:

```bash
# Extract the gfx950 ELF from the bundle
/opt/rocm/llvm/bin/clang-offload-bundler --unbundle \
    -type=o \
    -input=/opt/rocm/lib/hipblaslt/library/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx950.co \
    -output=/tmp/extracted.elf \
    -targets=hipv4-amdgcn-amd-amdhsa--gfx950

# List kernel symbols
/opt/rocm/llvm/bin/llvm-readelf -s /tmp/extracted.elf | grep "Cijk.*MT256x256"

# Search for specific kernel parameters
strings -n 50 /tmp/extracted.elf | grep "CMS.*SK3"
```

Note: `strings` on the raw bundled `.co` will fragment long symbol names.
Always extract first with `clang-offload-bundler --unbundle`.

## Reproducing Tensile TFLOPS via hipBLASLt API (test_hipblaslt_api.cpp)

### Critical: Data Initialization Must Match

Tensile initializes matrices with **small random integers** cast to BFloat16:

```cpp
// From tensilelite/client/include/DataInitialization.hpp
inline BFloat16 getValue<BFloat16, InitMode::Random>() {
    return static_cast<BFloat16>((tl_rand() % 7) - 3);  // [-3, 3]
}
```

Using constant data (e.g., `hipMemset(ptr, 0x3f, size)`) gives **10-15% inflated
TFLOPS** because the GPU can exploit cache/compression optimizations on uniform data.

Tensile init settings (from `ClientParameters.ini`):
```
init-a=Random    # bf16 integers in [-3, 3]
init-b=Random    # bf16 integers in [-3, 3]
init-c=Random    # bf16 integers in [-3, 3]
init-d=Zero      # all zeros
init-alpha=One   # 1.0f
init-beta=Zero   # 0.0f
```

### Matching Tensile Timing Exactly

To reproduce Tensile's reported TFLOPS through the hipBLASLt API:

1. **Random data init**: Fill A, B, C with random bf16 in [-3, 3], D with zeros
2. **Single event pair**: `hipEventRecord(start)` before all enqueues,
   `hipEventRecord(stop)` after all enqueues — NOT per-call events
3. **Average timing**: `total_time / num_enqueues` — NOT best-of-N
4. **Matching rotating buffer**: Use the same `rotating-buffer-size` that Tensile
   used (check `ClientParameters.ini` or `compute_rotating_buffer_mb()` output).
   For shapes where RotBuf>0, you must also cycle through separate allocations to
   reproduce cold-cache conditions.  If RotBuf=0, warm-cache is correct.
5. **30 warmups, 50 timed enqueues**

### Results (API vs Tensile vs hipblaslt-bench)

With all parameters aligned, the API driver matches Tensile within 1-2%:

| Shape              | API TF | Tensile TF | API/T | Bench TF | API/B |
|--------------------|-------:|----------:|------:|--------:|------:|
| attn_q+out 4096x   | 1546   |     1570  | 0.98x |    1570 | 0.98x |
| attn_qkv 6144x     | 1617   |     1604  | 1.01x |       — |     — |
| mlp_gate 14336x    | 1722   |     1734  | 0.99x |    1537 | 1.12x |
| mlp_gate_up 28672x | 1631   |     1620  | 1.01x |    1480 | 1.10x |
| mlp_down 4096x14k  | 1783   |     1798  | 0.99x |    1629 | 1.09x |

**Conclusion**: The installed tuned kernels deliver the Tensile-reported performance
when called through `hipblasLtMatmul` with matching cache conditions. The ~10% gap
vs `hipblaslt-bench` comes from `hipblaslt-bench`'s use of rotating buffers
(cold L2 cache), not from kernel quality.

`run_shapes.py` dynamically injects the same `RotatingBufferSize` into all three
drivers (Tensile YAML, `hipblaslt-bench --rotating`, `test_hipblaslt_api --rotating`)
via `compute_rotating_buffer_mb()`, so all measure under identical cache conditions.

### BiasDataTypeList Causes Two Client Passes

The Tensile client benchmarks each solution **twice** when `BiasDataTypeList`
contains multiple entries. In our template (`bf16_gemm_gfx950.yaml`):

```yaml
BiasDataTypeList: ['S','B']   # line 57 — FP32 and BF16 bias types
BiasTypeArgs: ['S','B']       # line 111 — same list for final benchmark
```

This produces two passes visible in the `.tensile.log`:

- Pass `0,0/1` — bias type `Float` (S)
- Pass `0,1/1` — bias type `BFloat16` (B)

The progress counter (e.g., `0,0/1,209/231`) resets to 0 when the second pass
begins, which can look like the client restarted. The `SkipSlowSolutionRatio: 0.7`
setting skips solutions that are >1/0.7 ≈ 1.43x slower than the current best during
warmup, reducing wasted time on poor candidates.

**Impact**: This doubles the client benchmarking time. If only BF16 bias is needed,
changing to `['B']` would halve client runtime. We keep both for broader library
coverage so the tuned kernels work regardless of bias precision.

### Tensile Kernel Name Abbreviations

Tensile encodes every tuning parameter into the kernel name as abbreviated
key-value pairs. Common abbreviations:

| Abbrev | Full Name | Example | Meaning |
|--------|-----------|---------|---------|
| `MT256x256x64` | MacroTile | 256×256×64 | Tile dimensions (M×N×K per workgroup) |
| `MI16x16x1` | MatrixInstruction | 16×16×1 | MFMA instruction shape |
| `AFC0`/`AFC1` | AssertFree0ElementMultiple | 0 or 1 | No alignment constraint on M. Higher values (e.g. AFC8) require M%8==0 but enable wider vector loads |
| `AG0` | AssertGrouped | 0 | Not a grouped GEMM (standard GEMM) |
| `GSU0` | GlobalSplitU | 0 | No K-splitting across workgroups; GSU>0 splits K reduction |
| `TLDS1` | TransposeLDS | 1 | LDS data layout is transposed |
| `DTL{A,B}1` | DirectToLds{A,B} | 1 | Load global mem directly into LDS (bypass registers) |
| `WG32_8_1` | WorkGroup | 32×8×1 | Workgroup thread layout |
| `SK3` | StreamK | 3 | Stream-K partitioning enabled for better load balance |
| `NTB0`/`NTB4` | NonTemporalB | 0 or 4 | 0=normal cached loads for B; 4=non-temporal (bypass cache coherency) |
| `PLR1` | PrefetchLocalRead | 1 | Prefetch from LDS enabled |
| `PGR2` | PrefetchGlobalRead | 2 | Double-buffer global prefetch |
| `LBSPPA`/`LBSPPB` | LdsBlockSizePerPadA/B | bytes | LDS padding to avoid bank conflicts |
| `LPA`/`LPB` | LdsPadA/B | elements | Additional LDS padding |
| `WGM8` | WorkGroupMapping | 8 | Workgroup mapping stride for cache locality |
| `WGMXCC8` | WorkGroupMappingXCC | 8 | Cross-XCC workgroup mapping (MI300-specific) |

`AFC0` and `AFC1` are functionally equivalent — both mean "no alignment
requirement." The distinction only matters at `AFC8` or higher where the kernel
assumes M is a multiple of that value for performance.

### MI4 → MI9 Expansion (MatrixInstruction Fields)

Tensile YAML templates specify MatrixInstruction as 4 elements (MI4):
`[mi_M, mi_N, mi_K, mi_B]` — e.g. `[16, 16, 32, 1]`.

`run_shapes.py` expands each MI4 into many MI9 tile configurations:
`[mi_M, mi_N, mi_K, mi_B, bm, tt0, tt1, wm, wn]`

| Index | Field | Name | Description |
|-------|-------|------|-------------|
| 0 | mi_M | MFMA M | MFMA instruction M dimension (16 or 32) |
| 1 | mi_N | MFMA N | MFMA instruction N dimension (16 or 32) |
| 2 | mi_K | MFMA K | MFMA instruction K dimension (32 for BF16) |
| 3 | mi_B | MFMA Blocks | Number of blocks per instruction (usually 1) |
| 4 | bm | Block Multiple | Power-of-2 multiplier from mi_B. Scales M tile. |
| 5 | tt0 | ThreadTile0 | Wave tile count in M direction (1–16) |
| 6 | tt1 | ThreadTile1 | Wave tile count in N direction (1–16) |
| 7 | wm | WaveGroup M | Waves in M direction (from combos: 1, 2, or 4) |
| 8 | wn | WaveGroup N | Waves in N direction (from combos: 4, 1, or 2) |

Macro tile dimensions:
- `MacroTile0 = mi_M × bm × tt0 × wm`
- `MacroTile1 = mi_N × tt1 × wn`

Example: `[16, 16, 32, 1, 1, 8, 8, 2, 2]`
→ MT0 = 16×1×8×2 = 256, MT1 = 16×8×2 = 256 → **MT256×256**
→ Kernel name: `MIWT8_8` (wave tile 8×8), `WG32_8_1` (32×8 threads = 2×2 waves)

Origami pruning (`--origami-top-n N`) scores all MI9 tiles and keeps the top N
predicted performers, dramatically reducing the search space.

### NonTemporalD Was Missing from Search Space (Critical Finding)

Analysis of 79 completed shapes (Llama-3.1-70B) revealed a systematic mismatch:

| Parameter | Tensile winners | API installed kernels |
|-----------|----------------|---------------------|
| **NTD0** (temporal D writes) | **78** | 2 |
| **NTD4** (non-temporal D writes) | **0** | **76** |
| **NTB0** (temporal B loads) | 73 | **78** |
| **NTB4** (non-temporal B loads) | 5 | 0 |

**Root cause**: `NonTemporalD` was not listed in the template's `ForkParameters`,
so all Tensile-generated solutions defaulted to `NTD0`. The production hipBLASLt
library uses `NTD4` on 76/78 shapes — bypassing cache coherency on output matrix
writes, which provides a consistent ~2-5% speedup for large GEMMs.

This explained why nearly all T/A ratios were 95-99% rather than ≥100%.

**Fix**: Added `- NonTemporalD: [0,4]` to `ForkParameters` in
`templates/bf16_gemm_gfx950.yaml`. This doubles the solution count but allows
Tensile to explore the same kernel configuration space as the production library.

Note: `NonTemporalB: [0,4]` was already enabled in the template. The API library
consistently uses `NTB0`, so this hasn't been a source of mismatch.

**Update**: After adding NTD4 to the search space and re-running
`mlp_gate_up mbs=2 fwd`, Tensile's winner was still NTD0 (1322 TFLOPS), not
NTD4. Under Tensile's cold-cache (rotating buffer) measurement, NTD4 provides
no benefit — non-temporal D writes bypass cache, which doesn't help when the
cache is already cold. The API bench (warm cache) measures 1404 TFLOPS with
NTD4 because bypassing cache avoids polluting L2 with output data. This means
the T/A gap for NTD is a **measurement artifact**, not a real kernel quality
difference in production workloads.

### DirectToLds (DTL) Analysis

Both Tensile and the installed API library strongly prefer `DTLA=1 DTLB=1`
(direct global-to-LDS loads for both matrices A and B):

| Config | Tensile winners | API installed |
|--------|----------------|--------------|
| DTLA=1, DTLB=1 | 78 / 79 | 76 / 79 |
| DTLA=0, DTLB=0 | 0 | 2 (small attn_k shapes) |
| Custom kernel | 1 | 1 |

DirectToLds bypasses registers and loads data directly into LDS, reducing
register pressure and latency. It is overwhelmingly preferred for all shapes
except very small ones (N=1024) where the tile size doesn't benefit from it.

`TransposeLDS` is also well-aligned: TLDS=1 wins 51 shapes, TLDS=0 wins 27,
identical split between Tensile and API. Neither parameter is a source of
T/A mismatch.

### StaggerU — Memory Channel Conflict Avoidance

StaggerU offsets each workgroup's starting position along the K dimension to
avoid DRAM bank/channel conflicts when matrix dimensions are large powers of 2.
Without it, all workgroups load from addresses separated by exact power-of-2
strides, causing hot memory channels and TLB thrashing.

- `StaggerU`: max stagger "clicks" (0=disabled, 4=up to 4 offsets per WG)
- `StaggerUStride`: bytes per click (256 = one MI300X memory channel width)
- `StaggerUMapping`: which WG dimension determines offset (0=wg0, 1=wg1)

The production hipBLASLt library (`gfx950_Cijk_*_UserArgs.yaml`) uses
`StaggerUStride` values of 0, 64, 128, and 256 across different kernels.
Our templates now match: `StaggerU: [0, 4]`, `StaggerUStride: [0, 64, 128, 256]`.

Previously we had `StaggerU: [0]` (disabled) with `StaggerUStride: [16]` — this
missed the entire staggering optimization. For LLM GEMMs with dimensions like
8192, 16384, 32768, this could be significant.

### Tensile Client vs hipblasLtMatmul Dispatch Gap (~7%)

**Critical finding**: The exact same kernel binary shows ~7% lower TFLOPS when
benchmarked through the Tensile client vs `hipblasLtMatmul`. Verified on
`mlp_gate_up mbs=2 fwd` (M=57344, N=16384, K=8192):

| Dispatch path | Latency | TFLOPS |
|---------------|---------|--------|
| Tensile client (`hipExtModuleLaunchKernel`) | 11,737 µs | 1312 |
| API bench (`hipblasLtMatmul`) | 10,966 µs | 1404 |

Conditions were identical: same kernel name, same warm-cache (rotating=0),
same data init (random [-3,3] BF16), same event-pair timing around 50 enqueues,
same assembler flags and code object version.

The ~15 µs/call gap comes from the dispatch path difference. This means
**T/A ratios of 93-95% do not indicate a worse kernel** — they reflect a
systematic measurement bias in the Tensile client.

See `tunning_results/archive/tensile-vs-api-dispatch-gap/` for full evidence.

### Why attn_k (M=1024) Is an Outlier

The smallest shape `attn_k` (M8192_N1024_K4096) shows 785 TF via API vs 1223 TF
from Tensile. This is likely due to the hipBLASLt heuristic selecting a different
(suboptimal) kernel than Tensile's tuned winner for this shape, or the AOT-compiled
variant behaving differently at this small tile occupancy.

## Production Library Kernel Parameter Distribution (gfx950)

Analyzed via `analyze_production_library.py`. Total kernels: **6,325**.

```
Field                          Unique  Values (% of kernels)
====================================================================================================
MatrixInstruction                   9  [16,16,32,1](67%)  [16,16,128,1](13%)  [32,32,16,1](11%)  [32,32,64,1](5%)  [16,16,4,1](3%)  +4 more
DepthU                              8  64(41%)  128(31%)  32(15%)  256(9%)  16(2%)  512(2%)  +2 more
TransposeLDS                        3  1(66%)  2(21%)  0(14%)
DirectToLdsA                        2  false(55%)  true(45%)
DirectToLdsB                        2  false(55%)  true(45%)
NonTemporalA                        8  0(36%)  4(15%)  1(14%)  2(11%)  3(11%)  5(5%)  +2 more
NonTemporalB                        8  0(40%)  1(15%)  2(12%)  3(12%)  4(12%)  5(3%)  +2 more
NonTemporalD                        8  4(26%)  0(24%)  5(10%)  2(9%)  7(8%)  3(8%)  +2 more
StaggerU                            4  0(56%)  8(27%)  16(16%)  32(0%)
StaggerUStride                      7  0(56%)  128(17%)  256(13%)  512(12%)  64(1%)  1024(1%)  +1 more
StaggerUMapping                     2  1(60%)  0(40%)
WorkGroupMappingXCC                 7  1(27%)  8(25%)  2(15%)  4(15%)  16(10%)  32(7%)  +1 more
GlobalSplitU                        3  0(99%)  1(1%)  6(0%)
GlobalSplitUAlgorithm               1  MultipleBuffer(100%)
StreamK                             2  3(99%)  0(1%)
SourceSwap                          4  1(90%)  0(10%)
PreloadKernArgs                     2  true(100%)
ClusterLocalRead                    4  0(61%)  1(38%)
PrefetchGlobalRead                  5  2(94%)  1(5%)  4(1%)  3(1%)
PrefetchLocalRead                   2  1(83%)  0(17%)
ScheduleIterAlg                     2  3(100%)
1LDSBuffer                  (not in scan — see separate analysis: 1=70%, 0=30%)
```

### Key Gaps Between Our Template and Production

| Parameter | Our template | Production | Impact |
|-----------|-------------|-----------|--------|
| DepthU | [32, 64] | 8 values (16–1024) | Missing 128, 256 (40% of prod kernels) |
| StaggerU | [0, 4] | [0, 8, 16, 32] | Prod uses 8/16, not 4 |
| StaggerUStride | [0, 64, 128, 256] | +512, 1024, 2048 | Missing larger strides |
| NonTemporalA | not explored | 8 values (0–7) | 64% of prod uses NTA>0 |
| NonTemporalD | [0, 4] | 8 values (0–7) | Only covers 50% of prod |
| WorkGroupMappingXCC | [1, 8] | [1,2,4,8,16,32,64] | Missing 73% of prod values |
| MatrixInstruction | 2 types | 9 types | Missing gfx950 MI128/MI64 |
| GlobalSplitUAlgorithm | MultipleBufferSingleKernel | MultipleBuffer (100%) | Mismatch |
| 1LDSBuffer | [-1] (auto→0) | 1 (70%) | Defaulting to wrong value |
| TransposeLDS | [0, 1] | [0, 1, 2] | Missing TLDS=2 (21% of prod) |

Estimated theoretical search space: ~9×10²⁰ combinations (most invalid).
The production library's 6,325 kernels represent a heavily curated subset.
