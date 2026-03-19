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

### Why attn_k (M=1024) Is an Outlier

The smallest shape `attn_k` (M8192_N1024_K4096) shows 785 TF via API vs 1223 TF
from Tensile. This is likely due to the hipBLASLt heuristic selecting a different
(suboptimal) kernel than Tensile's tuned winner for this shape, or the AOT-compiled
variant behaving differently at this small tile occupancy.
