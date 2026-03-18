# hipBLASLt Tuning Helper — BF16 Dense GEMM

Tensile-based tuning scripts for **BF16 standard GEMM** shapes derived from
dense LLM architectures (Llama-2, Llama-3.1, Qwen-2.5, Mistral-7B).

## Prerequisites

- ROCm with hipBLASLt installed (`/opt/rocm/bin/hipblaslt-bench`)
- Python 3.8+

## Setup

### Quick clone (~10 MB, recommended)

Clone without submodules first, then selectively init only what's needed:

```bash
git clone https://github.com/Z-Y00/hipblaslt-tunning-helper.git
cd hipblaslt-tunning-helper

# Init turbo submodule (~3 MB)
git submodule update --init turbo
```

The other submodules are **not needed** for normal use:
- **`origami`** — installed from `tmp_rebuild/rocm-libraries` by `init_build.sh`
- **`hipblaslt`** (7 GB) — `rebuild_hipblaslt.sh` clones `rocm-libraries` into
  `tmp_rebuild/` and `init_build.sh` applies patches from `patches/` automatically

> **Warning:** Do NOT run `git sparse-checkout` in the main repo directory.
> Sparse checkout is used only inside `tmp_rebuild/rocm-libraries` (handled
> automatically by `init_build.sh`). Running it on the main repo will hide
> tracked files like `templates/`, `patches/`, etc.

### Build

```bash
# Clone rocm-libraries (sparse), apply patches, build TensileLite client, install Origami
./init_build.sh

# (After tuning) Merge tuned kernels and rebuild hipBLASLt
./rebuild_hipblaslt.sh
```

## Quick start

```bash
# List all 84 shapes (7 models × 4 layers × 3 batch sizes)
python3 run_shapes.py --list

# Generate YAML configs only (no GPU needed)
python3 run_shapes.py --gen-only

# Full run: Tensile tuning + hipblaslt-bench comparison
python3 run_shapes.py --run

# Filter to one model, limit shapes
python3 run_shapes.py --run --filter "Llama-3.1-8B" --max-shapes 4

# Skip Tensile, only run hipblaslt-bench baseline
python3 run_shapes.py --run --skip-tensile

# Multi-GPU parallel
python3 run_shapes.py --run --parallel 8
python3 run_shapes.py --run --gpu-list 0,2,4,6

# Use Origami to prune MI configs (top 30 tiles per shape)
python3 run_shapes.py --run --origami-top-n 30
```

## Full tuning run (nohup)

Use `launch_tuning.sh` for long-running tuning jobs.  It runs under `nohup` so
it survives terminal disconnects, and saves output to a timestamped log.

```bash
# All models, fwd + bwd, 8 GPUs, origami pruning
./launch_tuning.sh

# Filter to one model
./launch_tuning.sh --filter Llama-3.1-8B

# Forward only
./launch_tuning.sh --fwd-only

# Combine options
./launch_tuning.sh --filter Llama-3.1-8B --fwd-only --max-shapes 4

# Follow the log
tail -f tunning_results/launch_tuning_*.log
```

Tuning supports **resume**: if a shape already has a valid `.report.md`, it is
skipped automatically.  Use `--force` to re-tune everything.

## Shapes

Shapes come from `config.py` which defines 7 dense LLM models.  For each
model, 4 GEMM layers are extracted:

| Layer | M | N | K |
|-------|---|---|---|
| `attn_qkv` | seqlen × MBS | (n_heads + 2×n_kv) × head_dim | hidden_size |
| `attn_out` | seqlen × MBS | hidden_size | hidden_size |
| `mlp_gate_up` | seqlen × MBS | 2 × intermediate_size | hidden_size |
| `mlp_down` | seqlen × MBS | hidden_size | intermediate_size |

Batch sizes (MBS): 1, 2, 4

## Structure

```
├── README.md
├── config.py                  # Model definitions + shape generation
├── run_shapes.py              # Main tuning + comparison script
├── launch_tuning.sh           # nohup wrapper for multi-GPU tuning
├── init_build.sh              # Build TensileLite client, apply patches, install Origami
├── rebuild_hipblaslt.sh       # Clone rocm-libraries + build hipBLASLt from source
├── test_hipblaslt_api.cpp     # API bench driver (Tensile-aligned timing)
├── patches/                   # Tensile patches (applied by init_build.sh)
├── templates/
│   ├── bf16_gemm_gfx950.yaml  # Tensile YAML template (BF16, gfx950)
│   └── f8_gemm_gfx950.yaml   # Tensile YAML template (FP8, gfx950)
├── turbo/                     # Submodule: AMD-AGI/Primus-Turbo (benchmark configs)
├── hipblaslt/                 # Submodule: Z-Y00/hipBLASLt (optional, for patch dev)
└── origami/                   # Submodule: ROCm/rocm-libraries (sparse: shared/origami only)
```

### Submodules

- **turbo** (`AMD-AGI/Primus-Turbo`): Provides `DenseModelConfigs` and shape
  generation so tuning shapes stay in sync with upstream benchmarks.
- **hipblaslt** (`Z-Y00/hipBLASLt`, branch `cosmo-dev`): Patched TensileLite
  with fixes for TFLOPS reporting, rotating buffer overflow, hsaco filename
  alias, empty assembly guards, and thread-local RNG for data init.
- **origami** (`ROCm/rocm-libraries`, sparse checkout of `shared/origami/`):
  Analytical tile-selection model that predicts GEMM kernel performance without
  benchmarking.  Used with `--origami-top-n N` to prune the MI config search
  space.

## Transpose convention

The transpose settings are derived from the turbo benchmark (`bench_gemm_torch.py`)
which computes `A(M×K) @ B(N×K).T`.  In BLAS column-major terms this maps to:

| | Benchmark | hipblaslt-bench | Tensile YAML |
|-|-----------|-----------------|--------------|
| **A** | `(M, K)` row-major | `--transA T` | `TransposeA: 1` |
| **B** | `(N, K)` row-major, transposed | `--transB N` | `TransposeB: 0` |

The script reads `trans_a` / `trans_b` from each shape (set in `config.py`) and
overrides the template YAML's `TransposeA` / `TransposeB` values accordingly.
This ensures the tuning always matches the benchmark convention regardless of
what the template file contains.

## How it works

For each shape, the script:

1. **Generates a Tensile YAML** from the template, expanding MI4 entries to
   MI9 (9-element MatrixInstruction) sized for the problem dimensions,
   and auto-computing `RotatingBufferSize` to ensure cold-cache benchmarking.

2. **Runs TensileLite** to compile assembly kernels and benchmark them,
   producing a `CSVWinner.csv` with the best kernel per problem size.

3. **Runs `hipblaslt-bench`** with the same shape as a baseline comparison
   (uses the pre-built production library kernels).

4. **Reports** per-shape markdown files and a summary CSV with TFLOPS
   comparison between the Tensile-tuned kernel and production hipblaslt-bench.

## Configuration

- **Template**: Override with `--template path/to/custom.yaml`
- **TensileLite path**: Prefers `tmp_rebuild/rocm-libraries/.../tensilelite` (build-tree),
  falls back to `hipblaslt/tensilelite` submodule. Override with `TENSILE_WD` env var.
- **Output**: Set `--output-dir` (default: `tunning_results/`)
- **Cleanup**: Post-tuning cleanup deletes `.o`/`.s`/`.co`/`.hsaco` and compresses each
  shape dir to `.tar.zst` (~2 MB/shape vs ~1.9 GB). Disable with `--no-cleanup`.
