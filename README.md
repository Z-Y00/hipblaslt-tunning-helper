# hipBLASLt Tuning Helper — BF16 Dense GEMM

Tensile-based tuning scripts for **BF16 standard GEMM** shapes derived from
dense LLM architectures (Llama-2, Llama-3.1, Qwen-2.5, Mistral-7B).

## Prerequisites

- ROCm with hipBLASLt installed (`/opt/rocm/bin/hipblaslt-bench`)
- Python 3.8+

## Setup

```bash
git clone --recursive https://github.com/Z-Y00/hipblaslt-tunning-helper.git
cd hipblaslt-tunning-helper

# The origami submodule uses sparse checkout (only shared/origami/ from rocm-libraries).
# After clone, enable sparse checkout for it:
cd origami
git sparse-checkout init --cone
git sparse-checkout set shared/origami
cd ..

# (Optional) Install Origami for analytical search-space pruning
cd origami/shared/origami/python
CMAKE_PREFIX_PATH=/opt/rocm CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ pip install -e .
cd ../../../..

# Build TensileLite client (needed for Tensile benchmarking)
cd hipblaslt/tensilelite
pip3 install invoke
invoke build-client
cd ../..
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
├── config.py                  # Imports from turbo submodule + shape generation
├── run_shapes.py              # Main tuning + comparison script
├── templates/
│   └── bf16_gemm_gfx950.yaml  # Tensile YAML template (BF16, gfx950)
├── turbo/                     # Submodule: AMD-AGI/Primus-Turbo (benchmark configs)
├── hipblaslt/                 # Submodule: Z-Y00/hipBLASLt @ cosmo-dev (patched TensileLite)
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
- **TensileLite path**: Defaults to `hipblaslt/tensilelite` submodule.
  Override with `TENSILE_WD` environment variable.
- **Output**: Set `--output-dir` (default: `tunning_results/`)
