# hipBLASLt Tuning Helper — BF16 Dense GEMM

Tensile-based tuning scripts for **BF16 standard GEMM** shapes derived from
dense LLM architectures (Llama-2, Llama-3.1, Qwen-2.5, Mistral-7B).

## Prerequisites

- ROCm with hipBLASLt installed (`/opt/rocm/bin/hipblaslt-bench`)
- hipBLASLt source tree with TensileLite built (set `TENSILE_WD` env var)
- Python 3.8+

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
├── config.py                  # Model configs and shape generation
├── run_shapes.py              # Main tuning + comparison script
└── templates/
    └── bf16_gemm_gfx950.yaml  # Tensile YAML template (BF16, gfx950)
```

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
- **TensileLite path**: Set `TENSILE_WD` environment variable
- **Output**: Set `--output-dir` (default: `tunning_results/`)
