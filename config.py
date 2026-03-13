"""Shape configurations for hipBLASLt BF16 GEMM tuning.

Re-exports DenseModelConfigs and BATCH_SIZE_LIST from the turbo submodule
so that shape definitions stay in sync with upstream benchmarks.

Shapes are standard (non-grouped) GEMMs derived from dense LLM layers.
The benchmark convention (bench_gemm_torch.py) is:
  A (M×K) @ B (N×K)^T  →  BLAS column-major: TransposeA=T, TransposeB=N
"""

import sys
from pathlib import Path

_TURBO_OPS = str(Path(__file__).resolve().parent / "turbo" / "benchmark" / "ops")
if _TURBO_OPS not in sys.path:
    sys.path.insert(0, _TURBO_OPS)

from config import (  # noqa: E402
    BATCH_SIZE_LIST,
    DenseModelConfigs,
    gen_gemm_test_cases,
)

# Transpose convention from the turbo benchmark (bench_gemm_torch.py):
#   a = randn(M, K);  b = randn(N, K);  out = a @ b.T
# In BLAS column-major terms:
#   Row-major (M,K) = col-major (K,M) → TransposeA = T
#   Row-major (N,K) = col-major (K,N) → TransposeB = N
TRANSPOSE_A = True   # "T" in hipblaslt-bench, 1 in Tensile YAML
TRANSPOSE_B = False  # "N" in hipblaslt-bench, 0 in Tensile YAML


def gen_all_shapes(model_filter=None):
    """Generate all (model, layer, mbs, M, N, K, trans_a, trans_b) tuples."""
    shapes = []
    for name, cfg in DenseModelConfigs.items():
        if model_filter and model_filter.lower() not in name.lower():
            continue
        cases = gen_gemm_test_cases(cfg)
        for mbs in BATCH_SIZE_LIST:
            for i, (M_base, N, K) in enumerate(cases):
                M = M_base * mbs
                layers = ["attn_qkv", "attn_out", "mlp_gate_up", "mlp_down"]
                shapes.append({
                    "model": name,
                    "layer": layers[i],
                    "mbs": mbs,
                    "M": M,
                    "N": N,
                    "K": K,
                    "trans_a": TRANSPOSE_A,
                    "trans_b": TRANSPOSE_B,
                })
    return shapes
