"""Shape configurations for hipBLASLt BF16 GEMM tuning.

Re-exports DenseModelConfigs and BATCH_SIZE_LIST from the turbo submodule
so that shape definitions stay in sync with upstream benchmarks.

The benchmark (bench_gemm_torch.py) stores:
    a = randn(M, K)   # row-major
    b = randn(N, K)   # row-major
    out = a @ b.T      # forward

Backward of out = a @ b.T produces two more GEMMs:
    grad_a = grad_out @ b         # (M,N) @ (N,K)
    grad_b = grad_out.T @ a       # (N,M) @ (M,K)

Mapping to BLAS column-major (row-major X(R,C) = col-major (C,R)):

  Phase  | PyTorch            | BLAS m | n | k | TransA | TransB
  -------+--------------------+--------+---+---+--------+-------
  fwd    | a(M,K) @ b(N,K).T  |   M    | N | K |   T    |   N
  grad_a | dC(M,N) @ b(N,K)   |   M    | K | N |   T    |   T
  grad_b | dC(M,N).T @ a(M,K) |   N    | K | M |   N    |   T
"""

import importlib.util
import sys
from pathlib import Path

_TURBO_CONFIG = str(
    Path(__file__).resolve().parent / "turbo" / "benchmark" / "ops" / "config.py"
)
_spec = importlib.util.spec_from_file_location("turbo_ops_config", _TURBO_CONFIG)
_turbo_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_turbo_cfg)

BATCH_SIZE_LIST = _turbo_cfg.BATCH_SIZE_LIST
DenseModelConfigs = _turbo_cfg.DenseModelConfigs
gen_gemm_test_cases = _turbo_cfg.gen_gemm_test_cases


def _fwd_bwd_shapes(model, layer, mbs, M, N, K):
    """Return up to three shapes for forward + backward of one GEMM layer.

    Each entry is a dict with (model, layer, mbs, M, N, K, trans_a, trans_b, phase).
    """
    base = dict(model=model, layer=layer, mbs=mbs)
    return [
        {**base, "phase": "fwd",    "M": M, "N": N, "K": K,
         "trans_a": True,  "trans_b": False},
        {**base, "phase": "grad_a", "M": M, "N": K, "K": N,
         "trans_a": True,  "trans_b": True},
        {**base, "phase": "grad_b", "M": N, "N": K, "K": M,
         "trans_a": False, "trans_b": True},
    ]


def gen_all_shapes(model_filter=None, include_bwd=True):
    """Generate GEMM shape tuples for tuning.

    By default every forward shape is followed by the two backward GEMMs
    (grad_a, grad_b) with the correct transpose + dimension mapping derived
    from the benchmark convention.  Pass *include_bwd=False* for forward only.
    """
    shapes = []
    for name, cfg in DenseModelConfigs.items():
        if model_filter and model_filter.lower() not in name.lower():
            continue
        cases = gen_gemm_test_cases(cfg)
        for mbs in BATCH_SIZE_LIST:
            for i, (M_base, N, K) in enumerate(cases):
                M = M_base * mbs
                layer = ["attn_qkv", "attn_out", "mlp_gate_up", "mlp_down"][i]
                if include_bwd:
                    shapes.extend(_fwd_bwd_shapes(name, layer, mbs, M, N, K))
                else:
                    shapes.append({
                        "model": name,
                        "layer": layer,
                        "mbs": mbs,
                        "M": M,
                        "N": N,
                        "K": K,
                        "trans_a": True,
                        "trans_b": False,
                        "phase": "fwd",
                    })
    return shapes
