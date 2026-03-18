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

PyTorch internally swaps A and B when calling column-major BLAS, so the
BLAS m/n are always swapped relative to PyTorch's M/N:

  Phase  | PyTorch            | BLAS m | n | k | TransA | TransB
  -------+--------------------+--------+---+---+--------+-------
  fwd    | a(M,K) @ b(N,K).T  |   N    | M | K |   T    |   N
  grad_a | dC(M,N) @ b(N,K)   |   K    | M | N |   N    |   N
  grad_b | dC(M,N).T @ a(M,K) |   K    | N | M |   N    |   T

The shape dicts below use PyTorch convention (M=tokens, N=output_features).
run_shapes.py swaps M<->N when generating Tensile YAML and hipblaslt-bench
commands to match the BLAS convention.
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

BATCH_SIZE_LIST = list(range(1,10)) # _turbo_cfg.BATCH_SIZE_LIST
BATCH_SIZE_LIST = [1]
DenseModelConfigs = _turbo_cfg.DenseModelConfigs
gen_gemm_test_cases = _turbo_cfg.gen_gemm_test_cases

# _VOCAB_SIZES = {
#     "Llama-2-7B": 32000,
#     "Llama-2-70B": 32000,
#     "Llama-3.1-8B": 128256,
#     "Llama-3.1-405B": 128256,
#     "Qwen2.5-7B": 152064,
#     "Qwen2.5-72B": 152064,
#     "Mistral-7B": 32000,
# }

# for _name, _vocab in _VOCAB_SIZES.items():
#     if _name in DenseModelConfigs:
#         DenseModelConfigs[_name]["vocab_size"] = _vocab


def gen_gemm_test_cases_extended(cfg):
    """Generate GEMM shapes for both fused and split projections, plus lm_head.

    Returns list of (layer_name, M_base, N, K) tuples.
    Skips attn_v (identical to attn_k) and mlp_up (identical to mlp_gate).
    """
    seq = cfg["seqlen"]
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    # vocab = cfg.get("vocab_size")

    shapes = []
    # -- Fused attention (e.g. vLLM, FasterTransformer) --
    shapes.append(("attn_qkv",    seq, (n_heads + 2 * n_kv) * hd, h))
    # -- Split attention (e.g. torchtitan) --
    shapes.append(("attn_q",      seq, n_heads * hd,               h))
    shapes.append(("attn_k",      seq, n_kv * hd,                  h))
    shapes.append(("attn_out",    seq, h,                           h))
    # -- Fused MLP --
    shapes.append(("mlp_gate_up", seq, 2 * inter,                   h))
    # -- Split MLP --
    shapes.append(("mlp_gate",    seq, inter,                        h))
    shapes.append(("mlp_down",    seq, h,                            inter))
    # -- LM head --
    # if vocab: # lorri, drop due to too long
        # shapes.append(("lm_head", seq, vocab,                        h))

    return shapes


def _fwd_bwd_shapes(model, layer, mbs, M, N, K):
    """Return up to three shapes for forward + backward of one GEMM layer.

    Each entry is a dict with (model, layer, mbs, M, N, K, trans_a, trans_b, phase).
    """
    base = dict(model=model, layer=layer, mbs=mbs)
    return [
        {**base, "phase": "fwd",    "M": M, "N": N, "K": K,
         "trans_a": True,  "trans_b": False},
        {**base, "phase": "grad_a", "M": M, "N": K, "K": N,
         "trans_a": False, "trans_b": False},
        {**base, "phase": "grad_b", "M": N, "N": K, "K": M,
         "trans_a": False, "trans_b": True},
    ]


def gen_all_shapes(model_filter=None, include_bwd=True):
    """Generate GEMM shape tuples for tuning.

    Covers both fused (QKV, gate+up) and split (Q, K, gate) projections
    plus lm_head.  Deduplicates shapes that have identical (M, N, K,
    trans_a, trans_b) within the same model+mbs, merging layer names.
    """
    shapes = []
    for name, cfg in DenseModelConfigs.items():
        if model_filter and model_filter.lower() not in name.lower():
            continue
        cases = gen_gemm_test_cases_extended(cfg)
        for mbs in BATCH_SIZE_LIST:
            seen = {}
            raw = []
            for layer, M_base, N, K in cases:
                M = M_base * mbs
                if include_bwd:
                    entries = _fwd_bwd_shapes(name, layer, mbs, M, N, K)
                else:
                    entries = [{
                        "model": name, "layer": layer, "mbs": mbs,
                        "M": M, "N": N, "K": K,
                        "trans_a": True, "trans_b": False, "phase": "fwd",
                    }]
                for e in entries:
                    dedup_key = (e["M"], e["N"], e["K"],
                                 e["trans_a"], e["trans_b"], e["phase"])
                    if dedup_key in seen:
                        prev = raw[seen[dedup_key]]
                        if layer not in prev["layer"]:
                            prev["layer"] += "+" + layer
                    else:
                        seen[dedup_key] = len(raw)
                        raw.append(e)
            shapes.extend(raw)
    return shapes
