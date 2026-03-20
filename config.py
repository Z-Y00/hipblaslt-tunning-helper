"""Shape configurations for hipBLASLt BF16 GEMM tuning.

Model configs originally from turbo/benchmark/ops/config.py, inlined here
so we can extend the search space independently.

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

###############################################################################
# Batch sizes — per-model, derived from BF16 training benchmarks ±2.
#
# Source tables (Primus / Primus-TT BF16 rows):
#   Llama-2-7B       MBS=10  seqlen=4096
#   Llama-2-70B      MBS=17  seqlen=4096
#   Llama-3.1-8B     MBS=4   seqlen=8192  (Primus-TT: MBS=6)
#   Llama-3.1-70B    MBS=1   seqlen=8192  (Primus-TT: MBS=3, Llama-3.3-70B: MBS=6)
#   Qwen2.5-7B       MBS=16  seqlen=2048
#   Qwen2.5-72B      MBS=16  seqlen=2048
#   DeepSeek-V2-Lite MBS=12  seqlen=4096
#   DeepSeek-V3      MBS=8   seqlen=4096  (proxy)
#   DeepSeek-V3-16B  MBS=13  seqlen=4096  (Primus-TT, config TBD)
#
# Models with multiple BF16 entries use the union of ±2 ranges.
# Llama-3.3-70B shares architecture with Llama-3.1-70B (same shapes).
###############################################################################

BATCH_SIZE_LIST = list(range(1, 10))  # fallback default

###############################################################################
# Dense Model Configurations
###############################################################################

DenseModelConfigs = {
    # https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
    # BF16 MBS=1 (Primus) + MBS=3 (Primus-TT) + MBS=6 (Llama-3.3-70B, same arch)
    # Prioritized: test BS=4 first
    "Llama-3.1-70B": {
        "seqlen": 8192,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "vocab_size": 128256,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "batch_sizes": [4, 1, 2, 3, 5, 6, 7, 8], # Lorri, for quick check for regression case
    },
    # https://huggingface.co/meta-llama/Llama-2-7b/blob/main/config.json
    # BF16 MBS=10
    "Llama-2-7B": {
        "seqlen": 4096,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "batch_sizes": [8, 9, 10, 11, 12],
    },
    # https://huggingface.co/meta-llama/Llama-2-70b/blob/main/config.json
    # BF16 MBS=17
    "Llama-2-70B": {
        "seqlen": 4096,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "vocab_size": 32000,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "batch_sizes": [15, 16, 17, 18, 19],
    },
    # https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
    # BF16 MBS=4 (Primus) + MBS=6 (Primus-TT)
    "Llama-3.1-8B": {
        "seqlen": 8192,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "batch_sizes": [2, 3, 4, 5, 6, 7, 8],
    },
    # https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
    # Not in BF16 benchmark table — keeping conservative range
    "Llama-3.1-405B": {
        "seqlen": 8192,
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "vocab_size": 128256,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "batch_sizes": [1, 2], # lorri, cap as 1-2
        # https://github.com/AMD-AGI/Primus/blob/ace9a5e522b490840c8c0c66c90f6ce284123889/examples/torchtitan/configs/MI355X/llama3.1_405B-BF16-pretrain.yaml#L19,
        # https://github.com/AMD-AGI/Primus/blob/ace9a5e522b490840c8c0c66c90f6ce284123889/examples/megatron/configs/MI355X/llama3.1_405B-BF16-pretrain.yaml#L20
    },
    # https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/file/view/master/config.json
    # BF16 MBS=16 at seqlen=2048
    "Qwen2.5-7B": {
        "seqlen": 2048,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "vocab_size": 152064,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "batch_sizes": [14, 15, 16, 17, 18],
    },
    # https://modelscope.cn/models/Qwen/Qwen2.5-72B-Instruct
    # BF16 MBS=16 at seqlen=2048
    "Qwen2.5-72B": {
        "seqlen": 2048,
        "hidden_size": 8192,
        "intermediate_size": 29568,
        "vocab_size": 152064,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "batch_sizes": [14, 15, 16, 17, 18],
    },
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/config.json
    # Not in BF16 benchmark table (Mixtral-8x7B is MoE, different model)
    "Mistral-7B": {
        "seqlen": 4096,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "vocab_size": 32000,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "batch_sizes": [2, 3, 4, 5, 6],
    },
    # https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json
    # MoE model (15.7B total, 2.4B active) with MLA attention.
    # Dense FFN: intermediate_size=10944 (first_k_dense_replace=1 layer).
    # MLA: qk_nope=128, qk_rope=64, v=128, kv_lora_rank=512.
    # Attention shapes skipped — MLA needs real profiling to determine GEMM dims.
    # BF16 MBS=12
    "DeepSeek-V2-Lite": {
        "seqlen": 4096,
        "hidden_size": 2048,
        "intermediate_size": 10944,
        "vocab_size": 102400,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "mla": True,
        "batch_sizes": [10, 11, 12, 13, 14],
    },
    # https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
    # MoE model (671B total, 37B active) with MLA attention.
    # Dense FFN (first 3 of 61 layers): intermediate_size=18432.
    # Shared expert FFN (58 MoE layers): 1 × moe_intermediate_size=2048.
    # MLA: q_lora_rank=1536, kv_lora_rank=512, qk_nope=128, qk_rope=64, v=128.
    # Attention shapes skipped — MLA needs real profiling to determine GEMM dims.
    # MLP shapes (gate_up, gate, down) are accurate for the dense FFN layers.
    # BF16 MBS=8 (proxy)
    "DeepSeek-V3": {
        "seqlen": 4096,
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "vocab_size": 129280,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "head_dim": 128,
        "mla": True,
        "batch_sizes": [6, 7, 8, 9, 10],
    },
}


def gen_gemm_test_cases(model_config):
    """Generate GEMM test cases from model config (turbo-compatible)."""
    seq = model_config["seqlen"]
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]
    num_attention_heads = model_config["num_attention_heads"]
    num_key_value_heads = model_config["num_key_value_heads"]
    head_dim = model_config["head_dim"]

    return [
        [seq, int((num_attention_heads + 2 * num_key_value_heads) * head_dim), hidden_size],
        [seq, hidden_size, hidden_size],
        [seq, int(2 * intermediate_size), hidden_size],
        [seq, hidden_size, intermediate_size],
    ]


def gen_gemm_test_cases_extended(cfg, model_name=None):
    """Generate GEMM shapes for both fused and split projections, plus lm_head.

    Returns list of (layer_name, M_base, N, K) tuples.
    Skips attn_v (identical to attn_k) and mlp_up (identical to mlp_gate).
    For MLA models (mla=True), attention shapes are skipped — those require
    real profiling to determine the actual GEMM dimensions.
    """
    seq = cfg["seqlen"]
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    is_mla = cfg.get("mla", False)

    shapes = []

    if is_mla:
        print(f"  [TODO] {model_name or '?'}: skipping attention shapes — "
              f"MLA projections need real profiling to determine GEMM dims")
    else:
        shapes.append(("attn_qkv",    seq, (n_heads + 2 * n_kv) * hd, h))
        shapes.append(("attn_q",      seq, n_heads * hd,               h))
        shapes.append(("attn_k",      seq, n_kv * hd,                  h))
        shapes.append(("attn_out",    seq, h,                           h))

    # -- Fused MLP --
    shapes.append(("mlp_gate_up", seq, 2 * inter,                   h))
    # -- Split MLP --
    shapes.append(("mlp_gate",    seq, inter,                        h))
    shapes.append(("mlp_down",    seq, h,                            inter))

    # -- Vocab projection (lm_head) --
    vocab = cfg.get("vocab_size")
    if vocab:
        shapes.append(("lm_head",  seq, vocab,                        h))

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
        cases = gen_gemm_test_cases_extended(cfg, model_name=name)
        bs_list = cfg.get("batch_sizes", BATCH_SIZE_LIST)
        for mbs in bs_list:
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
