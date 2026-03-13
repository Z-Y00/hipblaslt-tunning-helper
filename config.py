"""Shape configurations for hipBLASLt BF16 GEMM tuning.

Shapes are derived from dense LLM model architectures (attention QKV/out,
MLP gate+up/down).  Each shape is a standard (non-grouped) GEMM:
  D = alpha * A(M×K) @ B(K×N)^T + beta * C
"""

# Dense model configs — mirrors turbo benchmark/ops/config.py
DenseModelConfigs = {
    "Llama-2-7B": {
        "seqlen": 4096,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
    },
    "Llama-2-70B": {
        "seqlen": 4096,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "Llama-3.1-8B": {
        "seqlen": 8192,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "Llama-3.1-405B": {
        "seqlen": 8192,
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "Qwen2.5-7B": {
        "seqlen": 8192,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
    },
    "Qwen2.5-72B": {
        "seqlen": 8192,
        "hidden_size": 8192,
        "intermediate_size": 29568,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "Mistral-7B": {
        "seqlen": 4096,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
}

BATCH_SIZE_LIST = [1, 2, 4]


def gen_gemm_test_cases(model_config):
    """Generate GEMM (M, N, K) shapes from a dense model config.

    Returns list of dicts with keys: layer, M_base, N, K
      - M_base is the sequence-length dimension; actual M = M_base * batch_size
    """
    seq = model_config["seqlen"]
    hidden = model_config["hidden_size"]
    inter = model_config["intermediate_size"]
    n_heads = model_config["num_attention_heads"]
    n_kv = model_config["num_key_value_heads"]
    head_dim = model_config["head_dim"]

    return [
        {"layer": "attn_qkv",
         "M_base": seq,
         "N": (n_heads + 2 * n_kv) * head_dim,
         "K": hidden},
        {"layer": "attn_out",
         "M_base": seq,
         "N": hidden,
         "K": hidden},
        {"layer": "mlp_gate_up",
         "M_base": seq,
         "N": 2 * inter,
         "K": hidden},
        {"layer": "mlp_down",
         "M_base": seq,
         "N": hidden,
         "K": inter},
    ]


def gen_all_shapes(model_filter=None):
    """Generate all (case, M, N, K) tuples across models and batch sizes."""
    shapes = []
    for name, cfg in DenseModelConfigs.items():
        if model_filter and model_filter.lower() not in name.lower():
            continue
        cases = gen_gemm_test_cases(cfg)
        for mbs in BATCH_SIZE_LIST:
            for c in cases:
                M = c["M_base"] * mbs
                shapes.append({
                    "model": name,
                    "layer": c["layer"],
                    "mbs": mbs,
                    "M": M,
                    "N": c["N"],
                    "K": c["K"],
                })
    return shapes
