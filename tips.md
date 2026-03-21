# Tuning Tips & Findings

## StaggerU Gives 30% Speedup on Small-Dimension Shapes

**Shapes**: `attn_k` with K=1024 or M=1024 (Llama-2-70B, MBS=1)

| Shape | Tensile (with StaggerU) | hipblaslt-bench (stock) | Improvement |
|-------|------------------------|------------------------|-------------|
| grad_a (M=4096, N=8192, K=1024) | 960 TFLOPS | 745 TFLOPS | **+29%** |
| grad_b (M=1024, N=8192, K=4096) | 766 TFLOPS | 590 TFLOPS | **+30%** |

### What helped

1. **StaggerU=8 with StaggerUStride=256**: The biggest win. For small K or M
   dimensions that are powers of 2, all workgroups load from addresses with
   exact power-of-2 strides, causing DRAM bank and memory channel conflicts.
   StaggerU offsets each workgroup's starting K position by 256 bytes (one
   MI300X memory channel width), spreading traffic across channels.

2. **Smaller tile (grad_b only)**: MT128x256x64 instead of stock's MT256x256x64.
   For M=1024, the smaller M-tile gives better occupancy — more workgroups with
   less wasted compute at boundaries.

3. **Larger MFMA (grad_b only)**: MI32x32x1 instead of MI16x16x1. The larger
   instruction works better for this specific shape/occupancy pattern.

### Why the stock library misses this

The stock hipBLASLt library uses `StaggerU=0` (disabled) and `StaggerUStride=128`
for these shapes. Our template previously also had `StaggerU: [0]` — the
optimization was invisible until we added `StaggerU: [0, 8, 16]` and
`StaggerUStride: [0, 64, 128, 256]` to match the production library's range.

### Takeaway

Always include StaggerU in the search space for shapes with power-of-2
dimensions, especially small ones (K or M ≤ 4096). The cost is 2-3x more
solutions to benchmark, but the payoff can be 30%+ for affected shapes.
