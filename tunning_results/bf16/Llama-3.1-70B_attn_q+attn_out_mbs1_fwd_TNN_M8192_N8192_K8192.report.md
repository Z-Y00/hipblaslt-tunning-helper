# Llama-3.1-70B — attn_q+attn_out [fwd]

| Parameter | Value |
|-----------|-------|
| Model | Llama-3.1-70B |
| Layer | attn_q+attn_out |
| Phase | fwd |
| MBS | 1 |
| Trans (ABC) | TNN |
| M | 8192 |
| N | 8192 |
| K | 8192 |
| FLOPs | 1.100e+12 |

## Results

| Method | TFLOPS | Time (us) | vs Bench | vs API |
|--------|-------:|----------:|---------:|-------:|
| **Tensile tuned** | N/A | N/A | N/A | N/A |
| **API bench (installed)** | 1417.52 | 775.66 | 115.2% | — |
| **hipblaslt-bench (stock)** | 1230.33 | 893.67 | — | — |

## Kernels

- **Tensile winner**: `N/A`
- **API bench (installed)**: `Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname0_gfx950`
- **hipblaslt-bench**: `Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname0_gfx950`

## hipblaslt-bench

```bash
/opt/rocm/bin/hipblaslt-bench -m 8192 -n 8192 -k 8192 --precision bf16_r --compute_type f32_r --transA T --transB N -i 50 -j 30 --rotating 0 --use_gpu_timer --flush --initialization trig_float --print_kernel_info
```

<details><summary>Raw output</summary>

```
hipBLASLt version: 100200
hipBLASLt git version: de5c1aebb6-dirty
Query device success: there are 1 devices. (Target device ID is 0)
Device ID 0 : AMD Instinct MI350X gfx950:sramecc+:xnack-
with 270.6 GB memory, max. SCLK 2200 MHz, max. MCLK 1900 MHz, compute capability 9.5
maxGridDimX 2147483647, sharedMemPerBlock 163.8 KB, maxThreadsPerBlock 1024, warpSize 64

Is supported 1 / Total solutions: 1
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,swizzle_a,swizzle_b,activation_type,bias_vector,bias_type,aux_type,rotating_buffer,flush,use_gpu_timer,hipblaslt-Gflops,hipblaslt-GB/s,us
    T,N,0,1,8192,8192,8192,1,8192,67108864,0,8192,67108864,8192,67108864,8192,67108864,bf16_r,bf16_r,bf16_r,bf16_r,f32_r,0,0,0,0,0,0,0,none,0,bf16_r,bf16_r,0,1,1,1.23033e+06,419.618,893.669
    --Solution index: 302286
    --Solution name:  Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname0_gfx950
    --kernel name:    Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname0_gfx950
```

</details>

## Parameter Differences (Tensile vs hipblaslt-bench)

**hipblaslt-bench-only params:** `MT=256x256x64`, `MI=16x16x1`, `UserArgs`, `shortname=0`, `gfx=950`

