# Llama-3.1-8B — attn_k [grad_b]

| Parameter | Value |
|-----------|-------|
| Model | Llama-3.1-8B |
| Layer | attn_k |
| Phase | grad_b |
| MBS | 1 |
| Trans (ABC) | NTN |
| M | 1024 |
| N | 4096 |
| K | 8192 |
| FLOPs | 6.872e+10 |

## Results

| Method | TFLOPS | Time (us) | vs Bench | vs API |
|--------|-------:|----------:|---------:|-------:|
| **Tensile tuned** | N/A | N/A | N/A | N/A |
| **API bench (installed)** | 627.35 | 109.54 | 106.9% | — |
| **hipblaslt-bench (stock)** | 587.08 | 117.05 | — | — |

## Kernels

- **Tensile winner**: `N/A`
- **API bench (installed)**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x256x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA2048_LBSPPB4096_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1`
- **hipblaslt-bench**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x256x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA2048_LBSPPB4096_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU0_SUM0_SUS128_SPO0_SRVW0_SSO0_SVW4_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM6_WGMXCC1_WGMXCCGn1`

## hipblaslt-bench

```bash
/opt/rocm/bin/hipblaslt-bench -m 4096 -n 1024 -k 8192 --precision bf16_r --compute_type f32_r --transA N --transB T -i 50 -j 30 --rotating 640 --use_gpu_timer --flush --initialization trig_float --print_kernel_info
```

<details><summary>Raw output</summary>

```
hipBLASLt version: 100200
hipBLASLt git version: de5c1aebb6-dirty
Query device success: there are 1 devices. (Target device ID is 0)
Device ID 0 : AMD Instinct MI350X gfx950:sramecc+:xnack-
with 270.6 GB memory, max. SCLK 2200 MHz, max. MCLK 1900 MHz, compute capability 9.5
maxGridDimX 2147483647, sharedMemPerBlock 163.8 KB, maxThreadsPerBlock 1024, warpSize 64

Rotating buffer 640 MiB. Needed Size: 88 MiB. Needed block count: 8 (Capped to max iters: 50)
Is supported 1 / Total solutions: 1
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,swizzle_a,swizzle_b,activation_type,bias_vector,bias_type,aux_type,rotating_buffer,flush,use_gpu_timer,hipblaslt-Gflops,hipblaslt-GB/s,us
    N,T,0,1,4096,1024,8192,1,4096,33554432,0,1024,8388608,4096,4194304,4096,4194304,bf16_r,bf16_r,bf16_r,bf16_r,f32_r,0,0,0,0,0,0,0,none,0,bf16_r,bf16_r,640,1,1,587079,734.175,117.053
    --Solution index: 300075
    --Solution name:  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x256x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA2048_LBSPPB4096_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU0_SUM0_SUS128_SPO0_SRVW0_SSO0_SVW4_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM6_WGMXCC1_WGMXCCGn1
    --kernel name:    Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x256x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA2048_LBSPPB4096_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1
```

</details>

## Parameter Differences (Tensile vs hipblaslt-bench)

**hipblaslt-bench-only params:** `MT=128x256x64`, `MI=16x16x1`, `SN`, `LDSB=1`, `AFC=1`, `AFEM=1`, `ASEM=1`, `CLR=1`, `CADS=0`, `DTLA=0`, `DTLB=0`, `DTVA=0`, `DTVB=0`, `EPS=0`, `FDSI=0`, `GRPM=1`, `GRVWA=8`, `GRVWB=8`, `GSU=0`, `GSUAMB`, `GSUC=0`, `GSUWGMRR=0`, `GLS=0`, `ISA=950`, `IU=1`, `K=1`, `LBSPPA=2048`, `LBSPPB=4096`, `LBSPPM=0`, `LPA=0`, `LPB=0`, `LPM=0`, `LRVW=8`, `LWPM=n1`, `MIAV=0`, `MIWT=4_8`, `MO=40`, `NT=n1`, `NTA=0`, `NTB=0`, `NTC=0`, `NTD=0`, `NTM=0`, `NEPBS=0`, `NLCA=1`, `NLCB=1`, `ONLL=1`, `PGR=2`, `PLR=1`, `PKA=1`, `SIA=3`, `SS=1`, `SU=0`, `SUM=0`, `SUS=128`, `SPO=0`, `SRVW=0`, `SSO=0`, `SVW=4`, `SK=3`, `SKXCCM=8`, `TLDS=0`, `ULSGRO=0`, `USL=1`, `UIOFGRO=0`, `USFGRO=0`, `VS=n1`, `VWA=4`, `VWB=8`, `WSGRA=0`, `WSGRB=0`, `WS=64`, `WG=32_8_1`, `WGM=6`, `WGMXCC=1`, `WGMXCCG=n1`

