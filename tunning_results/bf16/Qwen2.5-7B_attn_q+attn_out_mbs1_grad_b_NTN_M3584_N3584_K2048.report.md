# Qwen2.5-7B — attn_q+attn_out [grad_b]

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-7B |
| Layer | attn_q+attn_out |
| Phase | grad_b |
| MBS | 1 |
| Trans (ABC) | NTN |
| M | 3584 |
| N | 3584 |
| K | 2048 |
| FLOPs | 5.261e+10 |

## Results

| Method | TFLOPS | Time (us) | vs Bench | vs API |
|--------|-------:|----------:|---------:|-------:|
| **Tensile tuned** | 1034.41 | 50.86 | 169.7% | 145.5% |
| **API bench (installed)** | 710.82 | 74.02 | 116.6% | — |
| **hipblaslt-bench (stock)** | 609.62 | 86.31 | — | — |

> **Tuned/API ratio (gate metric):** 145.52%

## Kernels

- **Tensile winner**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SU16_SUM0_SUS128_SPO0_SRVW0_SSO0_SVW8_SK3_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP1_USFGROn1_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM8_WGMXCC8_WGMXCCGn1`
- **API bench (installed)**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x32_MI32x32x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_4_MO40_NTn1_NTA0_NTB0_NTC4_NTD4_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO4_SVW4_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WG64_4_1`
- **hipblaslt-bench**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x32_MI32x32x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_4_MO40_NTn1_NTA0_NTB0_NTC4_NTD4_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU32_SUM0_SUS256_SPO1_SRVW0_SSO4_SVW4_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WG64_4_1_WGM8_WGMXCC1_WGMXCCGn1`

## hipblaslt-bench

```bash
/opt/rocm/bin/hipblaslt-bench -m 3584 -n 3584 -k 2048 --precision bf16_r --compute_type f32_r --transA N --transB T -i 50 -j 30 --rotating 640 --use_gpu_timer --flush --initialization trig_float --print_kernel_info
```

<details><summary>Raw output</summary>

```
hipBLASLt version: 100200
hipBLASLt git version: de5c1aebb6-dirty
Query device success: there are 1 devices. (Target device ID is 0)
Device ID 0 : AMD Instinct MI350X gfx950:sramecc+:xnack-
with 270.6 GB memory, max. SCLK 2200 MHz, max. MCLK 1900 MHz, compute capability 9.5
maxGridDimX 2147483647, sharedMemPerBlock 163.8 KB, maxThreadsPerBlock 1024, warpSize 64

Rotating buffer 640 MiB. Needed Size: 52 MiB. Needed block count: 13 (Capped to max iters: 50)
Is supported 1 / Total solutions: 1
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,swizzle_a,swizzle_b,activation_type,bias_vector,bias_type,aux_type,rotating_buffer,flush,use_gpu_timer,hipblaslt-Gflops,hipblaslt-GB/s,us
    N,T,0,1,3584,3584,2048,1,3584,7340032,0,3584,7340032,3584,12845056,3584,12845056,bf16_r,bf16_r,bf16_r,bf16_r,f32_r,0,0,0,0,0,0,0,none,0,bf16_r,bf16_r,640,1,1,609615,594.045,86.3058
    --Solution index: 299969
    --Solution name:  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x32_MI32x32x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_4_MO40_NTn1_NTA0_NTB0_NTC4_NTD4_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU32_SUM0_SUS256_SPO1_SRVW0_SSO4_SVW4_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WG64_4_1_WGM8_WGMXCC1_WGMXCCGn1
    --kernel name:    Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x32_MI32x32x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_4_MO40_NTn1_NTA0_NTB0_NTC4_NTD4_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO4_SVW4_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WG64_4_1
```

</details>

## Parameter Differences (Tensile vs hipblaslt-bench)

| Parameter | Tensile | hipblaslt-bench |
|-----------|---------|-----------------|
| MT | 256x256x64 | 256x256x32 |
| MI | 16x16x1 | 32x32x1 |
| LDSB | 0 | 1 |
| AFC | 1 | 0 |
| DTLA | 1 | 0 |
| DTLB | 1 | 0 |
| MIWT | 8_8 | 4_4 |
| NTC | 0 | 4 |
| NEPBS | 0 | 16 |
| SU | 16 | 32 |
| SUS | 128 | 256 |
| SPO | 0 | 1 |
| SSO | 0 | 4 |
| SVW | 8 | 4 |
| SKXCCM | 0 | 8 |
| VWA | 8 | 4 |
| VWB | 8 | 4 |
| WG | 32_8_1 | 64_4_1 |
| WGMXCC | 8 | 1 |

**Tensile-only params:** `CMS`, `AG=0`, `DTVSM=0`, `DPLB=0`, `ELFLR=0`, `EMLL=n1`, `LDSTI=0`, `MGRIPM=1`, `SGROB=0`, `SKFTR=0`, `SGRO=0`, `TIN=0`, `TLDSM=n1`, `UPLRP=1`

