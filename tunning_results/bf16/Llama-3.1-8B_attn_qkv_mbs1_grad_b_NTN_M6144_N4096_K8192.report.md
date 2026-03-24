# Llama-3.1-8B — attn_qkv [grad_b]

| Parameter | Value |
|-----------|-------|
| Model | Llama-3.1-8B |
| Layer | attn_qkv |
| Phase | grad_b |
| MBS | 1 |
| Trans (ABC) | NTN |
| M | 6144 |
| N | 4096 |
| K | 8192 |
| FLOPs | 4.123e+11 |

## Results

| Method | TFLOPS | Time (us) | vs Bench | vs API |
|--------|-------:|----------:|---------:|-------:|
| **Tensile tuned** | 1092.66 | 377.35 | 119.0% | 108.9% |
| **API bench (installed)** | 1003.38 | 410.93 | 109.3% | — |
| **hipblaslt-bench (stock)** | 918.37 | 448.96 | — | — |

> **Tuned/API ratio (gate metric):** 108.90%

## Kernels

- **Tensile winner**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SU0_SUM0_SUS128_SPO0_SRVW0_SSO0_SVW8_SK3_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP1_USFGROn1_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM8_WGMXCC1_WGMXCCGn1`
- **API bench (installed)**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x192x64_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB3072_LBSPPM0_LPA0_LPB32_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB3_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1`
- **hipblaslt-bench**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x192x64_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB3072_LBSPPM0_LPA0_LPB32_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB3_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU0_SUM0_SUS128_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM6_WGMXCC1_WGMXCCGn1`

## hipblaslt-bench

```bash
/opt/rocm/bin/hipblaslt-bench -m 4096 -n 6144 -k 8192 --precision bf16_r --compute_type f32_r --transA N --transB T -i 50 -j 30 --rotating 1600 --use_gpu_timer --flush --initialization trig_float --print_kernel_info
```

<details><summary>Raw output</summary>

```
hipBLASLt version: 100200
hipBLASLt git version: de5c1aebb6-dirty
Query device success: there are 1 devices. (Target device ID is 0)
Device ID 0 : AMD Instinct MI350X gfx950:sramecc+:xnack-
with 270.6 GB memory, max. SCLK 2200 MHz, max. MCLK 1900 MHz, compute capability 9.5
maxGridDimX 2147483647, sharedMemPerBlock 163.8 KB, maxThreadsPerBlock 1024, warpSize 64

Rotating buffer 1600 MiB. Needed Size: 208 MiB. Needed block count: 8 (Capped to max iters: 50)
Is supported 1 / Total solutions: 1
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,swizzle_a,swizzle_b,activation_type,bias_vector,bias_type,aux_type,rotating_buffer,flush,use_gpu_timer,hipblaslt-Gflops,hipblaslt-GB/s,us
    N,T,0,1,4096,6144,8192,1,4096,33554432,0,6144,50331648,4096,25165824,4096,25165824,bf16_r,bf16_r,bf16_r,bf16_r,f32_r,0,0,0,0,0,0,0,none,0,bf16_r,bf16_r,1600,1,1,918371,452.429,448.965
    --Solution index: 300049
    --Solution name:  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x192x64_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB3072_LBSPPM0_LPA0_LPB32_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB3_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU0_SUM0_SUS128_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM6_WGMXCC1_WGMXCCGn1
    --kernel name:    Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x192x64_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB3072_LBSPPM0_LPA0_LPB32_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB3_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1
```

</details>

## Parameter Differences (Tensile vs hipblaslt-bench)

| Parameter | Tensile | hipblaslt-bench |
|-----------|---------|-----------------|
| MT | 256x256x64 | 256x192x64 |
| CLR | 0 | 1 |
| DTLA | 1 | 0 |
| DTLB | 1 | 0 |
| LBSPPA | 0 | 4096 |
| LBSPPB | 0 | 3072 |
| LPB | 0 | 32 |
| MIWT | 8_8 | 8_6 |
| NLCB | 1 | 3 |
| SKXCCM | 0 | 8 |
| USFGRO | n1 | 0 |
| VWB | 8 | 2 |
| WGM | 8 | 6 |

**Tensile-only params:** `CMS`, `AG=0`, `DTVSM=0`, `DPLB=0`, `ELFLR=0`, `EMLL=n1`, `LDSTI=0`, `MGRIPM=1`, `SGROB=0`, `SKFTR=0`, `SGRO=0`, `TIN=0`, `TLDSM=n1`, `UPLRP=1`

