# Llama-2-70B — attn_q+attn_out [grad_b]

| Parameter | Value |
|-----------|-------|
| Model | Llama-2-70B |
| Layer | attn_q+attn_out |
| Phase | grad_b |
| MBS | 1 |
| Trans (ABC) | NTN |
| M | 8192 |
| N | 8192 |
| K | 4096 |
| FLOPs | 5.498e+11 |

## Results

| Method | TFLOPS | Time (us) | vs Bench | vs API |
|--------|-------:|----------:|---------:|-------:|
| **Tensile tuned** | 1162.04 | 473.09 | 94.2% | 87.2% |
| **API bench (installed)** | 1333.10 | 412.39 | 108.1% | — |
| **hipblaslt-bench (stock)** | 1233.55 | 445.67 | — | — |

> **Tuned/API ratio (gate metric):** 87.17%

## Kernels

- **Tensile winner**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SU16_SUM0_SUS256_SPO0_SRVW0_SSO0_SVW8_SK3_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP1_USFGROn1_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM8_WGMXCC8_WGMXCCGn1`
- **API bench (installed)**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB4096_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1`
- **hipblaslt-bench**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB4096_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU0_SUM0_SUS128_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM16_WGMXCC2_WGMXCCGn1`

## hipblaslt-bench

```bash
/opt/rocm/bin/hipblaslt-bench -m 8192 -n 8192 -k 4096 --precision bf16_r --compute_type f32_r --transA N --transB T -i 50 -j 30 --rotating 2400 --use_gpu_timer --flush --initialization trig_float --print_kernel_info
```

<details><summary>Raw output</summary>

```
hipBLASLt version: 100200
hipBLASLt git version: de5c1aebb6-dirty
Query device success: there are 1 devices. (Target device ID is 0)
Device ID 0 : AMD Instinct MI350X gfx950:sramecc+:xnack-
with 270.6 GB memory, max. SCLK 2200 MHz, max. MCLK 1900 MHz, compute capability 9.5
maxGridDimX 2147483647, sharedMemPerBlock 163.8 KB, maxThreadsPerBlock 1024, warpSize 64

Rotating buffer 2400 MiB. Needed Size: 256 MiB. Needed block count: 1 (Capped to max iters: 50)
Is supported 1 / Total solutions: 1
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,swizzle_a,swizzle_b,activation_type,bias_vector,bias_type,aux_type,rotating_buffer,flush,use_gpu_timer,hipblaslt-Gflops,hipblaslt-GB/s,us
    N,T,0,1,8192,8192,4096,1,8192,33554432,0,8192,33554432,8192,67108864,8192,67108864,bf16_r,bf16_r,bf16_r,bf16_r,f32_r,0,0,0,0,0,0,0,none,0,bf16_r,bf16_r,2400,1,1,1.23355e+06,560.955,445.668
    --Solution index: 300155
    --Solution name:  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB4096_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU0_SUM0_SUS128_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM16_WGMXCC2_WGMXCCGn1
    --kernel name:    Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB4096_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1
```

</details>

## Parameter Differences (Tensile vs hipblaslt-bench)

| Parameter | Tensile | hipblaslt-bench |
|-----------|---------|-----------------|
| LBSPPA | 0 | 4096 |
| LBSPPB | 0 | 4096 |
| NTD | 0 | 4 |
| SU | 16 | 0 |
| SUS | 256 | 128 |
| USFGRO | n1 | 0 |
| WGM | 8 | 16 |
| WGMXCC | 8 | 2 |

**Tensile-only params:** `AG=0`, `DTVSM=0`, `DPLB=0`, `ELFLR=0`, `EMLL=n1`, `LDSTI=0`, `MGRIPM=1`, `SGROB=0`, `SKFTR=0`, `SGRO=0`, `TIN=0`, `TLDSM=n1`, `UPLRP=1`

