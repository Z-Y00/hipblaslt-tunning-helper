# Qwen2.5-7B — attn_k [grad_b]

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-7B |
| Layer | attn_k |
| Phase | grad_b |
| MBS | 1 |
| Trans (ABC) | NTN |
| M | 512 |
| N | 3584 |
| K | 2048 |
| FLOPs | 7.516e+09 |

## Results

| Method | TFLOPS | Time (us) | vs Bench | vs API |
|--------|-------:|----------:|---------:|-------:|
| **Tensile tuned** | 293.92 | 25.57 | 90.9% | 81.0% |
| **API bench (installed)** | 362.87 | 20.71 | 112.2% | — |
| **hipblaslt-bench (stock)** | 323.28 | 23.25 | — | — |

> **Tuned/API ratio (gate metric):** 81.00%

## Kernels

- **Tensile winner**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x64x64_MI32x32x1_SN_LDSB0_AFC1_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_MGRIPM1_NTn1_NTA0_NTB1_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SU16_SUM0_SUS256_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA2_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1_WGM8_WGMXCC8_WGMXCCGn1`
- **API bench (installed)**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x64x128_MI32x32x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_2_MO40_NTn1_NTA0_NTB0_NTC3_NTD3_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG128_2_1`
- **hipblaslt-bench**: `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x64x128_MI32x32x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_2_MO40_NTn1_NTA0_NTB0_NTC3_NTD3_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU0_SUM0_SUS256_SPO0_SRVW0_SSO0_SVW1_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG128_2_1_WGM6_WGMXCC1_WGMXCCGn1`

## hipblaslt-bench

```bash
/opt/rocm/bin/hipblaslt-bench -m 3584 -n 512 -k 2048 --precision bf16_r --compute_type f32_r --transA N --transB T -i 50 -j 30 --rotating 640 --use_gpu_timer --flush --initialization trig_float --print_kernel_info
```

<details><summary>Raw output</summary>

```
hipBLASLt version: 100200
hipBLASLt git version: de5c1aebb6-dirty
Query device success: there are 1 devices. (Target device ID is 0)
Device ID 0 : AMD Instinct MI350X gfx950:sramecc+:xnack-
with 270.6 GB memory, max. SCLK 2200 MHz, max. MCLK 1900 MHz, compute capability 9.5
maxGridDimX 2147483647, sharedMemPerBlock 163.8 KB, maxThreadsPerBlock 1024, warpSize 64

Rotating buffer 640 MiB. Needed Size: 19 MiB. Needed block count: 33 (Capped to max iters: 50)
Is supported 1 / Total solutions: 1
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,swizzle_a,swizzle_b,activation_type,bias_vector,bias_type,aux_type,rotating_buffer,flush,use_gpu_timer,hipblaslt-Gflops,hipblaslt-GB/s,us
    N,T,0,1,3584,512,2048,1,3584,7340032,0,512,1048576,3584,1835008,3584,1835008,bf16_r,bf16_r,bf16_r,bf16_r,f32_r,0,0,0,0,0,0,0,none,0,bf16_r,bf16_r,640,1,1,323283,819.066,23.2496
    --Solution index: 300119
    --Solution name:  Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x64x128_MI32x32x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_2_MO40_NTn1_NTA0_NTB0_NTC3_NTD3_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU0_SUM0_SUS256_SPO0_SRVW0_SSO0_SVW1_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG128_2_1_WGM6_WGMXCC1_WGMXCCGn1
    --kernel name:    Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x64x128_MI32x32x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_2_MO40_NTn1_NTA0_NTB0_NTC3_NTD3_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKXCCM8_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG128_2_1
```

</details>

## Parameter Differences (Tensile vs hipblaslt-bench)

| Parameter | Tensile | hipblaslt-bench |
|-----------|---------|-----------------|
| MT | 128x64x64 | 128x64x128 |
| AFC | 1 | 0 |
| CLR | 1 | 0 |
| DTLA | 1 | 0 |
| DTLB | 1 | 0 |
| MIAV | 0 | 1 |
| MIWT | 2_1 | 1_2 |
| NTB | 1 | 0 |
| NTC | 0 | 3 |
| NTD | 4 | 3 |
| NEPBS | 0 | 16 |
| SU | 16 | 0 |
| SVW | 2 | 1 |
| SKXCCM | 0 | 8 |
| VWA | 2 | 1 |
| WG | 64_4_1 | 128_2_1 |
| WGM | 8 | 6 |
| WGMXCC | 8 | 1 |

**Tensile-only params:** `AG=0`, `DTVSM=0`, `DPLB=0`, `ELFLR=0`, `EMLL=n1`, `LDSTI=0`, `MGRIPM=1`, `SGROB=0`, `SKFTR=0`, `SGRO=0`, `TIN=0`, `TLDSM=n1`, `UPLRP=0`

