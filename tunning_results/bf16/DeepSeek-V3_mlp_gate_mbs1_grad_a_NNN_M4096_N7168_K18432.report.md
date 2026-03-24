# DeepSeek-V3 — mlp_gate [grad_a]

| Parameter | Value |
|-----------|-------|
| Model | DeepSeek-V3 |
| Layer | mlp_gate |
| Phase | grad_a |
| MBS | 1 |
| Trans (ABC) | NNN |
| M | 4096 |
| N | 7168 |
| K | 18432 |
| FLOPs | 1.082e+12 |

## Results

| Method | TFLOPS | Time (us) | vs Bench | vs API |
|--------|-------:|----------:|---------:|-------:|
| **Tensile tuned** | N/A | N/A | N/A | N/A |
| **API bench (installed)** | 1354.52 | 799.05 | 113.5% | — |
| **hipblaslt-bench (stock)** | 1193.01 | 907.23 | — | — |

## Kernels

- **Tensile winner**: `N/A`
- **API bench (installed)**: `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB1024_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1`
- **hipblaslt-bench**: `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB1024_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU0_SUM0_SUS128_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM16_WGMXCC2_WGMXCCGn1`

## hipblaslt-bench

```bash
/opt/rocm/bin/hipblaslt-bench -m 7168 -n 4096 -k 18432 --precision bf16_r --compute_type f32_r --transA N --transB N -i 50 -j 30 --rotating 3175 --use_gpu_timer --flush --initialization trig_float --print_kernel_info
```

<details><summary>Raw output</summary>

```
hipBLASLt version: 100200
hipBLASLt git version: de5c1aebb6-dirty
Query device success: there are 1 devices. (Target device ID is 0)
Device ID 0 : AMD Instinct MI350X gfx950:sramecc+:xnack-
with 270.6 GB memory, max. SCLK 2200 MHz, max. MCLK 1900 MHz, compute capability 9.5
maxGridDimX 2147483647, sharedMemPerBlock 163.8 KB, maxThreadsPerBlock 1024, warpSize 64

Rotating buffer 3175 MiB. Needed Size: 452 MiB. Needed block count: 1 (Capped to max iters: 50)
Is supported 1 / Total solutions: 1
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,swizzle_a,swizzle_b,activation_type,bias_vector,bias_type,aux_type,rotating_buffer,flush,use_gpu_timer,hipblaslt-Gflops,hipblaslt-GB/s,us
    N,N,0,1,7168,4096,18432,1,7168,132120576,0,18432,75497472,7168,29360128,7168,29360128,bf16_r,bf16_r,bf16_r,bf16_r,f32_r,0,0,0,0,0,0,0,none,0,bf16_r,bf16_r,3175,1,1,1.19301e+06,486.546,907.225
    --Solution index: 300658
    --Solution name:  Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB1024_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SU0_SUM0_SUS128_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM16_WGMXCC2_WGMXCCGn1
    --kernel name:    Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA4096_LBSPPB1024_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK3_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1
```

</details>

## Parameter Differences (Tensile vs hipblaslt-bench)

**hipblaslt-bench-only params:** `MT=256x256x64`, `MI=16x16x1`, `CMS`, `SN`, `LDSB=0`, `AFC=1`, `AFEM=1`, `ASEM=1`, `CLR=1`, `CADS=0`, `DTLA=1`, `DTLB=1`, `DTVA=0`, `DTVB=0`, `EPS=0`, `FDSI=0`, `GRPM=1`, `GRVWA=8`, `GRVWB=8`, `GSU=0`, `GSUAMB`, `GSUC=0`, `GSUWGMRR=0`, `GLS=0`, `ISA=950`, `IU=1`, `K=1`, `LBSPPA=4096`, `LBSPPB=1024`, `LBSPPM=0`, `LPA=0`, `LPB=16`, `LPM=0`, `LRVW=8`, `LWPM=n1`, `MIAV=0`, `MIWT=8_8`, `MO=40`, `NT=n1`, `NTA=0`, `NTB=0`, `NTC=0`, `NTD=4`, `NTM=0`, `NEPBS=0`, `NLCA=1`, `NLCB=1`, `ONLL=1`, `PGR=2`, `PLR=1`, `PKA=1`, `SIA=3`, `SS=1`, `SU=0`, `SUM=0`, `SUS=128`, `SPO=0`, `SRVW=0`, `SSO=0`, `SVW=8`, `SK=3`, `SKXCCM=0`, `TLDS=1`, `ULSGRO=0`, `USL=1`, `UIOFGRO=0`, `USFGRO=0`, `VS=n1`, `VWA=8`, `VWB=8`, `WSGRA=0`, `WSGRB=0`, `WS=64`, `WG=32_8_1`, `WGM=16`, `WGMXCC=2`, `WGMXCCG=n1`

