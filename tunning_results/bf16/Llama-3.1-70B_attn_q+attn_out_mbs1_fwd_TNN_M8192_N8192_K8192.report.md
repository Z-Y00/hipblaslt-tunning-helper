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
| **Tensile tuned** | 1298.73 | 846.60 | 100.5% | 87.8% |
| **API bench (installed)** | 1478.52 | 743.65 | 114.4% | — |
| **hipblaslt-bench (stock)** | 1292.11 | 850.94 | — | — |

> **Tuned/API ratio (gate metric):** 87.84%

## Kernels

- **Tensile winner**: `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_AFC1_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GSUC0_GSUWGMRR0_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SU16_SUM0_SUS256_SPO0_SRVW0_SSO0_SVW8_SK3_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1_WGM8_WGMXCC8_WGMXCCGn1`
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
    T,N,0,1,8192,8192,8192,1,8192,67108864,0,8192,67108864,8192,67108864,8192,67108864,bf16_r,bf16_r,bf16_r,bf16_r,f32_r,0,0,0,0,0,0,0,none,0,bf16_r,bf16_r,0,1,1,1.29211e+06,440.689,850.941
    --Solution index: 302286
    --Solution name:  Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname0_gfx950
    --kernel name:    Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname0_gfx950
```

</details>

## Parameter Differences (Tensile vs hipblaslt-bench)

**Tensile-only params:** `CMS`, `SN`, `LDSB=0`, `AFC=1`, `AG=0`, `AFEM=1`, `ASEM=1`, `CLR=0`, `CADS=0`, `DTLA=1`, `DTLB=1`, `DTVA=0`, `DTVB=0`, `DTVSM=0`, `DPLB=0`, `EPS=0`, `ELFLR=0`, `EMLL=n1`, `FDSI=0`, `GRPM=1`, `GRVWA=8`, `GRVWB=8`, `GSU=0`, `GSUAMB`, `GSUC=0`, `GSUWGMRR=0`, `GLS=0`, `ISA=950`, `IU=1`, `K=1`, `LDSTI=0`, `LBSPPA=1024`, `LBSPPB=1024`, `LBSPPM=0`, `LPA=16`, `LPB=16`, `LPM=0`, `LRVW=8`, `LWPM=n1`, `MIAV=0`, `MIWT=8_8`, `MO=40`, `MGRIPM=1`, `NT=n1`, `NTA=0`, `NTB=0`, `NTC=0`, `NTD=4`, `NTM=0`, `NEPBS=0`, `NLCA=1`, `NLCB=1`, `ONLL=1`, `PGR=2`, `PLR=1`, `PKA=1`, `SGROB=0`, `SIA=3`, `SS=1`, `SU=16`, `SUM=0`, `SUS=256`, `SPO=0`, `SRVW=0`, `SSO=0`, `SVW=8`, `SK=3`, `SKFTR=0`, `SKXCCM=0`, `SGRO=0`, `TIN=0`, `TLDS=1`, `TLDSM=n1`, `ULSGRO=0`, `USL=1`, `UIOFGRO=0`, `UPLRP=0`, `USFGRO=n1`, `VS=n1`, `VWA=8`, `VWB=8`, `WSGRA=0`, `WSGRB=0`, `WS=64`, `WG=32_8_1`, `WGM=8`, `WGMXCC=8`, `WGMXCCG=n1`

**hipblaslt-bench-only params:** `UserArgs`, `shortname=0`, `gfx=950`

