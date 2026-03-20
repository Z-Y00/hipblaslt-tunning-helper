# Tensile Client vs hipblasLtMatmul Dispatch Gap Investigation

## Shape
- **Model**: Llama-3.1-70B
- **Layer**: mlp_gate_up (fwd)
- **MBS**: 2
- **BLAS dims**: M=57344, N=16384, K=8192 (TransA=T, TransB=N)

## Finding

The **exact same kernel** shows a 7% performance gap between the Tensile
benchmarking client and the hipblasLtMatmul API:

| Measurement | Kernel | Latency | TFLOPS |
|-------------|--------|---------|--------|
| Tensile client | API kernel (NTD4+SK3+LBSPPA1024+LPA16+TLDS1+WG32_8_1) | 11,737 µs | 1312 |
| API bench (`hipblasLtMatmul`) | Same kernel | 10,966 µs | 1404 |
| hipblaslt-bench | Same kernel | 12,432 µs | 1238 |

The API kernel was identified by name:
```
Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_SN_LDSB0_
AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_
GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LBSPPA1024_LBSPPB1024_LBSPPM0_
LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT8_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_
NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK3_
SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA8_VWB8_WSGRA0_WSGRB0_
WS64_WG32_8_1
```

This kernel was found in the Tensile search (4 matches in BF16 pass) and scored
1312 TFLOPS — not enough to win over the NTD0 variant (1322 TFLOPS).

## Conditions

- Both use `rotating=0` (warm L2 cache) — tensor set is 4736 MB >> 2×LLC (512 MB)
- Both use single event pair around 50 enqueues, total/N averaging
- Both use identical data init: random integers [-3, 3] cast to BF16
- Same assembler flags (clang -x assembler --target=amdgcn-amd-amdhsa -mcpu=gfx950)
- Same code object version (v4)

## Root Cause

The gap is in the **dispatch path**, not the kernel binary:
- Tensile client: `hipExtModuleLaunchKernel()` — raw kernel dispatch
- API bench: `hipblasLtMatmul()` — hipBLASLt runtime dispatch

The hipBLASLt runtime achieves ~771 µs less total time across 50 iterations
(~15 µs/call), suggesting better kernel argument setup, pipelining, or
reduced inter-kernel gaps.

## Impact on T/A Ratio

This ~7% dispatch gap means T/A ratios around 93-95% do **not** indicate the
Tensile-tuned kernel is slower. The tuned kernel may be equally good or better
than the installed kernel, but the Tensile client's measurement systematically
underreports performance relative to hipblasLtMatmul.

## Files

- `tensile_ntd4.log` — Full Tensile log with NTD=[0,4] in search space (840 solutions)
- `mlp_gate_up_mbs2_ntd4.yaml` — The Tensile YAML config used
- `Cijk_*_CSVWinner.csv/yaml` — Winner results from the Tensile run
- `*.report.md` — Original shape report from the tuning run
