#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#define HIP_CHECK(expr)                                                      \
    do {                                                                     \
        hipError_t _e = (expr);                                              \
        if (_e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n", _e,             \
                    hipGetErrorString(_e), __FILE__, __LINE__);              \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define HIPBLASLT_CHECK(expr)                                                \
    do {                                                                     \
        hipblasStatus_t _s = (expr);                                         \
        if (_s != HIPBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "hipBLASLt error %d at %s:%d\n", (int)_s,       \
                    __FILE__, __LINE__);                                     \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

struct GemmShape {
    const char* name;
    int64_t m, n, k;       // col-major hipblaslt dims
    hipblasOperation_t opA, opB;
    double tensile_tflops;  // reference from tuning
    double bench_tflops;    // reference from hipblaslt-bench
};

static double run_gemm(hipblasLtHandle_t handle, const GemmShape& s,
                       int warmup, int iters, bool verbose) {
    const int64_t m = s.m, n = s.n, k = s.k;
    const int64_t lda = (s.opA == HIPBLAS_OP_T) ? k : m;
    const int64_t ldb = (s.opB == HIPBLAS_OP_T) ? n : k;
    const int64_t ldc = m, ldd = m;

    size_t sizeA = lda * ((s.opA == HIPBLAS_OP_T) ? m : k);
    size_t sizeB = ldb * ((s.opB == HIPBLAS_OP_T) ? k : n);
    size_t sizeC = ldc * n;
    size_t sizeD = ldd * n;
    size_t bytesA = sizeA * sizeof(hip_bfloat16);
    size_t bytesB = sizeB * sizeof(hip_bfloat16);
    size_t bytesC = sizeC * sizeof(hip_bfloat16);
    size_t bytesD = sizeD * sizeof(hip_bfloat16);
    // Tensile init: A,B,C = random bf16 in [-3,3], D = zero, alpha=1, beta=0
    auto fill_random_bf16 = [](hip_bfloat16* host, size_t count, unsigned seed) {
        std::mt19937 rng(seed);
        for (size_t i = 0; i < count; i++)
            host[i] = static_cast<hip_bfloat16>(static_cast<float>((int)(rng() % 7) - 3));
    };

    std::vector<hip_bfloat16> hA(sizeA), hB(sizeB), hC(sizeC);
    fill_random_bf16(hA.data(), sizeA, 0);
    fill_random_bf16(hB.data(), sizeB, 1);
    fill_random_bf16(hC.data(), sizeC, 2);

    hip_bfloat16 *dA, *dB, *dC, *dD;
    HIP_CHECK(hipMalloc(&dA, bytesA));
    HIP_CHECK(hipMalloc(&dB, bytesB));
    HIP_CHECK(hipMalloc(&dC, bytesC));
    HIP_CHECK(hipMalloc(&dD, bytesD));
    HIP_CHECK(hipMemcpy(dA, hA.data(), bytesA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), bytesB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC.data(), bytesC, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dD, 0, bytesD));

    hipblasLtMatmulDesc_t matmulDesc;
    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmulDesc,
                    HIPBLAS_COMPUTE_32F, HIP_R_32F));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmulDesc,
                    HIPBLASLT_MATMUL_DESC_TRANSA, &s.opA, sizeof(s.opA)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmulDesc,
                    HIPBLASLT_MATMUL_DESC_TRANSB, &s.opB, sizeof(s.opB)));

    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
    int64_t rowsA = (s.opA == HIPBLAS_OP_T) ? k : m;
    int64_t colsA = (s.opA == HIPBLAS_OP_T) ? m : k;
    int64_t rowsB = (s.opB == HIPBLAS_OP_T) ? n : k;
    int64_t colsB = (s.opB == HIPBLAS_OP_T) ? k : n;

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16BF,
                    rowsA, colsA, lda));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16BF,
                    rowsB, colsB, ldb));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16BF,
                    m, n, ldc));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_16BF,
                    m, n, ldd));

    hipblasLtMatmulPreference_t pref;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    size_t maxWorkspace = 256 * 1024 * 1024;  // 256 MB
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,
                    HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &maxWorkspace, sizeof(maxWorkspace)));

    const int requestCount = 10;
    std::vector<hipblasLtMatmulHeuristicResult_t> results(requestCount);
    int returnCount = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc,
                    layoutA, layoutB, layoutC, layoutD, pref,
                    requestCount, results.data(), &returnCount));

    if (returnCount == 0) {
        fprintf(stderr, "  No algorithms found for %s!\n", s.name);
        hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dD);
        hipblasLtMatmulDescDestroy(matmulDesc);
        hipblasLtMatrixLayoutDestroy(layoutA);
        hipblasLtMatrixLayoutDestroy(layoutB);
        hipblasLtMatrixLayoutDestroy(layoutC);
        hipblasLtMatrixLayoutDestroy(layoutD);
        hipblasLtMatmulPreferenceDestroy(pref);
        return 0;
    }

    if (verbose)
        printf("  Algorithms returned: %d\n", returnCount);

    std::string solName = hipblaslt_ext::getSolutionNameFromAlgo(handle, results[0].algo);
    std::string kernName = hipblaslt_ext::getKernelNameFromAlgo(handle, results[0].algo);

    void* workspace = nullptr;
    size_t wsSize = results[0].workspaceSize;
    if (wsSize > 0)
        HIP_CHECK(hipMalloc(&workspace, wsSize));

    float alpha = 1.0f, beta = 0.0f;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Warmup (Tensile: num-warmups=30)
    for (int i = 0; i < warmup; i++) {
        HIPBLASLT_CHECK(hipblasLtMatmul(handle, matmulDesc,
                        &alpha, dA, layoutA, dB, layoutB,
                        &beta, dC, layoutC, dD, layoutD,
                        &results[0].algo, workspace, wsSize, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // Tensile-exact timing:
    //   ONE event pair around all enqueues, total_time / N = avg per-enqueue
    //   No per-call events, no rotating buffers (rotating-buffer-size=0)
    const int enqueues_per_sync = iters;  // all enqueues in one batch
    double flops = 2.0 * m * n * k;

    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    HIP_CHECK(hipEventRecord(ev_start, stream));
    for (int j = 0; j < enqueues_per_sync; j++) {
        HIPBLASLT_CHECK(hipblasLtMatmul(handle, matmulDesc,
                        &alpha, dA, layoutA, dB, layoutB,
                        &beta, dC, layoutC, dD, layoutD,
                        &results[0].algo, workspace, wsSize, stream));
    }
    HIP_CHECK(hipEventRecord(ev_stop, stream));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float total_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&total_ms, ev_start, ev_stop));
    double total_us = total_ms * 1000.0;
    double avg_us = total_us / enqueues_per_sync;
    double avg_tflops = flops / (avg_us * 1e6);

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    HIP_CHECK(hipStreamDestroy(stream));
    if (workspace) hipFree(workspace);
    hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dD);
    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutC);
    hipblasLtMatrixLayoutDestroy(layoutD);
    hipblasLtMatmulPreferenceDestroy(pref);

    printf("  %-36s m=%5ld n=%5ld k=%5ld | avg %7.1f us  %7.1f TF | Tensile=%7.1f  Bench=%7.1f\n",
           s.name, (long)m, (long)n, (long)k,
           avg_us, avg_tflops,
           s.tensile_tflops, s.bench_tflops);
    printf("    kernel: %s\n", kernName.c_str());

    return avg_tflops;
}

int main(int argc, char** argv) {
    int device = 0;
    int warmup = 50;
    int iters  = 200;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            iters = atoi(argv[++i]);
    }

    HIP_CHECK(hipSetDevice(device));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    printf("Device %d: %s\n", device, props.name);
    printf("Warmup: %d, Iters: %d\n\n", warmup, iters);

    hipblasLtHandle_t handle;
    HIPBLASLT_CHECK(hipblasLtCreate(&handle));

    // All shapes from tuning: hipblaslt col-major (m=N_row, n=M_row, k=K)
    // transA=T, transB=N for all TNN shapes
    GemmShape shapes[] = {
        {"attn_k(M8192_N1024_K4096)",
         1024, 8192, 4096, HIPBLAS_OP_T, HIPBLAS_OP_N, 1223.23, 702.51},
        {"attn_q+out(M8192_N4096_K4096)",
         4096, 8192, 4096, HIPBLAS_OP_T, HIPBLAS_OP_N, 1570.28, 1570.28},
        {"attn_qkv(M8192_N6144_K4096)",
         6144, 8192, 4096, HIPBLAS_OP_T, HIPBLAS_OP_N, 1603.70, 0},
        {"mlp_gate(M8192_N14336_K4096)",
         14336, 8192, 4096, HIPBLAS_OP_T, HIPBLAS_OP_N, 1733.63, 1537.05},
        {"mlp_gate_up(M8192_N28672_K4096)",
         28672, 8192, 4096, HIPBLAS_OP_T, HIPBLAS_OP_N, 1619.47, 1480.34},
        {"mlp_down(M8192_N4096_K14336)",
         4096, 8192, 14336, HIPBLAS_OP_T, HIPBLAS_OP_N, 1797.47, 1629.03},
    };

    printf("  Tensile-exact: %d warmup, %d enqueues (1 event pair), rotating=0, use-gpu-timer=True\n\n",
           warmup, iters);
    printf("  %-36s %5s %5s %5s   %18s   %10s %10s\n",
           "Shape", "m", "n", "k",
           "avg us / TF", "Tensile", "Bench");
    printf("%s\n", std::string(120, '-').c_str());

    for (auto& s : shapes)
        run_gemm(handle, s, warmup, iters, false);

    HIPBLASLT_CHECK(hipblasLtDestroy(handle));
    return 0;
}
