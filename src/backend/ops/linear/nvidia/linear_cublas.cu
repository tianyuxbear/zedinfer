#include "backend/ops/linear/nvidia/linear_cublas.cuh"
#include "utils/check.hpp"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace zedinfer::ops::nvidia::cublas {

// ============================================================================
// Singleton handle + workspace
// ============================================================================

static cublasLtHandle_t &handle() {
    static cublasLtHandle_t h = []() {
        cublasLtHandle_t h;
        if (cublasLtCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLASLt handle");
        }
        return h;
    }();
    return h;
}

static constexpr size_t WORKSPACE_SIZE = 32 * 1024 * 1024;

static void *&workspace() {
    static void *ws = []() {
        void *ptr = nullptr;
        cudaMalloc(&ptr, WORKSPACE_SIZE);
        return ptr;
    }();
    return ws;
}

// ============================================================================
// Dtype mapping
// ============================================================================

static cudaDataType_t to_cuda_dtype(zedinferDataType_t dt) {
    switch (dt) {
    case ZEDINFER_DTYPE_F32:  return CUDA_R_32F;
    case ZEDINFER_DTYPE_F16:  return CUDA_R_16F;
    case ZEDINFER_DTYPE_BF16: return CUDA_R_16BF;
    default: throw std::runtime_error("Unsupported dtype for cuBLASLt linear");
    }
}

// ============================================================================
// Algorithm cache
// ============================================================================

struct MatmulKey {
    size_t M, N, K;
    zedinferDataType_t dtype;
    bool has_bias;

    bool operator==(const MatmulKey &o) const {
        return M == o.M && N == o.N && K == o.K && dtype == o.dtype && has_bias == o.has_bias;
    }
};

struct MatmulKeyHash {
    size_t operator()(const MatmulKey &k) const {
        size_t h = k.M * 2654435761u;
        h ^= k.N * 40503u;
        h ^= k.K * 12289u;
        h ^= static_cast<size_t>(k.dtype) << 3;
        h ^= static_cast<size_t>(k.has_bias) << 7;
        return h;
    }
};

struct CachedAlgo {
    cublasLtMatmulAlgo_t algo;
    bool valid = false;
};

static std::unordered_map<MatmulKey, CachedAlgo, MatmulKeyHash> algo_cache;
static std::mutex algo_mutex;

// ============================================================================
// Core matmul
//
// We want: C[M,N] = A[M,K] * B[N,K]^T + bias[N]
//   A is row-major [M,K], B is row-major [N,K], C is row-major [M,N]
//
// cuBLASLt is column-major. The standard trick for row-major GEMM:
//   Row-major C[M,N] = A[M,K] * B^T[K,N]
//   is equivalent to:
//   Col-major C^T[N,M] = B[N,K] * A^T[K,M]
//
// So we compute: D[N,M] = B[N,K] * A^T[K,M]
//   "A" param = B (weight), col-major [N,K], ld=K, CUBLAS_OP_N
//   "B" param = A (input),  col-major [K,M], which is row-major [M,K]^T
//              -> stored as row-major [M,K] with ld=K, use CUBLAS_OP_T to transpose
//              -> cuBLAS sees [M,K]^T = [K,M], ld=K (ld >= K rows, always valid)
//   "C" param = C (output), col-major [N,M], ld=N
//              -> row-major [M,N] with ld=N, which is col-major [N,M] with ld=N
// ============================================================================

void linear(std::byte *output, const std::byte *input, const std::byte *weight,
            const std::byte *bias, zedinferDataType_t type,
            size_t M, size_t N, size_t K, cudaStream_t stream) {

    auto lt = handle();
    auto cuda_dt = to_cuda_dtype(type);
    bool has_bias = (bias != nullptr);

    // Matmul descriptor
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Row-major to col-major mapping:
    //   Row-major X[R,C] in memory = col-major [C,R] with ld=C
    //
    // We want: C[M,N] = A[M,K] * B[N,K]^T   (all row-major)
    // In col-major: C_col[N,M] = "A_cublas"[N,K] * "B_cublas"[K,M]
    //
    // "A_cublas" = [N,K]: weight row-major [N,K] → col-major [K,N] ld=K, OP_T → [N,K]
    // "B_cublas" = [K,M]: input  row-major [M,K] → col-major [K,M] ld=K, OP_N → [K,M]
    // "C_cublas" = [N,M]: output row-major [M,N] → col-major [N,M] ld=N
    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    // Bias epilogue: bias is [N], broadcast along M dimension
    if (has_bias) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                        &epilogue, sizeof(epilogue));
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                        &bias, sizeof(bias));
        cudaDataType_t biasDt = cuda_dt;
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                        &biasDt, sizeof(biasDt));
    }

    // Column-major layouts (row-major data reinterpreted):
    // "A" = weight: row-major [N,K] → col-major [K,N], ld=K. OP_T gives [N,K].
    //               ld >= rows before transpose = K. ld=K ✓
    // "B" = input:  row-major [M,K] → col-major [K,M], ld=K. OP_N gives [K,M].
    //               ld >= rows = K. ld=K ✓
    // "C" = output: row-major [M,N] → col-major [N,M], ld=N.
    //               ld >= rows = N. ld=N ✓
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatrixLayoutCreate(&layoutA, cuda_dt, K, N, K);   // weight: col-major [K,N], ld=K
    cublasLtMatrixLayoutCreate(&layoutB, cuda_dt, K, M, K);   // input:  col-major [K,M], ld=K
    cublasLtMatrixLayoutCreate(&layoutC, cuda_dt, N, M, N);   // output: col-major [N,M], ld=N

    float alpha = 1.0f, beta = 0.0f;

    // Heuristic (cached by shape)
    MatmulKey key{M, N, K, type, has_bias};
    cublasLtMatmulAlgo_t *algo_ptr = nullptr;

    {
        std::lock_guard<std::mutex> lock(algo_mutex);
        auto it = algo_cache.find(key);
        if (it != algo_cache.end() && it->second.valid) {
            algo_ptr = &it->second.algo;
        }
    }

    if (!algo_ptr) {
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &WORKSPACE_SIZE, sizeof(WORKSPACE_SIZE));

        cublasLtMatmulHeuristicResult_t result;
        int returnedResults = 0;
        cublasLtMatmulAlgoGetHeuristic(lt, matmulDesc, layoutA, layoutB, layoutC, layoutC,
                                        pref, 1, &result, &returnedResults);
        cublasLtMatmulPreferenceDestroy(pref);

        if (returnedResults > 0) {
            std::lock_guard<std::mutex> lock(algo_mutex);
            auto &cached = algo_cache[key];
            cached.algo = result.algo;
            cached.valid = true;
            algo_ptr = &cached.algo;
        }
    }

    // Execute: "A" = weight, "B" = input (swapped from math notation)
    cublasStatus_t status = cublasLtMatmul(
        lt, matmulDesc,
        &alpha,
        weight, layoutA,   // "A" = weight [N,K]
        input,  layoutB,   // "B" = input  [M,K], transposed to [K,M]
        &beta,
        output, layoutC,   // "C" = output [N,M] col-major = [M,N] row-major
        output, layoutC,
        algo_ptr,
        workspace(), WORKSPACE_SIZE,
        stream);

    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLASLt matmul failed with status " + std::to_string(status));
    }
}

} // namespace zedinfer::ops::nvidia::cublas
