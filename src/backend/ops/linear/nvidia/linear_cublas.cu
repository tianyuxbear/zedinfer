#include "backend/ops/linear/nvidia/linear_cublas.cuh"
#include "utils/check.hpp"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace zedinfer::ops::nvidia::cublas {

// ============================================================================
// Singleton handle + workspace
// ============================================================================

static cublasLtHandle_t& handle() {
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

static void*& workspace() {
    static void* ws = []() {
        void* ptr = nullptr;
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
        case ZEDINFER_DTYPE_F32:
            return CUDA_R_32F;
        case ZEDINFER_DTYPE_F16:
            return CUDA_R_16F;
        case ZEDINFER_DTYPE_BF16:
            return CUDA_R_16BF;
        default:
            throw std::runtime_error("Unsupported dtype for cuBLASLt linear");
    }
}

// ============================================================================
// Descriptor/layout cache
// ============================================================================

static void check_cublas(cublasStatus_t status, const char* op) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(op) + " failed with status " + std::to_string(static_cast<int>(status)));
    }
}

struct CachedAlgo {
    cublasLtMatmulAlgo_t algo;
    bool valid = false;
};

struct MatmulPlanKey {
    size_t M, N, K;
    zedinferDataType_t dtype;
    bool has_bias;

    bool operator==(const MatmulPlanKey& o) const {
        return M == o.M && N == o.N && K == o.K && dtype == o.dtype && has_bias == o.has_bias;
    }
};

struct MatmulPlanKeyHash {
    size_t operator()(const MatmulPlanKey& k) const {
        size_t h = k.M * 2654435761u;
        h ^= k.N * 40503u;
        h ^= k.K * 12289u;
        h ^= static_cast<size_t>(k.dtype) << 3;
        h ^= static_cast<size_t>(k.has_bias) << 7;
        return h;
    }
};

struct MatmulPlan {
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatrixLayout_t layoutA = nullptr;
    cublasLtMatrixLayout_t layoutB = nullptr;
    cublasLtMatrixLayout_t layoutC = nullptr;
    CachedAlgo algo;
    std::mutex mutex;

    ~MatmulPlan() {
        if (layoutA != nullptr) {
            cublasLtMatrixLayoutDestroy(layoutA);
        }
        if (layoutB != nullptr) {
            cublasLtMatrixLayoutDestroy(layoutB);
        }
        if (layoutC != nullptr) {
            cublasLtMatrixLayoutDestroy(layoutC);
        }
        if (matmul_desc != nullptr) {
            cublasLtMatmulDescDestroy(matmul_desc);
        }
    }
};

static std::unordered_map<MatmulPlanKey, std::shared_ptr<MatmulPlan>, MatmulPlanKeyHash> plan_cache;
static std::mutex plan_mutex;

static std::shared_ptr<MatmulPlan> get_or_create_plan(zedinferDataType_t type, size_t M, size_t N, size_t K,
                                                      bool has_bias) {
    const MatmulPlanKey key{M, N, K, type, has_bias};
    {
        std::lock_guard<std::mutex> lock(plan_mutex);
        auto it = plan_cache.find(key);
        if (it != plan_cache.end()) {
            return it->second;
        }
    }

    auto plan = std::make_shared<MatmulPlan>();
    const auto cuda_dt = to_cuda_dtype(type);

    check_cublas(cublasLtMatmulDescCreate(&plan->matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                 "cublasLtMatmulDescCreate");

    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    check_cublas(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)),
                 "cublasLtMatmulDescSetAttribute(TRANSA)");
    check_cublas(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)),
                 "cublasLtMatmulDescSetAttribute(TRANSB)");

    if (has_bias) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        check_cublas(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                                    sizeof(epilogue)),
                     "cublasLtMatmulDescSetAttribute(EPILOGUE)");
        check_cublas(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &cuda_dt,
                                                    sizeof(cuda_dt)),
                     "cublasLtMatmulDescSetAttribute(BIAS_DATA_TYPE)");
    }

    check_cublas(cublasLtMatrixLayoutCreate(&plan->layoutA, cuda_dt, K, N, K), "cublasLtMatrixLayoutCreate(A)");
    check_cublas(cublasLtMatrixLayoutCreate(&plan->layoutB, cuda_dt, K, M, K), "cublasLtMatrixLayoutCreate(B)");
    check_cublas(cublasLtMatrixLayoutCreate(&plan->layoutC, cuda_dt, N, M, N), "cublasLtMatrixLayoutCreate(C)");

    cublasLtMatmulPreference_t pref = nullptr;
    check_cublas(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
    check_cublas(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WORKSPACE_SIZE,
                                                      sizeof(WORKSPACE_SIZE)),
                 "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE_BYTES)");

    cublasLtMatmulHeuristicResult_t result;
    int returned_results = 0;
    auto heuristic_status
        = cublasLtMatmulAlgoGetHeuristic(handle(), plan->matmul_desc, plan->layoutA, plan->layoutB, plan->layoutC,
                                         plan->layoutC, pref, 1, &result, &returned_results);
    cublasLtMatmulPreferenceDestroy(pref);

    if (heuristic_status == CUBLAS_STATUS_SUCCESS && returned_results > 0) {
        plan->algo.algo = result.algo;
        plan->algo.valid = true;
    }

    std::lock_guard<std::mutex> lock(plan_mutex);
    auto [it, inserted] = plan_cache.emplace(key, plan);
    return inserted ? plan : it->second;
}

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

void linear(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
            zedinferDataType_t type, size_t M, size_t N, size_t K, cudaStream_t stream) {
    auto lt = handle();
    auto plan = get_or_create_plan(type, M, N, K, bias != nullptr);

    float alpha = 1.0f, beta = 0.0f;
    auto* algo_ptr = plan->algo.valid ? &plan->algo.algo : nullptr;

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    if (bias != nullptr) {
        const void* bias_ptr = bias;
        std::lock_guard<std::mutex> lock(plan->mutex);
        check_cublas(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
                                                    sizeof(bias_ptr)),
                     "cublasLtMatmulDescSetAttribute(BIAS_POINTER)");
        status = cublasLtMatmul(lt, plan->matmul_desc, &alpha, weight, plan->layoutA, // "A" = weight [N,K]
                                input, plan->layoutB,         // "B" = input [M,K], transposed to [K,M]
                                &beta, output, plan->layoutC, // "C" = output [N,M] col-major = [M,N] row-major
                                output, plan->layoutC, algo_ptr, workspace(), WORKSPACE_SIZE, stream);
    } else {
        // Execute: "A" = weight, "B" = input (swapped from math notation)
        status = cublasLtMatmul(lt, plan->matmul_desc, &alpha, weight, plan->layoutA, // "A" = weight [N,K]
                                input, plan->layoutB,         // "B" = input [M,K], transposed to [K,M]
                                &beta, output, plan->layoutC, // "C" = output [N,M] col-major = [M,N] row-major
                                output, plan->layoutC, algo_ptr, workspace(), WORKSPACE_SIZE, stream);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLASLt matmul failed with status " + std::to_string(status));
    }
}

} // namespace zedinfer::ops::nvidia::cublas
