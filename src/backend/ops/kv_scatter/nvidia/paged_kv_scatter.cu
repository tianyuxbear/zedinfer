#include "backend/ops/kv_scatter/nvidia/paged_kv_scatter.cuh"

#include "backend/core/context/context.hpp"

#ifdef USE_FLASHINFER
#include <flashinfer/page.cuh>
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace zedinfer::ops::nvidia {

namespace {

__global__ void scatter_paged_kv_vec16_kernel(const uint4* __restrict__ k_src, const uint4* __restrict__ v_src,
                                              uint4* __restrict__ k_pool, uint4* __restrict__ v_pool,
                                              const int* __restrict__ page_table, int block_size, int past_len,
                                              int token_vecs) {
    const int token_idx = blockIdx.x;
    const int global_pos = past_len + token_idx;
    const int logical_page = global_pos / block_size;
    const int token_offset = global_pos % block_size;
    const int physical_page = page_table[logical_page];

    const size_t src_base = static_cast<size_t>(token_idx) * token_vecs;
    const size_t dst_base = (static_cast<size_t>(physical_page) * block_size + token_offset) * token_vecs;

    for (int vec_idx = threadIdx.x; vec_idx < token_vecs; vec_idx += blockDim.x) {
        k_pool[dst_base + vec_idx] = k_src[src_base + vec_idx];
        v_pool[dst_base + vec_idx] = v_src[src_base + vec_idx];
    }
}

__global__ void scatter_paged_kv_bytes_kernel(const std::byte* __restrict__ k_src, const std::byte* __restrict__ v_src,
                                              std::byte* __restrict__ k_pool, std::byte* __restrict__ v_pool,
                                              const int* __restrict__ page_table, int block_size, int past_len,
                                              int token_bytes) {
    const int token_idx = blockIdx.x;
    const int global_pos = past_len + token_idx;
    const int logical_page = global_pos / block_size;
    const int token_offset = global_pos % block_size;
    const int physical_page = page_table[logical_page];

    const size_t src_base = static_cast<size_t>(token_idx) * token_bytes;
    const size_t dst_base = (static_cast<size_t>(physical_page) * block_size + token_offset) * token_bytes;

    for (int byte_idx = threadIdx.x; byte_idx < token_bytes; byte_idx += blockDim.x) {
        k_pool[dst_base + byte_idx] = k_src[src_base + byte_idx];
        v_pool[dst_base + byte_idx] = v_src[src_base + byte_idx];
    }
}

inline bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

int flashinfer_vec_size(zedinferDataType_t dtype, int head_dim) {
    const int dtype_bytes = (dtype == ZEDINFER_DTYPE_F16 || dtype == ZEDINFER_DTYPE_BF16) ? 2 : 0;
    if (dtype_bytes == 0) {
        return 0;
    }
    return std::max(16 / dtype_bytes, head_dim / 32);
}

bool can_use_flashinfer_append(zedinferDataType_t dtype, int num_kv_heads, int head_dim, int block_size) {
#ifdef USE_FLASHINFER
    if (std::getenv("ZEDINFER_DISABLE_FLASHINFER") != nullptr) {
        return false;
    }
    if ((dtype != ZEDINFER_DTYPE_F16 && dtype != ZEDINFER_DTYPE_BF16) || num_kv_heads <= 0 || block_size <= 0) {
        return false;
    }
    switch (head_dim) {
        case 64:
        case 128:
        case 256:
        case 512:
            break;
        default:
            return false;
    }
    const int vec_size = flashinfer_vec_size(dtype, head_dim);
    return vec_size > 0 && (head_dim / vec_size) * num_kv_heads <= 1024;
#else
    (void)dtype;
    (void)num_kv_heads;
    (void)head_dim;
    (void)block_size;
    return false;
#endif
}

void check_cuda(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("[paged_kv_scatter] ") + op + " failed: " + cudaGetErrorString(status));
    }
}

#ifdef USE_FLASHINFER
template <typename DType>
flashinfer::paged_kv_t<DType, int32_t> make_paged_kv(void* k_pool_base, void* v_pool_base, const int* kv_page_indices,
                                                     const int* kv_indptr, const int* kv_last_page_len, int batch_size,
                                                     int num_kv_heads, int head_dim, int block_size) {
    return flashinfer::paged_kv_t<DType, int32_t>(
        static_cast<uint32_t>(num_kv_heads), static_cast<uint32_t>(block_size), static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(batch_size), flashinfer::QKVLayout::kNHD, reinterpret_cast<DType*>(k_pool_base),
        reinterpret_cast<DType*>(v_pool_base), const_cast<int32_t*>(reinterpret_cast<const int32_t*>(kv_page_indices)),
        const_cast<int32_t*>(reinterpret_cast<const int32_t*>(kv_indptr)),
        const_cast<int32_t*>(reinterpret_cast<const int32_t*>(kv_last_page_len)));
}

template <typename DType>
void append_paged_kv_impl(const void* k_src, const void* v_src, void* k_pool_base, void* v_pool_base,
                          const int* kv_page_indices, const int* kv_indptr, const int* kv_last_page_len,
                          const int* batch_indices, const int* positions, int batch_size, int nnz_tokens,
                          int num_kv_heads, int head_dim, int block_size) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    auto paged_kv = make_paged_kv<DType>(k_pool_base, v_pool_base, kv_page_indices, kv_indptr, kv_last_page_len,
                                         batch_size, num_kv_heads, head_dim, block_size);

    auto status = flashinfer::AppendPagedKVCache<DType, int32_t>(
        paged_kv, reinterpret_cast<DType*>(const_cast<void*>(k_src)),
        reinterpret_cast<DType*>(const_cast<void*>(v_src)),
        const_cast<int32_t*>(reinterpret_cast<const int32_t*>(batch_indices)),
        const_cast<int32_t*>(reinterpret_cast<const int32_t*>(positions)), static_cast<uint32_t>(nnz_tokens),
        static_cast<size_t>(num_kv_heads) * head_dim, static_cast<size_t>(head_dim),
        static_cast<size_t>(num_kv_heads) * head_dim, static_cast<size_t>(head_dim), stream);
    check_cuda(status, "AppendPagedKVCache");
}

template <typename DType>
void append_paged_kv_decode_impl(const void* k_src, const void* v_src, void* k_pool_base, void* v_pool_base,
                                 const int* kv_page_indices, const int* kv_indptr, const int* kv_last_page_len,
                                 int batch_size, int num_kv_heads, int head_dim, int block_size) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    auto paged_kv = make_paged_kv<DType>(k_pool_base, v_pool_base, kv_page_indices, kv_indptr, kv_last_page_len,
                                         batch_size, num_kv_heads, head_dim, block_size);

    auto status = flashinfer::AppendPagedKVCacheDecode<DType, int32_t>(
        paged_kv, reinterpret_cast<DType*>(const_cast<void*>(k_src)),
        reinterpret_cast<DType*>(const_cast<void*>(v_src)), stream);
    check_cuda(status, "AppendPagedKVCacheDecode");
}
#endif

} // namespace

void scatter_paged_kv(const void* k_src, const void* v_src, void* k_pool_base, void* v_pool_base, const int* page_table,
                      int block_size, int past_len, int num_tokens, size_t token_bytes) {
    if (k_src == nullptr || v_src == nullptr || k_pool_base == nullptr || v_pool_base == nullptr
        || page_table == nullptr || block_size <= 0 || past_len < 0 || num_tokens <= 0 || token_bytes == 0) {
        throw std::invalid_argument("[paged_kv_scatter] invalid arguments");
    }

    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());

    if ((token_bytes % sizeof(uint4)) == 0 && is_aligned_16(k_src) && is_aligned_16(v_src) && is_aligned_16(k_pool_base)
        && is_aligned_16(v_pool_base)) {
        const int token_vecs = static_cast<int>(token_bytes / sizeof(uint4));
        const int threads = token_vecs < 256 ? token_vecs : 256;
        scatter_paged_kv_vec16_kernel<<<num_tokens, threads, 0, stream>>>(
            static_cast<const uint4*>(k_src), static_cast<const uint4*>(v_src), static_cast<uint4*>(k_pool_base),
            static_cast<uint4*>(v_pool_base), page_table, block_size, past_len, token_vecs);
        check_cuda(cudaGetLastError(), "scatter_paged_kv_vec16_kernel");
        return;
    }

    scatter_paged_kv_bytes_kernel<<<num_tokens, 256, 0, stream>>>(
        static_cast<const std::byte*>(k_src), static_cast<const std::byte*>(v_src),
        static_cast<std::byte*>(k_pool_base), static_cast<std::byte*>(v_pool_base), page_table, block_size, past_len,
        static_cast<int>(token_bytes));
    check_cuda(cudaGetLastError(), "scatter_paged_kv_bytes_kernel");
}

void append_paged_kv(const void* k_src, const void* v_src, void* k_pool_base, void* v_pool_base,
                     const int* kv_page_indices, const int* kv_indptr, const int* kv_last_page_len,
                     const int* batch_indices, const int* positions, int batch_size, int nnz_tokens, int num_kv_heads,
                     int head_dim, int block_size, zedinferDataType_t dtype) {
    if (k_src == nullptr || v_src == nullptr || k_pool_base == nullptr || v_pool_base == nullptr
        || kv_page_indices == nullptr || kv_indptr == nullptr || kv_last_page_len == nullptr || batch_indices == nullptr
        || positions == nullptr || batch_size <= 0 || nnz_tokens <= 0) {
        throw std::invalid_argument("[paged_kv_scatter] invalid append_paged_kv arguments");
    }
    if (!can_use_flashinfer_append(dtype, num_kv_heads, head_dim, block_size)) {
        throw std::invalid_argument("[paged_kv_scatter] FlashInfer append path is unavailable for this configuration");
    }

#ifdef USE_FLASHINFER
    switch (dtype) {
        case ZEDINFER_DTYPE_F16:
            return append_paged_kv_impl<half>(k_src, v_src, k_pool_base, v_pool_base, kv_page_indices, kv_indptr,
                                              kv_last_page_len, batch_indices, positions, batch_size, nnz_tokens,
                                              num_kv_heads, head_dim, block_size);
        case ZEDINFER_DTYPE_BF16:
            return append_paged_kv_impl<nv_bfloat16>(k_src, v_src, k_pool_base, v_pool_base, kv_page_indices, kv_indptr,
                                                     kv_last_page_len, batch_indices, positions, batch_size, nnz_tokens,
                                                     num_kv_heads, head_dim, block_size);
        default:
            break;
    }
#endif

    throw std::invalid_argument("[paged_kv_scatter] unsupported dtype for append_paged_kv");
}

void append_paged_kv_decode(const void* k_src, const void* v_src, void* k_pool_base, void* v_pool_base,
                            const int* kv_page_indices, const int* kv_indptr, const int* kv_last_page_len,
                            int batch_size, int num_kv_heads, int head_dim, int block_size, zedinferDataType_t dtype) {
    if (k_src == nullptr || v_src == nullptr || k_pool_base == nullptr || v_pool_base == nullptr
        || kv_page_indices == nullptr || kv_indptr == nullptr || kv_last_page_len == nullptr || batch_size <= 0) {
        throw std::invalid_argument("[paged_kv_scatter] invalid append_paged_kv_decode arguments");
    }
    if (!can_use_flashinfer_append(dtype, num_kv_heads, head_dim, block_size)) {
        throw std::invalid_argument(
            "[paged_kv_scatter] FlashInfer decode append path is unavailable for this configuration");
    }

#ifdef USE_FLASHINFER
    switch (dtype) {
        case ZEDINFER_DTYPE_F16:
            return append_paged_kv_decode_impl<half>(k_src, v_src, k_pool_base, v_pool_base, kv_page_indices, kv_indptr,
                                                     kv_last_page_len, batch_size, num_kv_heads, head_dim, block_size);
        case ZEDINFER_DTYPE_BF16:
            return append_paged_kv_decode_impl<nv_bfloat16>(k_src, v_src, k_pool_base, v_pool_base, kv_page_indices,
                                                            kv_indptr, kv_last_page_len, batch_size, num_kv_heads,
                                                            head_dim, block_size);
        default:
            break;
    }
#endif

    throw std::invalid_argument("[paged_kv_scatter] unsupported dtype for append_paged_kv_decode");
}

} // namespace zedinfer::ops::nvidia
