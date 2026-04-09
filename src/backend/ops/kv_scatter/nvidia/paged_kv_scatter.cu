#include "backend/ops/kv_scatter/nvidia/paged_kv_scatter.cuh"

#include "backend/core/context/context.hpp"

#include <cuda_runtime.h>

#include <cstdint>
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

void check_cuda(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("[paged_kv_scatter] ") + op + " failed: " + cudaGetErrorString(status));
    }
}

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

} // namespace zedinfer::ops::nvidia
