#include "backend/ops/self_attention/nvidia/flashinfer_wrapper.cuh"

#ifdef USE_FLASHINFER

#include "backend/core/context/context.hpp"
#include "backend/core/storage/storage.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#include <flashinfer/allocator.h>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>

namespace zedinfer::ops::nvidia {

namespace {

using DefaultAttention = flashinfer::DefaultAttention<false, false, false, false>;

template <typename T> T* ptr_from_offset(void* base, int64_t offset) {
    return flashinfer::GetPtrFromBaseOffset<T>(base, offset);
}

void check_cuda_status(cudaError_t status, std::string_view op_name) {
    if (status != cudaSuccess) {
        throw std::runtime_error("[FlashInfer] " + std::string(op_name) + " failed: " + cudaGetErrorString(status));
    }
}

bool is_workspace_overflow(const flashinfer::Error& err) {
    std::string_view msg(err.what());
    return msg.find("Increase the workspace buffer size") != std::string_view::npos
        || msg.find("Buffer overflow when allocating memory") != std::string_view::npos;
}

template <typename Func> void run_with_workspace_retry(size_t float_bytes, size_t int_bytes, Func&& func) {
    auto& runtime = core::context().runtime();

    float_bytes = std::max<size_t>(float_bytes, 16);
    int_bytes = std::max<size_t>(int_bytes, 16);

    for (int attempt = 0; attempt < 4; ++attempt) {
        auto float_ws = runtime.allocateDeviceStorage(float_bytes);
        auto int_ws = runtime.allocateDeviceStorage(int_bytes);
        auto host_int_ws = runtime.allocateHostStorage(int_bytes);

        try {
            func(float_ws->memory(), float_bytes, int_ws->memory(), host_int_ws->memory(), int_bytes);
            return;
        } catch (const flashinfer::Error& err) {
            if (!is_workspace_overflow(err) || attempt == 3) {
                throw;
            }
            float_bytes *= 2;
            int_bytes *= 2;
        }
    }
}

template <typename DType> flashinfer::paged_kv_t<DType, int32_t> make_paged_kv(const AttentionParams& p) {
    const auto& c = p.config;
    return flashinfer::paged_kv_t<DType, int32_t>(
        static_cast<uint32_t>(c.nkvhead), static_cast<uint32_t>(c.block_size), static_cast<uint32_t>(c.head_dim),
        static_cast<uint32_t>(p.kv_batch_size), flashinfer::QKVLayout::kNHD,
        reinterpret_cast<DType*>(const_cast<void*>(p.k_pool_base)),
        reinterpret_cast<DType*>(const_cast<void*>(p.v_pool_base)),
        const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.kv_page_indices)),
        const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.kv_indptr)),
        const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.kv_last_page_len)));
}

template <typename DType, uint32_t HEAD_DIM> void run_decode_head_dim(const AttentionParams& p, cudaStream_t stream) {
    using Params = flashinfer::BatchDecodeParams<DType, DType, DType, int32_t>;

    const auto& c = p.config;
    auto paged_kv = make_paged_kv<DType>(p);
    Params params(reinterpret_cast<DType*>(p.q->data()), nullptr, paged_kv, reinterpret_cast<DType*>(p.out->data()),
                  nullptr, nullptr, static_cast<uint32_t>(c.nhead), static_cast<int32_t>(c.nhead * c.head_dim),
                  static_cast<int32_t>(c.head_dim), -1, 0.0f, c.scale, 1.0f, 10000.0f);

    // A runtime escape hatch is kept for validating the planner path without rebuilding.
    if (p.kv_batch_size == 1 && std::getenv("ZEDINFER_FLASHINFER_DISABLE_FASTPATH") == nullptr) {
        // Single-request decode does not need FlashInfer's scheduling plan. Reuse the
        // trivial descriptor materialized by PagedForwardContext so its lifetime spans the
        // whole forward pass and cannot be recycled by the memory pool mid-kernel.
        int32_t local_descriptor_host[5] = {0, 0, 0, 1, 1};
        auto& runtime = core::context().runtime();
        auto local_descriptor_storage
            = (!p.fi_request_indices || !p.fi_kv_tile_indices || !p.fi_o_indptr || !p.fi_kv_chunk_size_ptr)
                ? runtime.allocateDeviceStorage(5 * sizeof(int32_t))
                : nullptr;
        if (local_descriptor_storage) {
            runtime.api()->memcpy_sync(local_descriptor_storage->memory(), local_descriptor_host,
                                       sizeof(local_descriptor_host), ZEDINFER_MEMCPY_H2D);
        }
        auto* descriptor
            = local_descriptor_storage ? reinterpret_cast<int32_t*>(local_descriptor_storage->memory()) : nullptr;
        params.padded_batch_size = 1;
        params.request_indices
            = descriptor ? descriptor : const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.fi_request_indices));
        params.kv_tile_indices = descriptor
                                   ? descriptor + 1
                                   : const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.fi_kv_tile_indices));
        params.o_indptr
            = descriptor ? descriptor + 2 : const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.fi_o_indptr));
        params.kv_chunk_size_ptr = descriptor
                                     ? descriptor + 4
                                     : const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.fi_kv_chunk_size_ptr));
        params.block_valid_mask = nullptr;

        check_cuda_status(
            flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, flashinfer::PosEncodingMode::kNone,
                                                              DefaultAttention, Params>(params, nullptr, nullptr, false,
                                                                                        stream),
            "BatchDecodeWithPagedKVCacheDispatched");
        return;
    }

    const int total_pages = p.kv_indptr_host[p.kv_batch_size];
    const size_t float_bytes
        = std::max<size_t>(4096, static_cast<size_t>(c.nhead) * std::max(total_pages, p.kv_batch_size)
                                     * (static_cast<size_t>(c.head_dim) + 1) * sizeof(float));
    const size_t int_bytes = std::max<size_t>(
        4096, (static_cast<size_t>(total_pages) * 3 + static_cast<size_t>(p.kv_batch_size) * 2 + 64) * sizeof(int)
                  + static_cast<size_t>(total_pages + p.kv_batch_size + 64) * sizeof(bool));

    run_with_workspace_retry(
        float_bytes, int_bytes,
        [&](void* float_buf, size_t float_size, void* int_buf, void* host_int_buf, size_t int_size) {
            flashinfer::DecodePlanInfo plan_info;
            auto work_estimation = [&](bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
                                       uint32_t& new_batch_size, uint32_t& gdy, uint32_t batch_size,
                                       int32_t* kv_indptr_h, uint32_t num_qo_heads, uint32_t page_size,
                                       bool enable_cuda_graph, cudaStream_t work_stream) -> cudaError_t {
                cudaError_t status = cudaSuccess;
                const int group_size = c.nhead / c.nkvhead;
                DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, {
                    status = flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
                        GROUP_SIZE, HEAD_DIM, flashinfer::PosEncodingMode::kNone, DefaultAttention, Params>(
                        split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy, batch_size, kv_indptr_h,
                        num_qo_heads, page_size, enable_cuda_graph, work_stream);
                });
                return status;
            };

            check_cuda_status(
                flashinfer::DecodePlan<HEAD_DIM, flashinfer::PosEncodingMode::kNone, DefaultAttention, Params>(
                    float_buf, float_size, int_buf, host_int_buf, int_size, plan_info,
                    const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.kv_indptr_host)),
                    static_cast<uint32_t>(p.kv_batch_size), static_cast<uint32_t>(c.nhead),
                    static_cast<uint32_t>(c.block_size), false, stream, work_estimation),
                "DecodePlan");

            params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
            params.request_indices = ptr_from_offset<int32_t>(int_buf, plan_info.request_indices_offset);
            params.kv_tile_indices = ptr_from_offset<int32_t>(int_buf, plan_info.kv_tile_indices_offset);
            params.o_indptr = ptr_from_offset<int32_t>(int_buf, plan_info.o_indptr_offset);
            params.kv_chunk_size_ptr = ptr_from_offset<int32_t>(int_buf, plan_info.kv_chunk_size_ptr_offset);
            params.block_valid_mask
                = plan_info.split_kv ? ptr_from_offset<bool>(int_buf, plan_info.block_valid_mask_offset) : nullptr;

            auto* tmp_v = plan_info.split_kv ? ptr_from_offset<DType>(float_buf, plan_info.v_offset) : nullptr;
            auto* tmp_s = plan_info.split_kv ? ptr_from_offset<float>(float_buf, plan_info.s_offset) : nullptr;

            check_cuda_status(
                flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, flashinfer::PosEncodingMode::kNone,
                                                                  DefaultAttention, Params>(params, tmp_v, tmp_s, false,
                                                                                            stream),
                "BatchDecodeWithPagedKVCacheDispatched");
        });
}

template <typename DType> void run_decode_impl(const AttentionParams& p) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    switch (p.config.head_dim) {
        case 64:
            return run_decode_head_dim<DType, 64>(p, stream);
        case 128:
            return run_decode_head_dim<DType, 128>(p, stream);
        case 256:
            return run_decode_head_dim<DType, 256>(p, stream);
        default:
            throw std::runtime_error("[FlashInfer] Unsupported decode head_dim: " + std::to_string(p.config.head_dim));
    }
}

template <typename DType, uint32_t HEAD_DIM> void run_prefill_head_dim(const AttentionParams& p, cudaStream_t stream) {
    using Params = flashinfer::BatchPrefillPagedParams<DType, DType, DType, int32_t>;

    const auto& c = p.config;
    auto paged_kv = make_paged_kv<DType>(p);
    Params params(reinterpret_cast<DType*>(p.q->data()), paged_kv, nullptr,
                  const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.qo_indptr)), nullptr, nullptr,
                  reinterpret_cast<DType*>(p.out->data()), nullptr, nullptr, static_cast<uint32_t>(c.nhead),
                  static_cast<int32_t>(c.nhead * c.head_dim), static_cast<int32_t>(c.head_dim), -1, 0.0f, c.scale, 1.0f,
                  10000.0f);

    const int total_pages = p.kv_indptr_host[p.kv_batch_size];
    const int total_q_rows = p.qo_indptr_host[p.kv_batch_size];
    const int group_size = c.nhead / c.nkvhead;
    const size_t float_bytes = 16; // prefill v1 keeps split-kv disabled
    const size_t int_bytes = std::max<size_t>(4096, (static_cast<size_t>(total_q_rows) * std::max(group_size, 1)
                                                     + static_cast<size_t>(total_pages) * 2
                                                     + static_cast<size_t>(p.kv_batch_size) * 8 + 128)
                                                        * sizeof(int));

    run_with_workspace_retry(
        float_bytes, int_bytes,
        [&](void* float_buf, size_t float_size, void* int_buf, void* host_int_buf, size_t int_size) {
            flashinfer::PrefillPlanInfo plan_info;
            check_cuda_status(flashinfer::PrefillPlan<int32_t>(
                                  float_buf, float_size, int_buf, host_int_buf, int_size, plan_info,
                                  const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.qo_indptr_host)),
                                  const_cast<int32_t*>(reinterpret_cast<const int32_t*>(p.kv_indptr_host)),
                                  static_cast<uint32_t>(total_q_rows), static_cast<uint32_t>(p.kv_batch_size),
                                  static_cast<uint32_t>(c.nhead), static_cast<uint32_t>(c.nkvhead),
                                  static_cast<uint32_t>(c.head_dim), static_cast<uint32_t>(c.head_dim),
                                  static_cast<uint32_t>(c.block_size), false, sizeof(DType), -1, 0, true, 0, stream),
                              "PrefillPlan");

            params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
            params.request_indices = ptr_from_offset<int32_t>(int_buf, plan_info.request_indices_offset);
            params.qo_tile_indices = ptr_from_offset<int32_t>(int_buf, plan_info.qo_tile_indices_offset);
            params.kv_tile_indices = ptr_from_offset<int32_t>(int_buf, plan_info.kv_tile_indices_offset);
            params.o_indptr = ptr_from_offset<int32_t>(int_buf, plan_info.o_indptr_offset);
            params.kv_chunk_size_ptr = ptr_from_offset<int32_t>(int_buf, plan_info.kv_chunk_size_ptr_offset);
            params.max_total_num_rows = static_cast<uint32_t>(total_q_rows);
            params.total_num_rows = nullptr;
            params.merge_indptr
                = plan_info.split_kv ? ptr_from_offset<int32_t>(int_buf, plan_info.merge_indptr_offset) : nullptr;
            params.block_valid_mask
                = plan_info.split_kv ? ptr_from_offset<bool>(int_buf, plan_info.block_valid_mask_offset) : nullptr;

            auto* tmp_v = plan_info.split_kv ? ptr_from_offset<DType>(float_buf, plan_info.v_offset) : nullptr;
            auto* tmp_s = plan_info.split_kv ? ptr_from_offset<float>(float_buf, plan_info.s_offset) : nullptr;

            DISPATCH_CTA_TILE_Q(static_cast<uint32_t>(plan_info.cta_tile_q), CTA_TILE_Q, {
                check_cuda_status(
                    flashinfer::BatchPrefillWithPagedKVCacheDispatched<
                        CTA_TILE_Q, HEAD_DIM, HEAD_DIM, flashinfer::PosEncodingMode::kNone, false,
                        flashinfer::MaskMode::kCausal, DefaultAttention, Params>(params, tmp_v, tmp_s, false, stream),
                    "BatchPrefillWithPagedKVCacheDispatched");
            });
        });
}

template <typename DType> void run_prefill_impl(const AttentionParams& p) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    switch (p.config.head_dim) {
        case 64:
            return run_prefill_head_dim<DType, 64>(p, stream);
        case 128:
            return run_prefill_head_dim<DType, 128>(p, stream);
        case 256:
            return run_prefill_head_dim<DType, 256>(p, stream);
        default:
            throw std::runtime_error("[FlashInfer] Unsupported prefill head_dim: " + std::to_string(p.config.head_dim));
    }
}

void validate_flashinfer_params(const AttentionParams& p, bool is_prefill) {
    const auto& c = p.config;
    if (c.device_type != ZEDINFER_DEVICE_NVIDIA) {
        throw std::runtime_error("[FlashInfer] NVIDIA device required");
    }
    if (p.k_pool_base == nullptr || p.v_pool_base == nullptr) {
        throw std::runtime_error("[FlashInfer] Missing KV pool base pointers");
    }
    if (p.kv_indptr == nullptr || p.kv_page_indices == nullptr || p.kv_last_page_len == nullptr
        || p.kv_indptr_host == nullptr || p.kv_batch_size <= 0) {
        throw std::runtime_error("[FlashInfer] Incomplete KV CSR metadata");
    }
    if (c.nkvhead <= 0 || c.nhead <= 0 || c.nhead % c.nkvhead != 0) {
        throw std::runtime_error("[FlashInfer] Unsupported GQA ratio");
    }
    if (is_prefill && (p.qo_indptr == nullptr || p.qo_indptr_host == nullptr)) {
        throw std::runtime_error("[FlashInfer] Prefill requires qo_indptr CSR metadata");
    }
}

template <typename Func> void wrap_flashinfer_call(std::string_view name, Func&& func) {
    try {
        func();
    } catch (const flashinfer::Error& err) {
        throw std::runtime_error("[FlashInfer] " + std::string(name) + " error: " + err.what());
    }
}

} // namespace

void flashinfer_attention_decode(const AttentionParams& params) {
    wrap_flashinfer_call("decode", [&] {
        validate_flashinfer_params(params, false);
        switch (params.config.dtype) {
            case ZEDINFER_DTYPE_F16:
                return run_decode_impl<half>(params);
            case ZEDINFER_DTYPE_BF16:
                return run_decode_impl<nv_bfloat16>(params);
            default:
                throw std::runtime_error("[FlashInfer] Unsupported decode dtype");
        }
    });
}

void flashinfer_attention_prefill(const AttentionParams& params) {
    wrap_flashinfer_call("prefill", [&] {
        validate_flashinfer_params(params, true);
        switch (params.config.dtype) {
            case ZEDINFER_DTYPE_F16:
                return run_prefill_impl<half>(params);
            case ZEDINFER_DTYPE_BF16:
                return run_prefill_impl<nv_bfloat16>(params);
            default:
                throw std::runtime_error("[FlashInfer] Unsupported prefill dtype");
        }
    });
}

} // namespace zedinfer::ops::nvidia

#endif
