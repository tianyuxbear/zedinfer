#include "backend/ops/mamba/ssu.hpp"

#ifdef USE_FLASHINFER

#include "backend/core/context/context.hpp"
#include "backend/core/storage/storage.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// FlashInfer's invoke_selective_state_update_mtp.cuh expects four `constexpr int`
// globals and two index type aliases (cuSeqlensIndex_t / numAcceptedIndex_t).
// Upstream's selective_state_update_config.inc supplies them via codegen — we
// mirror that contract here. Using `constexpr int` (not `#define`) because the
// Mamba headers also use DIM/DSTATE as template parameters elsewhere, which a
// macro would shadow. Values match the M0 link probe known to compile + link
// (tests/integration/test_flashinfer_ssu_link.cu).
constexpr int DIM = 128;
constexpr int DSTATE = 128;
constexpr int NTOKENS_MTP = 1;
constexpr int PHILOX_ROUNDS = 0;
using cuSeqlensIndex_t = int32_t;
using numAcceptedIndex_t = int32_t;

#include <flashinfer/mamba/selective_state_update.cuh>

#include <stdexcept>
#include <string>

namespace zedinfer::ops::mamba {

namespace {

// Thread-local pinned-host + device scratch for cu_seqlens (varlen path).
// Reused across calls so each layer's SSU forward doesn't allocate or sync.
// Lifetime: persists until thread exit; matches the single inference thread
// model used by Scheduler. Lazy initialization keeps the cost off the cold
// path until the first varlen call.
struct CuSeqlensScratch {
    core::storage_t host_pinned; // 2 * int32 in pinned host memory
    core::storage_t device;      // 2 * int32 on the compute device
};

thread_local CuSeqlensScratch g_cu_seqlens;

int32_t* prepare_cu_seqlens(int N, cudaStream_t stream) {
    auto& runtime = core::context().runtime();
    constexpr size_t kBytes = 2 * sizeof(int32_t);

    if (!g_cu_seqlens.host_pinned) {
        g_cu_seqlens.host_pinned = runtime.allocateHostStorage(kBytes);
    }
    if (!g_cu_seqlens.device) {
        g_cu_seqlens.device = runtime.allocateDeviceStorage(kBytes);
    }

    auto* host = reinterpret_cast<int32_t*>(g_cu_seqlens.host_pinned->memory());
    host[0] = 0;
    host[1] = N;

    auto status = cudaMemcpyAsync(g_cu_seqlens.device->memory(), host, kBytes,
                                  cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("[ops::mamba::ssu] cu_seqlens H2D copy failed: ")
                                 + cudaGetErrorString(status));
    }
    return reinterpret_cast<int32_t*>(g_cu_seqlens.device->memory());
}

// Offset a raw byte base pointer by a signed byte count. The state base lives
// in device memory and is opaque to the host — pointer arithmetic in `char*`
// units is the standard idiom for these byte strides.
void* byte_offset(void* base, int64_t bytes) {
    return reinterpret_cast<char*>(base) + bytes;
}

} // namespace

void ssu(const SSUParams& p) {
    if (p.num_tokens <= 0) {
        throw std::runtime_error("[ops::mamba::ssu] num_tokens must be >= 1");
    }
    if (!p.state_view.ssm_base) {
        throw std::runtime_error("[ops::mamba::ssu] state_view.ssm_base is null");
    }

    auto& runtime = core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());

    // Per-token row stride (in input_t/bf16 elements) of the contiguous
    // [N, hidden] activations passed in. The kernel reads B-stride for the
    // between-sequence offset and reuses it for the between-token offset in
    // varlen mode (see kernel_selective_state_update_mtp_simple.cuh).
    const int64_t hidden = static_cast<int64_t>(p.state_view.num_v_heads)
                         * static_cast<int64_t>(p.state_view.value_head_dim);
    const int64_t bc_row = static_cast<int64_t>(1) * static_cast<int64_t>(p.state_view.d_state);
    const int64_t dt_row = static_cast<int64_t>(p.state_view.num_v_heads);

    // State pointer is offset to (slot_idx, layer_idx). Per-batch state stride
    // is 0 because we always run with batch=1 / a single slot per call.
    void* state_ptr = byte_offset(p.state_view.ssm_base,
                                  static_cast<int64_t>(p.slot_idx) * p.state_view.ssm_stride_slot
                                + static_cast<int64_t>(p.layer_idx) * p.state_view.ssm_stride_layer);

    flashinfer::mamba::mtp::SelectiveStateMTPParams params{};
    params.batch = 1;
    params.nheads = static_cast<uint32_t>(p.state_view.num_v_heads);
    params.ngroups = 1;
    params.dim = static_cast<uint32_t>(p.state_view.value_head_dim);
    params.dstate = static_cast<uint32_t>(p.state_view.d_state);
    params.state_cache_size = 1;
    params.ntokens_mtp = static_cast<uint32_t>(p.num_tokens);
    params.dt_softplus = true;

    params.x        = p.v ? p.v->data() : nullptr;
    params.dt       = p.a ? p.a->data() : nullptr;
    params.A        = p.A_log ? p.A_log->data() : nullptr;
    params.B        = p.b ? p.b->data() : nullptr;
    params.C        = p.q ? p.q->data() : nullptr;
    params.D        = nullptr; // Qwen3.5 linear-attn block has no D skip-connect.
    params.z        = p.z ? p.z->data() : nullptr;
    params.dt_bias  = p.dt_bias ? p.dt_bias->data() : nullptr;
    params.output   = p.out ? p.out->data() : nullptr;

    // Strides are in element units (input_t/state_t), matching the kernel's
    // reinterpret_cast + pointer-arithmetic access pattern.
    params.x_stride_batch   = hidden;
    params.dt_stride_batch  = dt_row;
    params.B_stride_batch   = bc_row;
    params.C_stride_batch   = bc_row;
    params.out_stride_batch = hidden;
    params.z_stride_batch   = hidden;
    params.state_stride_batch = 0;

    // MTP strides — unused when TOKENS_MTP==1, but set for safety in case the
    // dispatch reaches a kernel that reads them on a non-varlen path.
    params.x_stride_mtp   = hidden;
    params.dt_stride_mtp  = dt_row;
    params.B_stride_mtp   = bc_row;
    params.C_stride_mtp   = bc_row;
    params.out_stride_mtp = hidden;
    params.z_stride_mtp   = hidden;

    params.state = state_ptr;
    params.state_batch_indices = nullptr;
    params.dst_state_batch_indices = nullptr;
    params.intermediate_states = nullptr;
    params.intermediate_state_indices = nullptr;
    params.intermediate_state_scales = nullptr;
    params.num_accepted_tokens = nullptr;
    params.state_scale = nullptr;
    params.rand_seed = nullptr;

    // Varlen prefill carries cu_seqlens=[0, N]; decode (N==1) passes nullptr
    // and the kernel takes the fixed-token MTP path.
    params.cu_seqlens = (p.num_tokens > 1)
                          ? static_cast<void*>(prepare_cu_seqlens(p.num_tokens, stream))
                          : nullptr;

    flashinfer::mamba::mtp::invokeSelectiveStateUpdateMTP<
        __nv_bfloat16,  // input_t
        __nv_bfloat16,  // weight_t (dt_bias, D)
        float,          // matrixA_t (A_log)
        __nv_bfloat16,  // state_t
        int32_t,        // stateIndex_t
        void            // state_scale_t (no quantized state)
    >(params, flashinfer::mamba::SSUAlgorithm::kAuto, stream);
}

} // namespace zedinfer::ops::mamba

#else

#include <stdexcept>

namespace zedinfer::ops::mamba {

void ssu(const SSUParams&) {
    throw std::runtime_error("ops::mamba::ssu requires --flashinfer=y build");
}

} // namespace zedinfer::ops::mamba

#endif
