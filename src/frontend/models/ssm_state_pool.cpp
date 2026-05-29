#include "frontend/models/ssm_state_pool.hpp"

#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "backend/ops/fill_zero/cpu/fill_zero_cpu.hpp"
#include "backend/tensor/tensor.hpp"
#include "zedinfer/activation.hpp"

#ifdef ENABLE_NVIDIA_API
#include "backend/ops/fill_zero/nvidia/fill_zero_nvidia.cuh"
#endif

#include <cstring>
#include <plog/Log.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace zedinfer::model {

namespace {

// Byte width of an SSM/conv state element. Mirrors the dtype-bytes logic in
// Tensor (`elementSize`) but is reused at the byte-stride layer where we don't
// have a Tensor handle — and we want to fail fast if M1 ever asks for an
// unsupported state dtype rather than silently overflowing strides.
size_t state_dtype_bytes(zedinferDataType_t dt) {
    switch (dt) {
        case ZEDINFER_DTYPE_F32:
            return 4;
        case ZEDINFER_DTYPE_BF16:
        case ZEDINFER_DTYPE_F16:
            return 2;
        default:
            throw std::runtime_error("SSMStatePool: unsupported state dtype " + std::to_string(static_cast<int>(dt)));
    }
}

// Dispatch a byte-level memset on whichever device backs the SSM/conv buffers.
// We deliberately avoid a runtime->memsetAsync API (which does not exist in
// runtime_api.hpp) and instead route through the existing fill_zero op so the
// pool gets CPU + NVIDIA support for free and inherits the op's stream
// selection on NVIDIA (compute stream from the active runtime).
void zero_bytes(zedinferDeviceType_t device, std::byte* ptr, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (device == ZEDINFER_DEVICE_CPU) {
        ops::cpu::fill_zero(ptr, bytes);
        return;
    }
#ifdef ENABLE_NVIDIA_API
    if (device == ZEDINFER_DEVICE_NVIDIA) {
        ops::nvidia::fill_zero(ptr, bytes);
        return;
    }
#endif
    throw std::runtime_error("SSMStatePool: zero_bytes on unsupported device "
                             + std::to_string(static_cast<int>(device)));
}

} // namespace

SSMStatePool::SSMStatePool(const SSMStatePoolConfig& cfg, const ExecutorConfig& exec)
    : cfg_(cfg), slot_in_use_(static_cast<size_t>(cfg.max_concurrent > 0 ? cfg.max_concurrent : 0), 0) {
    if (cfg.max_concurrent <= 0) {
        throw std::runtime_error("SSMStatePool: max_concurrent must be > 0 (got " + std::to_string(cfg.max_concurrent)
                                 + ")");
    }
    if (cfg.num_linear_layers <= 0) {
        throw std::runtime_error("SSMStatePool: num_linear_layers must be > 0 (got "
                                 + std::to_string(cfg.num_linear_layers) + ")");
    }
    if (cfg.conv_kernel_dim < 2) {
        throw std::runtime_error("SSMStatePool: conv_kernel_dim must be >= 2 (got "
                                 + std::to_string(cfg.conv_kernel_dim) + ")");
    }
    if (cfg.num_v_heads <= 0 || cfg.value_head_dim <= 0 || cfg.d_state <= 0 || cfg.qkv_dim <= 0) {
        throw std::runtime_error("SSMStatePool: SSM/conv dims must all be > 0");
    }

    const size_t elem_bytes = state_dtype_bytes(cfg.state_dtype);

    // One extra slot beyond max_concurrent serves as the MTP spec-decode verify
    // scratch (spec_temp_slot()): the draft token's recurrent state is computed
    // there so the request's real slot keeps the post-last_token state and a
    // reject needs no rollback/redo. acquire_slot never hands this slot out.
    const size_t alloc_slots = static_cast<size_t>(cfg.max_concurrent) + 1;

    // SSM state: [slots, layers, num_v_heads, value_head_dim, d_state]
    ssm_buffer_
        = Tensor::create({alloc_slots, static_cast<size_t>(cfg.num_linear_layers), static_cast<size_t>(cfg.num_v_heads),
                          static_cast<size_t>(cfg.value_head_dim), static_cast<size_t>(cfg.d_state)},
                         cfg.state_dtype, exec.device_type, exec.device_id);

    // Conv state: [slots, layers, kernel - 1, qkv_dim]. The "kernel - 1" rows
    // hold the rolling causal-conv1d history; the current step is consumed
    // straight from the in-flight QKV projection, not the buffer.
    conv_buffer_ = Tensor::create({alloc_slots, static_cast<size_t>(cfg.num_linear_layers),
                                   static_cast<size_t>(cfg.conv_kernel_dim - 1), static_cast<size_t>(cfg.qkv_dim)},
                                  cfg.state_dtype, exec.device_type, exec.device_id);

    // Pre-compute per-slot and per-layer byte counts once. reset_slot is on the
    // request-start hot path; recomputing this on every call would multiply by
    // num_linear_layers * num_v_heads * ... for no reason.
    const size_t ssm_layer_bytes = static_cast<size_t>(cfg.num_v_heads) * static_cast<size_t>(cfg.value_head_dim)
                                 * static_cast<size_t>(cfg.d_state) * elem_bytes;
    const size_t conv_layer_bytes
        = static_cast<size_t>(cfg.conv_kernel_dim - 1) * static_cast<size_t>(cfg.qkv_dim) * elem_bytes;

    ssm_bytes_per_slot_ = static_cast<size_t>(cfg.num_linear_layers) * ssm_layer_bytes;
    conv_bytes_per_slot_ = static_cast<size_t>(cfg.num_linear_layers) * conv_layer_bytes;

    view_.ssm_base = ssm_buffer_->data();
    view_.conv_base = conv_buffer_->data();
    view_.ssm_stride_slot = static_cast<int64_t>(ssm_bytes_per_slot_);
    view_.ssm_stride_layer = static_cast<int64_t>(ssm_layer_bytes);
    view_.conv_stride_slot = static_cast<int64_t>(conv_bytes_per_slot_);
    view_.conv_stride_layer = static_cast<int64_t>(conv_layer_bytes);
    view_.num_v_heads = cfg.num_v_heads;
    view_.value_head_dim = cfg.value_head_dim;
    view_.d_state = cfg.d_state;
    view_.conv_kernel_dim = cfg.conv_kernel_dim;
    view_.qkv_dim = cfg.qkv_dim;
    view_.dtype = cfg.state_dtype;

    LOGI.printf("[SSMStatePool] max_concurrent=%d linear_layers=%d ssm_bytes/slot=%zu conv_bytes/slot=%zu total=%zu",
                cfg.max_concurrent, cfg.num_linear_layers, ssm_bytes_per_slot_, conv_bytes_per_slot_,
                static_cast<size_t>(cfg.max_concurrent) * bytes_per_slot());
}

int SSMStatePool::acquire_slot() {
    const int n = cfg_.max_concurrent;
    for (int i = 0; i < n; ++i) {
        const int idx = (next_hint_ + i) % n;
        if (!slot_in_use_[static_cast<size_t>(idx)]) {
            slot_in_use_[static_cast<size_t>(idx)] = 1;
            next_hint_ = (idx + 1) % n;
            return idx;
        }
    }
    throw std::runtime_error("SSMStatePool: no free slots (max_concurrent=" + std::to_string(n) + ")");
}

void SSMStatePool::release_slot(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= cfg_.max_concurrent) {
        return;
    }
    slot_in_use_[static_cast<size_t>(slot_idx)] = 0;
}

void SSMStatePool::reset_slot(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= cfg_.max_concurrent) {
        return;
    }
    // The two buffers always live on the same device (both ctor'd with the
    // ExecutorConfig pair) so a single device probe is enough.
    const zedinferDeviceType_t device = ssm_buffer_->deviceType();

    std::byte* ssm_byte_base = ssm_buffer_->data();
    std::byte* conv_byte_base = conv_buffer_->data();
    zero_bytes(device, ssm_byte_base + static_cast<size_t>(slot_idx) * ssm_bytes_per_slot_, ssm_bytes_per_slot_);
    zero_bytes(device, conv_byte_base + static_cast<size_t>(slot_idx) * conv_bytes_per_slot_, conv_bytes_per_slot_);
}

SSMStateSnapshot SSMStatePool::snapshot_slot(int slot_idx) const {
    if (slot_idx < 0 || slot_idx >= cfg_.max_concurrent) {
        throw std::runtime_error("SSMStatePool::snapshot_slot: slot_idx out of range " + std::to_string(slot_idx));
    }
    SSMStateSnapshot snap;
    snap.bytes.resize(ssm_bytes_per_slot_ + conv_bytes_per_slot_);

    const zedinferDeviceType_t device = ssm_buffer_->deviceType();
    const std::byte* ssm_src
        = static_cast<const std::byte*>(ssm_buffer_->data()) + static_cast<size_t>(slot_idx) * ssm_bytes_per_slot_;
    const std::byte* conv_src
        = static_cast<const std::byte*>(conv_buffer_->data()) + static_cast<size_t>(slot_idx) * conv_bytes_per_slot_;
    std::byte* ssm_dst = snap.bytes.data();
    std::byte* conv_dst = snap.bytes.data() + ssm_bytes_per_slot_;

    if (device == ZEDINFER_DEVICE_CPU) {
        std::memcpy(ssm_dst, ssm_src, ssm_bytes_per_slot_);
        std::memcpy(conv_dst, conv_src, conv_bytes_per_slot_);
    } else {
        auto* api = core::context().runtime().api();
        api->memcpy_sync(ssm_dst, ssm_src, ssm_bytes_per_slot_, ZEDINFER_MEMCPY_D2H);
        api->memcpy_sync(conv_dst, conv_src, conv_bytes_per_slot_, ZEDINFER_MEMCPY_D2H);
    }
    return snap;
}

void SSMStatePool::restore_slot(int slot_idx, const SSMStateSnapshot& snap) {
    if (slot_idx < 0 || slot_idx >= cfg_.max_concurrent) {
        throw std::runtime_error("SSMStatePool::restore_slot: slot_idx out of range " + std::to_string(slot_idx));
    }
    if (snap.bytes.size() != ssm_bytes_per_slot_ + conv_bytes_per_slot_) {
        throw std::runtime_error("SSMStatePool::restore_slot: snapshot size " + std::to_string(snap.bytes.size())
                                 + " != expected " + std::to_string(ssm_bytes_per_slot_ + conv_bytes_per_slot_));
    }

    const zedinferDeviceType_t device = ssm_buffer_->deviceType();
    std::byte* ssm_dst
        = static_cast<std::byte*>(ssm_buffer_->data()) + static_cast<size_t>(slot_idx) * ssm_bytes_per_slot_;
    std::byte* conv_dst
        = static_cast<std::byte*>(conv_buffer_->data()) + static_cast<size_t>(slot_idx) * conv_bytes_per_slot_;
    const std::byte* ssm_src = snap.bytes.data();
    const std::byte* conv_src = snap.bytes.data() + ssm_bytes_per_slot_;

    if (device == ZEDINFER_DEVICE_CPU) {
        std::memcpy(ssm_dst, ssm_src, ssm_bytes_per_slot_);
        std::memcpy(conv_dst, conv_src, conv_bytes_per_slot_);
    } else {
        auto* api = core::context().runtime().api();
        api->memcpy_sync(ssm_dst, ssm_src, ssm_bytes_per_slot_, ZEDINFER_MEMCPY_H2D);
        api->memcpy_sync(conv_dst, conv_src, conv_bytes_per_slot_, ZEDINFER_MEMCPY_H2D);
    }
}

void SSMStatePool::copy_layer_state(int dst_slot, int src_slot, int layer_idx) {
    if (dst_slot < 0 || dst_slot > cfg_.max_concurrent || src_slot < 0 || src_slot > cfg_.max_concurrent
        || layer_idx < 0 || layer_idx >= cfg_.num_linear_layers) {
        throw std::runtime_error("SSMStatePool::copy_layer_state: out-of-range slot/layer");
    }
    const size_t ssm_layer = static_cast<size_t>(view_.ssm_stride_layer);
    const size_t conv_layer = static_cast<size_t>(view_.conv_stride_layer);
    auto* ssm = static_cast<std::byte*>(ssm_buffer_->data());
    auto* conv = static_cast<std::byte*>(conv_buffer_->data());
    std::byte* ssm_dst
        = ssm + static_cast<size_t>(dst_slot) * ssm_bytes_per_slot_ + static_cast<size_t>(layer_idx) * ssm_layer;
    std::byte* ssm_src
        = ssm + static_cast<size_t>(src_slot) * ssm_bytes_per_slot_ + static_cast<size_t>(layer_idx) * ssm_layer;
    std::byte* conv_dst
        = conv + static_cast<size_t>(dst_slot) * conv_bytes_per_slot_ + static_cast<size_t>(layer_idx) * conv_layer;
    std::byte* conv_src
        = conv + static_cast<size_t>(src_slot) * conv_bytes_per_slot_ + static_cast<size_t>(layer_idx) * conv_layer;

    if (ssm_buffer_->deviceType() == ZEDINFER_DEVICE_CPU) {
        std::memcpy(ssm_dst, ssm_src, ssm_layer);
        std::memcpy(conv_dst, conv_src, conv_layer);
    } else {
        auto* api = core::context().runtime().api();
        auto stream = core::context().runtime().stream();
        api->memcpy_async(ssm_dst, ssm_src, ssm_layer, ZEDINFER_MEMCPY_D2D, stream);
        api->memcpy_async(conv_dst, conv_src, conv_layer, ZEDINFER_MEMCPY_D2D, stream);
    }
}

void SSMStatePool::copy_slot_state(int dst_slot, int src_slot) {
    if (dst_slot < 0 || dst_slot > cfg_.max_concurrent || src_slot < 0 || src_slot > cfg_.max_concurrent) {
        throw std::runtime_error("SSMStatePool::copy_slot_state: out-of-range slot");
    }
    auto* ssm = static_cast<std::byte*>(ssm_buffer_->data());
    auto* conv = static_cast<std::byte*>(conv_buffer_->data());
    std::byte* ssm_dst = ssm + static_cast<size_t>(dst_slot) * ssm_bytes_per_slot_;
    std::byte* ssm_src = ssm + static_cast<size_t>(src_slot) * ssm_bytes_per_slot_;
    std::byte* conv_dst = conv + static_cast<size_t>(dst_slot) * conv_bytes_per_slot_;
    std::byte* conv_src = conv + static_cast<size_t>(src_slot) * conv_bytes_per_slot_;

    if (ssm_buffer_->deviceType() == ZEDINFER_DEVICE_CPU) {
        std::memcpy(ssm_dst, ssm_src, ssm_bytes_per_slot_);
        std::memcpy(conv_dst, conv_src, conv_bytes_per_slot_);
    } else {
        auto* api = core::context().runtime().api();
        auto stream = core::context().runtime().stream();
        api->memcpy_async(ssm_dst, ssm_src, ssm_bytes_per_slot_, ZEDINFER_MEMCPY_D2D, stream);
        api->memcpy_async(conv_dst, conv_src, conv_bytes_per_slot_, ZEDINFER_MEMCPY_D2D, stream);
    }
}

int SSMStatePool::num_free_slots() const {
    int free_count = 0;
    for (char in_use : slot_in_use_) {
        if (!in_use) {
            ++free_count;
        }
    }
    return free_count;
}

} // namespace zedinfer::model
