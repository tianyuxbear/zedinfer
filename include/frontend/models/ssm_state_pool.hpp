#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"

#include <cstdint>
#include <vector>

namespace zedinfer {
struct ExecutorConfig;
} // namespace zedinfer

namespace zedinfer::model {

// Static config for an SSMStatePool instance. All dims come from the parsed
// Qwen3.5 linear-attention sub-config (LinearAttnConfig); max_concurrent is the
// number of concurrent requests the engine guarantees to admit.
struct SSMStatePoolConfig {
    int                num_linear_layers = 0;
    int                num_v_heads       = 0;
    int                value_head_dim    = 0;
    int                d_state           = 0;
    int                conv_kernel_dim   = 4;
    int                qkv_dim           = 0;
    int                max_concurrent    = 1;
    zedinferDataType_t state_dtype       = ZEDINFER_DTYPE_BF16;
};

// Lightweight view into an SSMStatePool's underlying buffers. The pool's ctor
// fills this once and `view()` returns a copy — callers (M1+ kernel sites) use
// the raw bases + per-slot/per-layer byte strides to address into the buffers
// without taking a dependency on tensor_t / TensorMeta. Stride fields are bytes.
struct SSMStateView {
    void*              ssm_base          = nullptr;
    void*              conv_base         = nullptr;
    int64_t            ssm_stride_slot   = 0;
    int64_t            ssm_stride_layer  = 0;
    int64_t            conv_stride_slot  = 0;
    int64_t            conv_stride_layer = 0;
    int                num_v_heads       = 0;
    int                value_head_dim    = 0;
    int                d_state           = 0;
    int                conv_kernel_dim   = 0;
    int                qkv_dim           = 0;
    zedinferDataType_t dtype             = ZEDINFER_DTYPE_BF16;
};

// Engine-level slot pool for Mamba2 SSM state + conv state used by Qwen3.5
// linear-attention layers. The pool owns two contiguous tensors:
//   ssm_buffer_  shape: [max_concurrent, num_linear_layers, num_v_heads, value_head_dim, d_state]
//   conv_buffer_ shape: [max_concurrent, num_linear_layers, conv_kernel_dim - 1, qkv_dim]
//
// Slot acquire/release is a linear scan over a `slot_in_use_` bitmap with a
// round-robin hint to keep amortized cost O(1). `reset_slot` zeroes that slot's
// span in both buffers via the existing fill_zero dispatcher — no kernels are
// launched from this class (FlashInfer SSU kernel wiring lands in M1 of P1).
class SSMStatePool {
public:
    SSMStatePool(const SSMStatePoolConfig& cfg, const ExecutorConfig& exec);
    ~SSMStatePool() = default;

    SSMStatePool(const SSMStatePool&)            = delete;
    SSMStatePool& operator=(const SSMStatePool&) = delete;
    SSMStatePool(SSMStatePool&&)                 = delete;
    SSMStatePool& operator=(SSMStatePool&&)      = delete;

    // Returns the index of a free slot and marks it in-use. Throws
    // std::runtime_error when every slot is currently held.
    int acquire_slot();

    // Marks a previously-acquired slot free. Out-of-range indices are silently
    // ignored so callers can release defensively in error paths.
    void release_slot(int slot_idx);

    // Zeroes the SSM and conv state spans for `slot_idx` across all linear
    // layers. Out-of-range indices are silently ignored. Dispatches by the
    // backing tensor's device type — no kernels are written here.
    void reset_slot(int slot_idx);

    // Returns a snapshot of the layout. The fields stay valid for the lifetime
    // of the pool; the snapshot itself is a value copy so callers cannot
    // accidentally mutate pool state.
    SSMStateView view() const { return view_; }

    int                       num_free_slots() const;
    size_t                    bytes_per_slot() const { return ssm_bytes_per_slot_ + conv_bytes_per_slot_; }
    const SSMStatePoolConfig& config() const { return cfg_; }

private:
    SSMStatePoolConfig cfg_;
    tensor_t           ssm_buffer_;
    tensor_t           conv_buffer_;
    // char (not bool) so we can iterate without vector<bool> bit-packing
    // surprises and so we can later swap for an atomic flag if we need
    // thread-safe acquire/release.
    std::vector<char>  slot_in_use_;
    int                next_hint_           = 0;
    size_t             ssm_bytes_per_slot_  = 0;
    size_t             conv_bytes_per_slot_ = 0;
    SSMStateView       view_{};
};

} // namespace zedinfer::model
