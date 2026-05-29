#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"

#include <cstddef>
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
    int num_linear_layers = 0;
    int num_v_heads = 0;
    int value_head_dim = 0;
    int d_state = 0;
    int conv_kernel_dim = 4;
    int qkv_dim = 0;
    int max_concurrent = 1;
    zedinferDataType_t state_dtype = ZEDINFER_DTYPE_BF16;
};

// Lightweight view into an SSMStatePool's underlying buffers. The pool's ctor
// fills this once and `view()` returns a copy — callers (M1+ kernel sites) use
// the raw bases + per-slot/per-layer byte strides to address into the buffers
// without taking a dependency on tensor_t / TensorMeta. Stride fields are bytes.
struct SSMStateView {
    void* ssm_base = nullptr;
    void* conv_base = nullptr;
    int64_t ssm_stride_slot = 0;
    int64_t ssm_stride_layer = 0;
    int64_t conv_stride_slot = 0;
    int64_t conv_stride_layer = 0;
    int num_v_heads = 0;
    int value_head_dim = 0;
    int d_state = 0;
    int conv_kernel_dim = 0;
    int qkv_dim = 0;
    zedinferDataType_t dtype = ZEDINFER_DTYPE_BF16;
};

// Frozen byte-level copy of one slot's combined SSM + conv state. Used by the
// SSMSnapshotCache to persist a prefilled prefix's linear-attention state so a
// future request with the same prompt can restore it instead of paying the
// prefill cost (and, more importantly, so prefix-cache reuse of the KV blocks
// produces output identical to a cold-cache run). Snapshot bytes live in
// pinned host memory; D2H copy on snapshot, H2D copy on restore.
struct SSMStateSnapshot {
    std::vector<std::byte> bytes; // length == SSMStatePool::snapshot_bytes()
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

    SSMStatePool(const SSMStatePool&) = delete;
    SSMStatePool& operator=(const SSMStatePool&) = delete;
    SSMStatePool(SSMStatePool&&) = delete;
    SSMStatePool& operator=(SSMStatePool&&) = delete;

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

    // Capture the slot's full SSM + conv state into a host-resident snapshot.
    // Throws on out-of-range slot_idx. Synchronous D2H copy on NVIDIA so
    // callers can safely move-store the result; the runtime stream is drained
    // through memcpy_sync.
    SSMStateSnapshot snapshot_slot(int slot_idx) const;

    // Inverse of snapshot_slot. Restores the snapshot bytes into the slot's
    // state buffers (H2D on NVIDIA, memcpy on CPU). Snapshot size must equal
    // snapshot_bytes(); throws on mismatch or out-of-range slot_idx.
    void restore_slot(int slot_idx, const SSMStateSnapshot& snapshot);

    // Total bytes per snapshot (ssm + conv slot bytes). Stays constant for the
    // lifetime of the pool.
    size_t snapshot_bytes() const { return ssm_bytes_per_slot_ + conv_bytes_per_slot_; }

    // Index of a dedicated scratch slot, never handed out by acquire_slot, used
    // by Qwen3.5 MTP spec-decode verify. The 2-token [last_token, draft] verify
    // commits token 0 into the request's real slot (-> post-last_token state)
    // and token 1 into this temp slot (-> post-draft state); on accept the
    // scheduler promotes temp->real via copy_slot_state, on reject the temp is
    // simply discarded (real already holds the correct post-last_token state),
    // so reject costs no extra forward. The pool over-allocates one slot for it.
    int spec_temp_slot() const { return cfg_.max_concurrent; }

    // Device-to-device copy of one layer's combined SSM + conv state from
    // src_slot to dst_slot. Stream-ordered on the compute stream so it composes
    // with the gdn/conv kernels that read/write the same slots. Used to seed the
    // spec temp slot with the post-last_token state before the draft is applied.
    void copy_layer_state(int dst_slot, int src_slot, int layer_idx);

    // Device-to-device copy of a whole slot (all layers, SSM + conv). Used to
    // promote the spec temp slot into the real slot on a spec-decode accept.
    void copy_slot_state(int dst_slot, int src_slot);

    // Returns a snapshot of the layout. The fields stay valid for the lifetime
    // of the pool; the snapshot itself is a value copy so callers cannot
    // accidentally mutate pool state.
    SSMStateView view() const { return view_; }

    int num_free_slots() const;
    size_t bytes_per_slot() const { return ssm_bytes_per_slot_ + conv_bytes_per_slot_; }
    const SSMStatePoolConfig& config() const { return cfg_; }

private:
    SSMStatePoolConfig cfg_;
    tensor_t ssm_buffer_;
    tensor_t conv_buffer_;
    // char (not bool) so we can iterate without vector<bool> bit-packing
    // surprises and so we can later swap for an atomic flag if we need
    // thread-safe acquire/release.
    std::vector<char> slot_in_use_;
    int next_hint_ = 0;
    size_t ssm_bytes_per_slot_ = 0;
    size_t conv_bytes_per_slot_ = 0;
    SSMStateView view_{};
};

} // namespace zedinfer::model
