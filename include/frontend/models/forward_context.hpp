#pragma once

#include "backend/kvcache/base.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "backend/tensor/tensor.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/batch_context.hpp"

#include <vector>

namespace zedinfer::model {

/**
 * Abstract execution context for the shared transformer forward loop.
 * Encapsulates the differences between single-request and batched execution:
 *   - Input preparation (token IDs, position IDs)
 *   - KV write (to KVCache or scatter to blocks)
 *   - Attention dispatch (single/batched, paged/contiguous)
 *   - Post-forward finalization
 */
class ForwardContext {
public:
    virtual ~ForwardContext() = default;

    virtual int num_tokens() const = 0;

    virtual void prepare_inputs(tensor_t &ids, tensor_t &pos_ids,
                                const ExecutorConfig &exec_config) = 0;

    // Write K and V for a layer. Called after projection + RoPE.
    // k: [N, nkvhead, head_dim], v: [N, kv_dim]
    virtual void write_kv(int layer, tensor_t k, tensor_t v) = 0;

    // Run attention for a layer. Returns [N, nhead, head_dim].
    virtual tensor_t attend(int layer, tensor_t q_rope, float scale,
                            const ExecutorConfig &exec_config,
                            size_t nhead, size_t nkvhead, size_t head_dim) = 0;

    // Post-forward (e.g., update KV cache length)
    virtual void finalize() = 0;
};

/**
 * Single-request forward context. Wraps KVCache interface.
 * Used by run_one() and session-based generation.
 */
class SingleForwardContext : public ForwardContext {
public:
    SingleForwardContext(const std::vector<int> &input_ids, int past_len,
                         kvcache::KVCache &kvcache);

    int num_tokens() const override { return sl_; }
    void prepare_inputs(tensor_t &ids, tensor_t &pos_ids,
                        const ExecutorConfig &exec_config) override;
    void write_kv(int layer, tensor_t k, tensor_t v) override;
    tensor_t attend(int layer, tensor_t q_rope, float scale,
                    const ExecutorConfig &exec_config,
                    size_t nhead, size_t nkvhead, size_t head_dim) override;
    void finalize() override;

private:
    const std::vector<int> &input_ids_;
    int past_len_;
    int sl_;
    kvcache::KVCache &kvcache_;
};

/**
 * Batched forward context. Wraps BatchContext + BlockPool.
 * Used by continuous batching engine loop.
 */
class BatchedForwardContext : public ForwardContext {
public:
    BatchedForwardContext(const BatchContext &batch,
                          kvcache::BlockAllocator &allocator,
                          const ExecutorConfig &exec_config);

    int num_tokens() const override { return total_; }
    void prepare_inputs(tensor_t &ids, tensor_t &pos_ids,
                        const ExecutorConfig &exec_config) override;
    void write_kv(int layer, tensor_t k, tensor_t v) override;
    tensor_t attend(int layer, tensor_t q_rope, float scale,
                    const ExecutorConfig &exec_config,
                    size_t nhead, size_t nkvhead, size_t head_dim) override;
    void finalize() override;

private:
    const BatchContext &batch_;
    kvcache::BlockAllocator &allocator_;
    int total_;
};

} // namespace zedinfer::model
