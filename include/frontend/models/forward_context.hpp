#pragma once

#include "backend/kvcache/dynamic.hpp"
#include "backend/tensor/tensor.hpp"
#include "zedinfer/activation.hpp"

#include <vector>

namespace zedinfer::model {

/**
 * Abstract execution context for the shared transformer forward loop.
 * Subclasses handle the differences in KV write and attention dispatch.
 */
class ForwardContext {
public:
    virtual ~ForwardContext() = default;

    virtual int num_tokens() const = 0;
    virtual void prepare_inputs(tensor_t &ids, tensor_t &pos_ids,
                                const ExecutorConfig &exec_config) = 0;
    virtual void write_kv(int layer, tensor_t k, tensor_t v) = 0;
    virtual tensor_t attend(int layer, tensor_t q_rope, float scale,
                            const ExecutorConfig &exec_config,
                            size_t nhead, size_t nkvhead, size_t head_dim) = 0;
    virtual void finalize() = 0;
};

/**
 * Contiguous forward context using DynamicKVCache.
 * Used only by warmup() and profile() — not for serving.
 */
class ContiguousForwardContext : public ForwardContext {
public:
    ContiguousForwardContext(const std::vector<int> &input_ids, int past_len,
                             kvcache::DynamicKVCache &kvcache);

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
    kvcache::DynamicKVCache &kvcache_;
};

} // namespace zedinfer::model
