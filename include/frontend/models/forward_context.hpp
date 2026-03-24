#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer/activation.hpp"

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

} // namespace zedinfer::model
