#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer/activation.hpp"

#include <stdexcept>

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
    // Write input data into pre-allocated tensors (for decode scratch path)
    virtual void prepare_inputs_into(tensor_t /*ids*/, tensor_t /*pos_ids*/) {
        throw std::runtime_error("prepare_inputs_into not implemented");
    }
    virtual void write_kv(int layer, tensor_t k, tensor_t v) = 0;
    virtual tensor_t attend(int layer, tensor_t q_rope, float scale,
                            const ExecutorConfig &exec_config,
                            size_t nhead, size_t nkvhead, size_t head_dim,
                            tensor_t pre_alloc_out = nullptr) = 0;
    virtual void finalize() = 0;
};

} // namespace zedinfer::model
