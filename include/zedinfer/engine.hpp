#pragma once

#include "backend/device/device.hpp"
#include "backend/kvcache/base.hpp"
#include "frontend/models/base.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/base.hpp"
#include "zedinfer/executor.hpp"
#include "zedinfer/generation_types.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace zedinfer {

// Forward declaration
class InferenceSession;

/**
 * Stateless inference engine shared across sessions
 * Thread-safe for concurrent session access
 */
class InferenceEngine : public std::enable_shared_from_this<InferenceEngine> {
public:
    /**
     * Factory method to create engine from model directory
     * @param model_path Directory containing config.json, weights, tokenizer
     * @param device Target device (CPU/CUDA)
     * @return Shared pointer to engine instance
     */
    static std::shared_ptr<InferenceEngine> create(
        const std::string &model_path,
        device::Device device);

    /**
     * Create new inference session with independent KV cache
     * @param gen_config Generation configuration
     * @return Unique session instance
     */
    std::unique_ptr<InferenceSession> create_session(const GenerationConfig &gen_config);

    /**
     * Generate text from prompt (low-level API)
     * @param kvcache Session-specific KV cache
     * @param prompt Input text
     * @param config Generation parameters
     * @return Generated text
     */
    std::string generate(
        kvcache::KVCache &kvcache,
        const std::string &prompt,
        const GenerationConfig &config);

    /**
     * Generate token IDs from input IDs (lowest-level API)
     * @param kvcache Session-specific KV cache
     * @param input_ids Tokenized input sequence
     * @param config Generation parameters
     * @return Generated token IDs
     */
    std::vector<int> generate_tokens(
        kvcache::KVCache &kvcache,
        const std::vector<int> &input_ids,
        const GenerationConfig &config);

    /**
     * Get statistics from last generation
     */
    const GenerationStats &last_stats() const { return last_stats_; }

private:
    // Private constructor - use create() factory method
    InferenceEngine(
        std::shared_ptr<model::Model> model,
        graph::compute_graph_t graph,
        std::shared_ptr<GraphExecutor> executor,
        std::shared_ptr<tokenizer::Tokenizer> tokenizer,
        std::shared_ptr<sampler::Sampler> sampler,
        device::Device device);

    // Core components (immutable, shared across sessions)
    std::shared_ptr<model::Model> model_;
    graph::compute_graph_t graph_;
    std::shared_ptr<GraphExecutor> executor_;
    std::shared_ptr<tokenizer::Tokenizer> tokenizer_;
    std::shared_ptr<sampler::Sampler> sampler_;
    device::Device device_;

    // Runtime state (not thread-safe, protect if needed)
    GenerationStats last_stats_;

    // Internal helpers
    bool should_stop(int token_id) const;
    void update_stats_prefill(double time_ms, int num_tokens);
    void update_stats_decode(double time_ms);

public:
    /**
     * Warm up engine by running dummy inference
     * Should be called after create() to initialize execution paths
     * @param prefill_len Prefill sequence length for warmup
     * @param decode_steps Number of decode steps to warmup
     */
    void warmup(size_t prefill_len = 128, size_t decode_steps = 128);

    /**
     * Profile engine performance with specified sequence parameters.
     * @param prefill_len Prefill sequence length for profiling
     * @param decode_steps Number of decode steps to profile
     * @return std::pair<double, double> Performance metrics (e.g., prefill latency, decode latency)
     */
    std::pair<double, double> profile(size_t prefill_len = 128, size_t decode_steps = 128);
};

} // namespace zedinfer