#pragma once

#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/base.hpp"
#include "graph/executor.hpp"
#include "kvcache/dynamic.hpp"
#include "models/base.hpp"
#include <functional>

#include <memory>
#include <string>
#include <vector>

namespace neollm {

/**
 * Generation configuration parameters
 */
struct GenerationConfig {
    int max_new_tokens = 512;
    int max_seq_len = 2048;

    // Sampling parameters
    sampler::SamplerParams sampler_params;
    sampler::SamplerType sampler_type = sampler::SamplerType::ARGMAX; // "argmax" or "general"

    // Special token handling
    bool skip_special_tokens = true;
    std::vector<int> stop_token_ids; // Additional stop tokens

    // Streaming
    bool stream = false;
    std::function<void(const std::string &)> stream_callback;

    // Verbose output
    bool verbose = false;
    bool print_stats = false;

    void validate() const;
    std::string info() const;
};

/**
 * Generation statistics
 */
struct GenerationStats {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int total_tokens = 0;

    double prefill_time_ms = 0.0;
    double decode_time_ms = 0.0;
    double total_time_ms = 0.0;

    double tokens_per_second() const {
        return total_time_ms > 0 ? (generated_tokens * 1000.0 / total_time_ms) : 0.0;
    }

    double prefill_tokens_per_second() const {
        return prefill_time_ms > 0 ? (prompt_tokens * 1000.0 / prefill_time_ms) : 0.0;
    }

    std::string summary() const;
};

/**
 * Main inference engine integrating all components
 */
class InferenceEngine {
public:
    /**
     * Create inference engine from model directory
     * @param model_path Path to model directory (containing config.json, weights, tokenizer)
     * @param device_type Device type (CPU/CUDA)
     * @param device_id Device ID (default: 0)
     * @param dtype Data type for computation (default: BF16)
     */
    static std::shared_ptr<InferenceEngine> create(
        const std::string &model_path,
        NeollmDeviceType_t device_type = NEOLLM_DEVICE_CPU,
        int device_id = 0,
        NeollmDataType_t dtype = NEOLLM_DTYPE_BF16);

    /**
     * Chat completion with message history
     * @param messages Chat history [(role, content), ...]
     * @param config Generation configuration
     * @return Assistant response
     */
    std::string chat(
        const std::vector<std::pair<std::string, std::string>> &messages,
        const GenerationConfig &config = GenerationConfig());

    /**
     * Generate text from prompt
     * @param prompt Input text prompt
     * @param config Generation configuration
     * @return Generated text
     */
    std::string generate(
        const std::string &prompt,
        const GenerationConfig &config = GenerationConfig());

    /**
     * Generate tokens from token IDs (low-level API)
     * @param input_ids Input token sequence
     * @param config Generation configuration
     * @return Generated token IDs
     */
    std::vector<int> generate_tokens(
        const std::vector<int> &input_ids,
        const GenerationConfig &config = GenerationConfig());

    /**
     * Reset KV cache and internal state
     */
    void reset();

    /**
     * Get last generation statistics
     */
    const GenerationStats &last_stats() const { return last_stats_; }

    /**
     * Print engine information
     */
    void print_info() const;

    /**
     * Access components
     */
    const model::Model *model() const { return model_.get(); }
    tokenizer::Tokenizer *tokenizer() const { return tokenizer_.get(); }
    kvcache::kvcache_t kv_cache() const { return kv_cache_; }

private:
    // Friend declaration: allow Builder to access private constructor
    friend class InferenceEngineBuilder;
    InferenceEngine(
        std::unique_ptr<model::Model> model,
        std::unique_ptr<tokenizer::Tokenizer> tokenizer,
        graph::compute_graph_t graph,
        kvcache::kvcache_t kv_cache,
        std::unique_ptr<graph::GraphExecutor> executor,
        const GenerationConfig &default_config);

    // Core components
    std::unique_ptr<model::Model> model_;
    std::unique_ptr<tokenizer::Tokenizer> tokenizer_;
    graph::compute_graph_t graph_;
    kvcache::kvcache_t kv_cache_;
    std::unique_ptr<graph::GraphExecutor> executor_;

    // Configuration
    GenerationConfig default_config_;
    NeollmDeviceType_t device_type_;
    int device_id_;
    NeollmDataType_t dtype_;

    // Runtime state
    GenerationStats last_stats_;

    // Helper methods
    bool should_stop(int token_id, const GenerationConfig &config) const;
    void update_stats_prefill(double time_ms, int num_tokens);
    void update_stats_decode(double time_ms);
    std::shared_ptr<sampler::Sampler> create_sampler(const GenerationConfig &config) const;
};

/**
 * Builder pattern for easier engine construction
 */
class InferenceEngineBuilder {
public:
    InferenceEngineBuilder &set_model_path(const std::string &path) {
        model_path_ = path;
        return *this;
    }

    InferenceEngineBuilder &set_device(NeollmDeviceType_t device_type, int device_id = 0) {
        device_type_ = device_type;
        device_id_ = device_id;
        return *this;
    }

    InferenceEngineBuilder &set_dtype(NeollmDataType_t dtype) {
        dtype_ = dtype;
        return *this;
    }

    InferenceEngineBuilder &set_kv_cache_config(const kvcache::DynamicKVCacheConfig &config) {
        kv_cache_config_ = config;
        use_custom_kv_config_ = true;
        return *this;
    }

    InferenceEngineBuilder &set_generation_config(const GenerationConfig &config) {
        gen_config_ = config;
        return *this;
    }

    std::unique_ptr<InferenceEngine> build();

private:
    std::string model_path_;
    NeollmDeviceType_t device_type_ = NEOLLM_DEVICE_CPU;
    int device_id_ = 0;
    NeollmDataType_t dtype_ = NEOLLM_DTYPE_BF16;

    kvcache::DynamicKVCacheConfig kv_cache_config_;
    bool use_custom_kv_config_ = false;

    GenerationConfig gen_config_;

    // Friend class to access InferenceEngine's private constructor
    friend class InferenceEngine;
};

} // namespace neollm