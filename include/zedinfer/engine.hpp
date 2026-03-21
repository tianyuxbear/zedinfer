#pragma once

#include "backend/device/device.hpp"
#include "backend/kvcache/base.hpp"
#include "frontend/models/base.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/base.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/generation_types.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace zedinfer {

// Forward declaration
class InferenceSession;

/**
 * Inference engine shared across sessions.
 * Owns model, tokenizer, sampler, and runtime config.
 */
class InferenceEngine : public std::enable_shared_from_this<InferenceEngine> {
public:
    static std::shared_ptr<InferenceEngine> create(
        const std::string &model_path,
        device::Device device);

    std::unique_ptr<InferenceSession> create_session(const GenerationConfig &gen_config);

    std::string generate(
        kvcache::KVCache &kvcache,
        const std::string &prompt,
        const GenerationConfig &config);

    std::vector<int> generate_tokens(
        kvcache::KVCache &kvcache,
        const std::vector<int> &input_ids,
        const GenerationConfig &config);

    const GenerationStats &last_stats() const { return last_stats_; }

    void warmup(size_t prefill_len = 128, size_t decode_steps = 128);
    std::pair<double, double> profile(size_t prefill_len = 128, size_t decode_steps = 128);

private:
    InferenceEngine(
        std::shared_ptr<model::Model> model,
        std::shared_ptr<tokenizer::Tokenizer> tokenizer,
        std::shared_ptr<sampler::Sampler> sampler,
        device::Device device,
        ExecutorConfig exec_config);

    std::shared_ptr<model::Model> model_;
    std::shared_ptr<tokenizer::Tokenizer> tokenizer_;
    std::shared_ptr<sampler::Sampler> sampler_;
    device::Device device_;
    ExecutorConfig exec_config_;
    GenerationStats last_stats_;

    bool should_stop(int token_id) const;
    void update_stats_prefill(double time_ms, int num_tokens);
    void update_stats_decode(double time_ms);
};

} // namespace zedinfer
