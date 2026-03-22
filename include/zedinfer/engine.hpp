#pragma once

#include "backend/device/device.hpp"
#include "backend/kvcache/base.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "frontend/models/base.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/base.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/generation_types.hpp"
#include "zedinfer/chat_template.hpp"
#include "zedinfer/batch_context.hpp"
#include "zedinfer/request.hpp"
#include "zedinfer/scheduler.hpp"

#include <cstddef>
#include <future>
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

    GenerationResult generate_tokens(
        kvcache::KVCache &kvcache,
        const std::vector<int> &input_ids,
        const GenerationConfig &config);

    void warmup(size_t prefill_len = 128, size_t decode_steps = 128);
    std::pair<double, double> profile(size_t prefill_len = 128, size_t decode_steps = 128);

    /**
     * Submit a request to the scheduler for batched processing.
     * Returns a future that will be fulfilled when generation completes.
     */
    std::future<GenerationResult> submit_async(std::unique_ptr<InferenceRequest> request);

    /**
     * Run one iteration of the batched engine loop.
     * Returns true if work was done, false if idle.
     */
    bool step();

    /**
     * Run the batched engine loop until all work is done.
     * For serving mode (PR-10), this runs on a dedicated thread.
     */
    void run_loop();

private:
    InferenceEngine(
        std::shared_ptr<model::Model> model,
        std::shared_ptr<tokenizer::Tokenizer> tokenizer,
        std::shared_ptr<sampler::Sampler> sampler,
        device::Device device,
        ExecutorConfig exec_config,
        ChatTemplate chat_template);

    std::shared_ptr<model::Model> model_;
    std::shared_ptr<tokenizer::Tokenizer> tokenizer_;
    std::shared_ptr<sampler::Sampler> sampler_;
    device::Device device_;
    ExecutorConfig exec_config_;
    ChatTemplate chat_template_;
    std::vector<int> stop_token_ids_; // All token IDs that end generation
    Scheduler scheduler_;
    SchedulerConfig scheduler_config_;
    std::unique_ptr<kvcache::BlockPool> block_pool_;
    std::unique_ptr<kvcache::BlockAllocator> block_allocator_;

    void init_block_pool();
    void build_stop_token_ids();
    bool should_stop(int token_id) const;
    std::unique_ptr<InferenceRequest> build_request(
        const std::vector<int> &input_ids,
        const GenerationConfig &config);
};

} // namespace zedinfer
