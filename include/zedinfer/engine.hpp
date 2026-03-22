#pragma once

#include "backend/device/device.hpp"
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

class InferenceSession;

class InferenceEngine : public std::enable_shared_from_this<InferenceEngine> {
public:
    static std::shared_ptr<InferenceEngine> create(
        const std::string &model_path,
        device::Device device);

    std::unique_ptr<InferenceSession> create_session(const GenerationConfig &gen_config);

    // Session-based generate (uses block table directly)
    std::string generate(
        kvcache::SequenceBlockTable &block_table,
        const std::string &prompt,
        const GenerationConfig &config);

    GenerationResult generate_tokens(
        kvcache::SequenceBlockTable &block_table,
        const std::vector<int> &input_ids,
        const GenerationConfig &config);

    // Batch mode
    std::future<GenerationResult> submit_async(std::unique_ptr<InferenceRequest> request);
    bool step();
    void run_loop();

    // Profiling (uses DynamicKVCache internally, no block table)
    void warmup(size_t prefill_len = 128, size_t decode_steps = 128);
    std::pair<double, double> profile(size_t prefill_len = 128, size_t decode_steps = 128);

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
    std::vector<int> stop_token_ids_;
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
