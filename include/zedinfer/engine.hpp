#pragma once

#include "backend/device/device.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "backend/kvcache/prefix_cache.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "frontend/models/base.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/base.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/generation_types.hpp"
#include "zedinfer/chat_template.hpp"
#include "zedinfer/request.hpp"
#include "zedinfer/serving_loop.hpp"
#include "zedinfer/profiler.hpp"

#include <cstddef>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace zedinfer {

class InferenceSession;

/**
 * Inference engine: resource container and factory.
 * Owns model, tokenizer, sampler, block pool.
 * Delegates serving to ServingLoop and profiling to Profiler.
 */
class InferenceEngine : public std::enable_shared_from_this<InferenceEngine> {
public:
    static std::shared_ptr<InferenceEngine> create(
        const std::string &model_path, device::Device device,
        SchedulerConfig sched_config = {});

    std::unique_ptr<InferenceSession> create_session(const GenerationConfig &gen_config);

    // Resource accessors (used by ServingLoop, Profiler, Session)
    model::Model &model() { return *model_; }
    tokenizer::Tokenizer &tokenizer() { return *tokenizer_; }
    sampler::Sampler &sampler() { return *sampler_; }
    const ExecutorConfig &exec_config() const { return exec_config_; }
    const ChatTemplate &chat_template() const { return chat_template_; }
    const std::vector<int> &stop_token_ids() const { return stop_token_ids_; }
    kvcache::BlockPool *block_pool() { return block_pool_.get(); }
    kvcache::BlockAllocator *block_allocator() { return block_allocator_.get(); }
    kvcache::PrefixCache *prefix_cache() { return prefix_cache_.get(); }
    model::DecodeScratch *decode_scratch() { return decode_scratch_.get(); }
    const std::string &model_name() const { return model_name_; }
    int pending_count() const { return serving_loop_ ? serving_loop_->pending_count() : 0; }
    int active_count() const { return serving_loop_ ? serving_loop_->active_count() : 0; }

    // Delegation to ServingLoop (backward compat for existing callers)
    std::string generate(kvcache::SequenceBlockTable &bt, const std::string &p, const GenerationConfig &c) {
        return serving_loop_->generate(bt, p, c);
    }
    GenerationResult generate_tokens(kvcache::SequenceBlockTable &bt, const std::vector<int> &ids, const GenerationConfig &c) {
        return serving_loop_->generate_tokens(bt, ids, c);
    }
    std::future<GenerationResult> submit_async(std::unique_ptr<InferenceRequest> r) {
        return serving_loop_->submit_async(std::move(r));
    }
    bool step() { return serving_loop_->step(); }
    void run_loop() { serving_loop_->run_loop(); }
    void run_serving() { serving_loop_->run_serving(); }
    void stop_serving() { serving_loop_->stop(); }

    // Delegation to Profiler
    void warmup(size_t pl = 128, size_t ds = 128) { profiler_->warmup(pl, ds); }
    std::pair<double, double> profile(size_t pl = 128, size_t ds = 128) { return profiler_->profile(pl, ds); }

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
    SchedulerConfig scheduler_config_;
    std::string model_name_;
    std::unique_ptr<kvcache::BlockPool> block_pool_;
    std::unique_ptr<kvcache::BlockAllocator> block_allocator_;
    std::unique_ptr<kvcache::PrefixCache> prefix_cache_;
    std::unique_ptr<model::DecodeScratch> decode_scratch_;

    // Owned sub-components
    std::unique_ptr<ServingLoop> serving_loop_;
    std::unique_ptr<Profiler> profiler_;

    void init_block_pool();
    void build_stop_token_ids();
};

} // namespace zedinfer
