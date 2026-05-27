#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/batch_context.hpp"
#include "zedinfer/request.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace zedinfer {

namespace model {
class Model;
}
namespace kvcache {
class KVCache;
class BlockAllocator;
class PrefixCache;
} // namespace kvcache
namespace model {
class SSMStatePool;
} // namespace model
namespace sampler {
class Sampler;
}
namespace tokenizer {
class Tokenizer;
}

/**
 * Configuration for the scheduler.
 */
struct SchedulerConfig {
    int max_batch_tokens = 2048;
    int max_batch_requests = 64;
    int max_prefill_tokens = 512;
    int max_queue_size = 256;

    // KV cache memory management
    float gpu_memory_utilization = 0.9f;
    int kv_block_size = 16;
    bool use_paged_kvcache = true;
};

/**
 * Request scheduler for inference engine.
 *
 * Single-request mode: run_one() processes one request at a time.
 * Batched mode: schedule() assembles multi-sequence batches,
 * process_results() advances request state after forward.
 */
class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config = {});

    // Set block allocator for batched mode (called after engine creates block pool)
    void set_block_allocator(kvcache::BlockAllocator* allocator);

    // Set prefix cache (optional, enables prefix sharing across requests)
    void set_prefix_cache(kvcache::PrefixCache* cache);

    // Wire an SSM state pool for hybrid models (Qwen3.5). When set, can_admit
    // also checks pool availability; admission grabs a slot, completion releases it.
    // Pass nullptr (the default) for non-hybrid models.
    void set_ssm_state_pool(model::SSMStatePool* pool);

    // Register the reasoning model's <think> / </think> token ids so the
    // scheduler can (a) initialize per-request in_thinking state by scanning
    // the prompt and (b) force-emit </think> after GenerationConfig::
    // max_think_tokens tokens to escape long-generation drift on GPTQ-Int4
    // weights. Pass -1 for models that do not have these as special tokens
    // (the force-emit path becomes a no-op).
    //
    // `double_newline_id` is the tokenizer id for "\n\n". After the scheduler
    // force-emits </think>, it also force-emits one "\n\n" to recreate the
    // </think>\n\n pattern the model was trained on, anchoring the post-
    // thinking state so it can transition to the answer instead of
    // continuing the truncated reasoning. Pass -1 to skip the newline.
    void set_think_token_ids(int open_id, int close_id, int double_newline_id = -1);

    /**
     * Submit a new request. Thread-safe (can be called from HTTP threads).
     */
    void submit(std::unique_ptr<InferenceRequest> request);

    /** True if there are pending or active requests. */
    bool has_work() const;
    int pending_count() const;
    int active_count() const;

    // Remove completed/failed requests from active list (called after fail_batch)
    void cleanup_failed_requests();

    /**
     * Schedule next batch for one forward pass.
     * Decode-first: all active decode requests, then admit new prefills.
     */
    ScheduledBatch schedule();

    /**
     * Process results after model forward.
     * Samples tokens, advances state, completes finished requests.
     */
    void process_results(ScheduledBatch& batch, tensor_t logits, sampler::Sampler& sampler,
                         tokenizer::Tokenizer& tokenizer, const std::vector<int>& stop_token_ids);

private:
    SchedulerConfig config_;
    kvcache::BlockAllocator* block_allocator_ = nullptr;
    kvcache::PrefixCache* prefix_cache_ = nullptr;
    model::SSMStatePool* ssm_state_pool_ = nullptr;
    int think_open_token_id_ = -1;
    int think_close_token_id_ = -1;
    int double_newline_token_id_ = -1;
    std::mutex submit_mutex_;

    std::deque<std::unique_ptr<InferenceRequest>> waiting_queue_;
    std::vector<std::unique_ptr<InferenceRequest>> active_requests_; // decode phase

    uint64_t next_request_id_ = 1;

    bool can_admit(const InferenceRequest& req) const;
    void allocate_blocks_for_request(InferenceRequest* req);
    void complete_request(InferenceRequest& req);
};

} // namespace zedinfer
