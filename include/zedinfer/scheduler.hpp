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
}
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
    void set_block_allocator(kvcache::BlockAllocator *allocator);

    /**
     * Submit a new request. Thread-safe (can be called from HTTP threads).
     */
    void submit(std::unique_ptr<InferenceRequest> request);

    /** True if there are pending or active requests. */
    bool has_work() const;
    int pending_count() const;
    int active_count() const;

    /**
     * Schedule next batch for one forward pass.
     * Decode-first: all active decode requests, then admit new prefills.
     */
    ScheduledBatch schedule();

    /**
     * Process results after model forward.
     * Samples tokens, advances state, completes finished requests.
     */
    void process_results(
        ScheduledBatch &batch,
        tensor_t logits,
        sampler::Sampler &sampler,
        tokenizer::Tokenizer &tokenizer,
        const std::vector<int> &stop_token_ids);

    /**
     * Single-request mode (backward compat for bench/chat/ping).
     */
    GenerationResult run_one(
        model::Model &model,
        kvcache::KVCache &kvcache,
        const ExecutorConfig &exec_config,
        sampler::Sampler &sampler,
        tokenizer::Tokenizer &tokenizer,
        const std::vector<int> &stop_token_ids);

private:
    SchedulerConfig config_;
    kvcache::BlockAllocator *block_allocator_ = nullptr;
    std::mutex submit_mutex_;

    std::deque<std::unique_ptr<InferenceRequest>> waiting_queue_;
    std::vector<std::unique_ptr<InferenceRequest>> active_requests_; // decode phase
    std::vector<std::unique_ptr<InferenceRequest>> completing_;      // just completed

    uint64_t next_request_id_ = 1;

    bool can_admit(const InferenceRequest &req) const;
    void complete_request(InferenceRequest &req);
};

} // namespace zedinfer
