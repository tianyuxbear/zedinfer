#pragma once

#include "zedinfer/activation.hpp"
#include "zedinfer/request.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

namespace zedinfer {

namespace model {
class Model;
}
namespace kvcache {
class KVCache;
}
namespace sampler {
class Sampler;
}
namespace tokenizer {
class Tokenizer;
}

/**
 * Configuration for the scheduler.
 * In single-request mode only max_queue_size is enforced.
 * Other fields are defined for PR-7 (continuous batching).
 */
struct SchedulerConfig {
    int max_batch_tokens = 2048;
    int max_batch_requests = 64;
    int max_prefill_tokens = 512;
    int max_queue_size = 256;
};

/**
 * Request scheduler for inference engine.
 *
 * Single-request mode (PR-6): processes one request at a time via run_one().
 * Batched mode (PR-7): assembles multi-sequence batches via schedule().
 */
class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config = {});

    /**
     * Submit a new request to the waiting queue.
     * Assigns a unique request_id. Throws if queue is full.
     */
    void submit(std::unique_ptr<InferenceRequest> request);

    /** True if there are pending or active requests. */
    bool has_work() const;

    /** Number of requests in the waiting queue. */
    int pending_count() const;

    /** Number of currently active requests (0 or 1 in single-request mode). */
    int active_count() const;

    /**
     * Run one request to completion (single-request mode).
     * Pops the front request from the queue, runs prefill + decode,
     * and returns the result.
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
    std::deque<std::unique_ptr<InferenceRequest>> waiting_queue_;
    InferenceRequest *active_request_ = nullptr;
    uint64_t next_request_id_ = 1;
};

} // namespace zedinfer
