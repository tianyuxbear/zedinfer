#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/batch_context.hpp"
#include "zedinfer/request.hpp"

#include <atomic>
#include <condition_variable>
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
class SSMSnapshotCache;
} // namespace kvcache
namespace model {
class SSMStatePool;
} // namespace model
namespace sampler {
class Sampler;
class GeneralSampler;
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

    // Qwen3.5 MTP speculative decoding. Off by default — must be opted in
    // via --mtp on the CLI (or ZEDINFER_MTP_SPEC=1 / ZEDINFER_MTP_DEBUG=1
    // env vars, kept as research toggles). When on AND the loaded model
    // ships an MTP head, the serving loop runs MTPModule after each main
    // forward to draft t+2 and the next scheduler step issues a 2-token
    // verify batch.
    bool mtp_enabled = false;
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

    // Set SSM snapshot cache (optional, paired with prefix_cache on hybrid
    // models). When both are set, the scheduler honors PrefixCache hits only
    // for the exact full prompt AND when the SSM snapshot is restored — this
    // keeps the linear-attention state coherent with the cached KV. Without
    // this cache wired, prefix matching on hybrid models is silently ignored
    // (partial hits would be prefix-blind for SSM layers).
    void set_ssm_snapshot_cache(kvcache::SSMSnapshotCache* cache);

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

    // Block the calling (serving) thread until there is work to do or `running`
    // becomes false. The wait shares submit_mutex_ with submit(), so the wakeup
    // condition is evaluated under the same lock that guards the queues — no
    // lost wakeups and no data race on the predicate state.
    void wait_for_work(const std::atomic<bool>& running);
    // Wake any thread parked in wait_for_work() (e.g. on stop()).
    void wake_waiters();

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
     *
     * Three samplers are passed so the scheduler can route each request:
     *   - default_sampler: used when the request has NO sampling overrides
     *     (matches the model's generation_config.json choice)
     *   - argmax_sampler:  used when override sets use_argmax=true or
     *     temperature == 0 (OpenAI greedy semantics)
     *   - general_sampler: used otherwise; per-request temperature/top_p/top_k
     *     /repetition_penalty/seed land via setParams() right before sample()
     */
    void process_results(ScheduledBatch& batch, tensor_t logits, sampler::Sampler& default_sampler,
                         sampler::Sampler& argmax_sampler, sampler::GeneralSampler& general_sampler,
                         tokenizer::Tokenizer& tokenizer, const std::vector<int>& stop_token_ids);

private:
    SchedulerConfig config_;
    kvcache::BlockAllocator* block_allocator_ = nullptr;
    kvcache::PrefixCache* prefix_cache_ = nullptr;
    kvcache::SSMSnapshotCache* ssm_snapshot_cache_ = nullptr;
    model::SSMStatePool* ssm_state_pool_ = nullptr;
    int think_open_token_id_ = -1;
    int think_close_token_id_ = -1;
    int double_newline_token_id_ = -1;
    // mutable so the const status queries (has_work/pending_count/active_count)
    // can lock it; they are called from HTTP threads concurrently with submit().
    mutable std::mutex submit_mutex_;
    std::condition_variable work_cv_;

    std::deque<std::unique_ptr<InferenceRequest>> waiting_queue_;
    std::vector<std::unique_ptr<InferenceRequest>> active_requests_; // decode phase

    uint64_t next_request_id_ = 1;

    bool can_admit(const InferenceRequest& req) const;
    void allocate_blocks_for_request(InferenceRequest* req);
    void complete_request(InferenceRequest& req);
};

} // namespace zedinfer
