#include "zedinfer/scheduler.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "frontend/models/base.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/base.hpp"
#include "backend/kvcache/base.hpp"
#include "utils/logging.hpp"

#include <algorithm>
#include <chrono>
#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer {

static bool is_stop_token(int token_id, const std::vector<int> &stop_ids) {
    for (int stop_id : stop_ids) {
        if (token_id == stop_id) return true;
    }
    return false;
}

Scheduler::Scheduler(SchedulerConfig config) : config_(config) {}

void Scheduler::set_block_allocator(kvcache::BlockAllocator *allocator) {
    block_allocator_ = allocator;
}

void Scheduler::submit(std::unique_ptr<InferenceRequest> request) {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    if (static_cast<int>(waiting_queue_.size()) >= config_.max_queue_size) {
        throw std::runtime_error("[Scheduler] Queue full (max_queue_size=" +
                                 std::to_string(config_.max_queue_size) + ")");
    }
    request->request_id = next_request_id_++;
    request->phase = RequestPhase::QUEUED;
    waiting_queue_.push_back(std::move(request));
}

bool Scheduler::has_work() const {
    return !waiting_queue_.empty() || !active_requests_.empty();
}

int Scheduler::pending_count() const {
    return static_cast<int>(waiting_queue_.size());
}

int Scheduler::active_count() const {
    return static_cast<int>(active_requests_.size());
}

bool Scheduler::can_admit(const InferenceRequest &req) const {
    if (!block_allocator_) return true;
    int prompt_len = static_cast<int>(req.input_ids.size());
    int bs = block_allocator_->block_size();
    int est_tokens = prompt_len + std::min(req.config.max_new_tokens, 256);
    int blocks_per_layer = (est_tokens + bs - 1) / bs;
    int blocks_needed = blocks_per_layer * 2; // rough per-layer estimate
    return block_allocator_->available_blocks() >= blocks_needed;
}

// ============================================================================
// Batched Scheduling
// ============================================================================

ScheduledBatch Scheduler::schedule() {
    ScheduledBatch batch;
    int token_budget = config_.max_batch_tokens;

    // 1. Decode-first: all active decode requests (1 token each)
    for (auto &req_ptr : active_requests_) {
        if (token_budget <= 0) break;
        batch.decode_requests.push_back(req_ptr.get());
        token_budget--;
    }

    // 2. Admit new prefill requests
    int prefill_budget = std::min(token_budget, config_.max_prefill_tokens);

    // Lock for queue access (submit may be concurrent)
    std::lock_guard<std::mutex> lock(submit_mutex_);

    while (!waiting_queue_.empty() && prefill_budget > 0) {
        auto *req = waiting_queue_.front().get();

        if (static_cast<int>(active_requests_.size()) +
            static_cast<int>(batch.prefill_requests.size()) >= config_.max_batch_requests)
            break;

        if (!can_admit(*req))
            break;

        // Allocate blocks for new request
        if (block_allocator_ && req->block_table.num_layers == 0) {
            int est_tokens = std::min(
                static_cast<int>(req->input_ids.size()) + 256,
                static_cast<int>(req->input_ids.size()) + req->config.max_new_tokens);
            req->block_table = block_allocator_->allocate_sequence(est_tokens);
        }

        // Chunked prefill
        int remaining_prompt = static_cast<int>(req->input_ids.size()) - req->prefill_progress;
        int chunk = std::min(remaining_prompt, prefill_budget);

        batch.prefill_requests.push_back(req);
        batch.prefill_chunk_sizes.push_back(chunk);
        prefill_budget -= chunk;

        req->prefill_progress += chunk;
        req->phase = RequestPhase::PREFILL;

        bool prefill_complete = (req->prefill_progress >= static_cast<int>(req->input_ids.size()));

        if (prefill_complete) {
            // Move ownership to active list, then pop from queue
            active_requests_.push_back(std::move(waiting_queue_.front()));
            waiting_queue_.pop_front();
        } else {
            // Chunked: request stays in queue for next iteration
            break;
        }
    }

    return batch;
}

void Scheduler::process_results(
    ScheduledBatch &batch,
    tensor_t logits,
    sampler::Sampler &sampler,
    tokenizer::Tokenizer &tokenizer,
    const std::vector<int> &stop_token_ids) {

    int offset = 0;

    // Process decode results
    for (auto *req : batch.decode_requests) {
        // Sample from this request's logit position
        auto req_logits = logits->slice(0, offset, offset + 1);
        int token = sampler.sample(req_logits);

        req->output_ids.push_back(token);
        req->last_token = token;
        req->generated_count++;
        req->block_table.seq_len++;

        if (is_stop_token(token, stop_token_ids) ||
            req->generated_count >= req->config.max_new_tokens) {
            complete_request(*req);
        } else if (req->config.stream && req->stream_callback) {
            req->stream_callback(tokenizer.decode({token}));
        }
        offset++;
    }

    // Process prefill results
    for (size_t i = 0; i < batch.prefill_requests.size(); ++i) {
        auto *req = batch.prefill_requests[i];
        int chunk = batch.prefill_chunk_sizes[i];

        // Update block table seq_len (tokens now in KV cache)
        req->block_table.seq_len += chunk;

        // If prefill complete, sample first token
        if (req->phase == RequestPhase::DECODE ||
            req->prefill_progress >= static_cast<int>(req->input_ids.size())) {
            req->phase = RequestPhase::DECODE;
            // Sample from last token of this prefill chunk
            auto req_logits = logits->slice(0, offset + chunk - 1, offset + chunk);
            int token = sampler.sample(req_logits);

            req->output_ids.push_back(token);
            req->last_token = token;
            req->generated_count = 1;
            req->stats.prompt_tokens = req->input_ids.size();

            if (is_stop_token(token, stop_token_ids)) {
                complete_request(*req);
            } else if (req->config.stream && req->stream_callback) {
                req->stream_callback(tokenizer.decode({token}));
            }
        }
        offset += chunk;
    }

    // Remove completed requests from active list
    active_requests_.erase(
        std::remove_if(active_requests_.begin(), active_requests_.end(),
            [](const auto &ptr) { return ptr->phase == RequestPhase::COMPLETE; }),
        active_requests_.end());
}

void Scheduler::complete_request(InferenceRequest &req) {
    req.phase = RequestPhase::COMPLETE;

    // Free blocks
    if (block_allocator_ && req.block_table.num_layers > 0) {
        block_allocator_->free_sequence(req.block_table);
    }

    // Fulfill promise for async callers
    GenerationResult result;
    result.output_ids = std::move(req.output_ids);
    result.stats = req.stats;
    result.stats.generated_tokens = result.output_ids.size();
    result.stats.total_tokens = result.stats.prompt_tokens + result.stats.generated_tokens;

    try {
        req.result_promise.set_value(std::move(result));
    } catch (...) {
        // Promise may already be satisfied or broken
    }
}

// ============================================================================
// Single-Request Mode (backward compatibility)
// ============================================================================

GenerationResult Scheduler::run_one(
    model::Model &model,
    kvcache::SequenceBlockTable &block_table,
    kvcache::BlockPool &pool,
    const ExecutorConfig &exec_config,
    sampler::Sampler &sampler,
    tokenizer::Tokenizer &tokenizer,
    const std::vector<int> &stop_token_ids) {

    std::unique_ptr<InferenceRequest> request_ptr;
    {
        std::lock_guard<std::mutex> lock(submit_mutex_);
        if (waiting_queue_.empty()) {
            throw std::runtime_error("[Scheduler] No pending requests");
        }
        request_ptr = std::move(waiting_queue_.front());
        waiting_queue_.pop_front();
    }

    InferenceRequest &req = *request_ptr;
    GenerationStats &stats = req.stats;
    stats.prompt_tokens = req.input_ids.size();
    req.output_ids.reserve(req.config.max_new_tokens);

    auto model_cfg = model.forward_config();

    // Prefill: use PagedForwardContext with the session's block_table
    req.phase = RequestPhase::PREFILL;
    auto t0 = std::chrono::high_resolution_clock::now();

    int past_len = block_table.seq_len;
    {
        model::PagedForwardContext ctx(req.input_ids, past_len, block_table, pool);
        tensor_t logits = model::transformer_forward(model_cfg, ctx, exec_config);
        int next_token = sampler.sample(logits);

        auto t1 = std::chrono::high_resolution_clock::now();
        stats.prefill_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats.total_time_ms += stats.prefill_time_ms;

        req.output_ids.push_back(next_token);
        req.last_token = next_token;
        req.generated_count = 1;

        if (is_stop_token(next_token, stop_token_ids)) {
            req.phase = RequestPhase::COMPLETE;
            stats.generated_tokens = req.output_ids.size();
            stats.total_tokens = stats.prompt_tokens + stats.generated_tokens;
            return GenerationResult{std::move(req.output_ids), stats};
        }

        if (req.config.stream && req.stream_callback) {
            req.stream_callback(tokenizer.decode({next_token}));
        }
    }

    // Decode loop: each step creates a PagedForwardContext with 1 token
    req.phase = RequestPhase::DECODE;

    for (int i = 1; i < req.config.max_new_tokens; ++i) {
        auto s0 = std::chrono::high_resolution_clock::now();

        // Ensure enough blocks for the new token
        int total_tokens = block_table.seq_len + 1;
        int bs = pool.config().block_size;
        int blocks_needed = (total_tokens + bs - 1) / bs;
        for (int layer = 0; layer < block_table.num_layers; ++layer) {
            while (static_cast<int>(block_table.k_blocks[layer].size()) < blocks_needed) {
                block_table.k_blocks[layer].push_back(pool.allocate());
            }
            while (static_cast<int>(block_table.v_blocks[layer].size()) < blocks_needed) {
                block_table.v_blocks[layer].push_back(pool.allocate());
            }
        }

        std::vector<int> decode_token = {req.last_token};
        model::PagedForwardContext ctx(decode_token, block_table.seq_len, block_table, pool);
        tensor_t logits = model::transformer_forward(model_cfg, ctx, exec_config);
        int next_token = sampler.sample(logits);

        auto s1 = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
        stats.decode_time_ms += step_ms;
        stats.total_time_ms += step_ms;

        req.output_ids.push_back(next_token);
        req.last_token = next_token;
        req.generated_count++;

        if (is_stop_token(next_token, stop_token_ids)) break;

        if (req.config.stream && req.stream_callback) {
            req.stream_callback(tokenizer.decode({next_token}));
        }

        if (block_table.seq_len >= static_cast<int>(tokenizer.get_config().model_max_length)) {
            if (req.config.verbose) {
                LOGI << "[Scheduler] Reached max sequence length";
            }
            break;
        }
    }

    req.phase = RequestPhase::COMPLETE;
    stats.generated_tokens = req.output_ids.size();
    stats.total_tokens = stats.prompt_tokens + stats.generated_tokens;
    return GenerationResult{std::move(req.output_ids), stats};
}

} // namespace zedinfer
