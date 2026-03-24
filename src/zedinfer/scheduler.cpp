#include "zedinfer/scheduler.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "backend/kvcache/prefix_cache.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/base.hpp"
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

void Scheduler::set_prefix_cache(kvcache::PrefixCache *cache) {
    prefix_cache_ = cache;
}

void Scheduler::submit(std::unique_ptr<InferenceRequest> request) {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    if (static_cast<int>(waiting_queue_.size()) >= config_.max_queue_size) {
        throw std::runtime_error("[Scheduler] Queue full");
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

void Scheduler::cleanup_failed_requests() {
    // Clean active requests (decode phase)
    active_requests_.erase(
        std::remove_if(active_requests_.begin(), active_requests_.end(),
            [](const auto &ptr) { return ptr->phase == RequestPhase::COMPLETE; }),
        active_requests_.end());

    // Also clean waiting queue — failed prefill requests (chunked) might still be there
    while (!waiting_queue_.empty() &&
           waiting_queue_.front()->phase == RequestPhase::COMPLETE) {
        waiting_queue_.pop_front();
    }
}

bool Scheduler::can_admit(const InferenceRequest &req) const {
    if (!block_allocator_) return true;

    int bs = block_allocator_->block_size();
    int num_layers = block_allocator_->num_layers();
    int prompt_len = static_cast<int>(req.input_ids.size());

    // Session multi-turn: only count ADDITIONAL blocks needed
    if (req.block_table_ref && req.block_table_ref->num_layers > 0) {
        int current_blocks = static_cast<int>(req.block_table_ref->k_blocks[0].size());
        int total_after = req.block_table_ref->seq_len + prompt_len +
                          std::min(req.config.max_new_tokens, 256);
        int needed_per_layer = (total_after + bs - 1) / bs;
        int additional = std::max(0, needed_per_layer - current_blocks);
        return block_allocator_->available_blocks() >= additional * num_layers * 2;
    }

    // New request: estimate full allocation
    int est_tokens = prompt_len + std::min(req.config.max_new_tokens, 256);
    int blocks_per_layer = (est_tokens + bs - 1) / bs;
    int blocks_needed = blocks_per_layer * num_layers * 2;
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

        // Extend blocks if needed for the next token
        if (block_allocator_) {
            auto &bt = req_ptr->active_block_table();
            int next_pos = bt.seq_len + 1;
            int bs = block_allocator_->block_size();
            int blocks_needed = (next_pos + bs - 1) / bs;
            for (int layer = 0; layer < bt.num_layers; ++layer) {
                while (static_cast<int>(bt.k_blocks[layer].size()) < blocks_needed)
                    block_allocator_->extend_sequence(bt, layer, true);
                while (static_cast<int>(bt.v_blocks[layer].size()) < blocks_needed)
                    block_allocator_->extend_sequence(bt, layer, false);
            }
        }

        batch.decode_requests.push_back(req_ptr.get());
        token_budget--;
    }

    // 2. Admit new prefill requests
    int prefill_budget = std::min(token_budget, config_.max_prefill_tokens);

    std::lock_guard<std::mutex> lock(submit_mutex_);

    while (!waiting_queue_.empty() && prefill_budget > 0) {
        auto *req = waiting_queue_.front().get();

        if (static_cast<int>(active_requests_.size()) +
            static_cast<int>(batch.prefill_requests.size()) >= config_.max_batch_requests)
            break;

        if (!can_admit(*req)) {
            LOGW << "[Scheduler] Cannot admit request " << req->request_id
                 << ": prompt=" << req->input_ids.size() << " tokens"
                 << ", available_blocks=" << (block_allocator_ ? block_allocator_->available_blocks() : -1)
                 << ", total_blocks=" << (block_allocator_ ? (block_allocator_->available_blocks()) : -1);
            break;
        }

        // Allocate or extend blocks
        if (block_allocator_) {
            auto &bt = req->active_block_table();
            if (bt.num_layers == 0) {
                // New request: try prefix cache match first
                int cached_tokens = 0;
                if (prefix_cache_ && !req->input_ids.empty()) {
                    kvcache::SequenceBlockTable matched;
                    cached_tokens = prefix_cache_->match_prefix(
                        req->input_ids, block_allocator_->block_size(), matched);
                    if (cached_tokens > 0) {
                        // Use matched blocks as starting point
                        if (req->block_table_ref) {
                            *req->block_table_ref = std::move(matched);
                        } else {
                            req->block_table = std::move(matched);
                        }
                        req->prefill_progress = cached_tokens;
                    }
                }

                auto &bt2 = req->active_block_table();
                if (bt2.num_layers == 0) {
                    // No prefix match — allocate from scratch
                    int est_tokens = std::min(
                        static_cast<int>(req->input_ids.size()) + 256,
                        static_cast<int>(req->input_ids.size()) + req->config.max_new_tokens);
                    if (req->block_table_ref) {
                        *req->block_table_ref = block_allocator_->allocate_sequence(est_tokens);
                    } else {
                        req->block_table = block_allocator_->allocate_sequence(est_tokens);
                    }
                } else {
                    // Prefix matched — extend for remaining tokens
                    int total_after = cached_tokens + (static_cast<int>(req->input_ids.size()) - cached_tokens) + 256;
                    int bs = block_allocator_->block_size();
                    int blocks_needed = (total_after + bs - 1) / bs;
                    for (int layer = 0; layer < bt2.num_layers; ++layer) {
                        while (static_cast<int>(bt2.k_blocks[layer].size()) < blocks_needed)
                            block_allocator_->extend_sequence(bt2, layer, true);
                        while (static_cast<int>(bt2.v_blocks[layer].size()) < blocks_needed)
                            block_allocator_->extend_sequence(bt2, layer, false);
                    }
                }
            } else {
                // Session multi-turn: extend blocks for new tokens
                int total_after = bt.seq_len + static_cast<int>(req->input_ids.size()) + 256;
                int bs = block_allocator_->block_size();
                int blocks_needed = (total_after + bs - 1) / bs;
                for (int layer = 0; layer < bt.num_layers; ++layer) {
                    while (static_cast<int>(bt.k_blocks[layer].size()) < blocks_needed)
                        block_allocator_->extend_sequence(bt, layer, true);
                    while (static_cast<int>(bt.v_blocks[layer].size()) < blocks_needed)
                        block_allocator_->extend_sequence(bt, layer, false);
                }
            }
        }

        // Chunked prefill
        int remaining_prompt = static_cast<int>(req->input_ids.size()) - req->prefill_progress;
        int chunk = std::min(remaining_prompt, prefill_budget);

        batch.prefill_requests.push_back(req);
        batch.prefill_chunk_starts.push_back(req->prefill_progress);  // record BEFORE advancing
        batch.prefill_chunk_sizes.push_back(chunk);
        prefill_budget -= chunk;

        req->prefill_progress += chunk;
        req->phase = RequestPhase::PREFILL;

        bool prefill_complete = (req->prefill_progress >= static_cast<int>(req->input_ids.size()));

        if (prefill_complete) {
            active_requests_.push_back(std::move(waiting_queue_.front()));
            waiting_queue_.pop_front();
        } else {
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
        // Check cancellation (client disconnected)
        if (req->cancelled && req->cancelled->load()) {
            LOGI << "[Scheduler] Request " << req->request_id << " cancelled";
            complete_request(*req);
            offset++;
            continue;
        }

        // Pass [offset : offset+1] to sampler — single row, no double-slice issue
        auto req_logits = logits->slice(0, offset, offset + 1);
        int token = sampler.sample(req_logits);

        req->output_ids.push_back(token);
        req->last_token = token;
        req->generated_count++;
        req->active_block_table().seq_len++;

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

        req->active_block_table().seq_len += chunk;

        if (req->phase == RequestPhase::DECODE ||
            req->prefill_progress >= static_cast<int>(req->input_ids.size())) {
            req->phase = RequestPhase::DECODE;

            // Insert full blocks into prefix cache after prefill completes
            if (prefix_cache_ && block_allocator_) {
                prefix_cache_->insert_blocks(
                    req->input_ids, block_allocator_->block_size(),
                    req->active_block_table());
            }

            auto req_logits = logits->slice(0, offset, offset + chunk);
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

    // Remove completed requests
    active_requests_.erase(
        std::remove_if(active_requests_.begin(), active_requests_.end(),
            [](const auto &ptr) { return ptr->phase == RequestPhase::COMPLETE; }),
        active_requests_.end());
}

void Scheduler::complete_request(InferenceRequest &req) {
    req.phase = RequestPhase::COMPLETE;

    // Release blocks for owned tables (not borrowed session tables).
    // Uses release (not free) so cached prefix blocks stay in the pool.
    if (block_allocator_ && !req.block_table_ref && req.block_table.num_layers > 0) {
        block_allocator_->release_sequence(req.block_table);
    }

    GenerationResult result;
    result.output_ids = std::move(req.output_ids);
    result.stats = req.stats;
    result.stats.generated_tokens = result.output_ids.size();
    result.stats.total_tokens = result.stats.prompt_tokens + result.stats.generated_tokens;

    try {
        req.result_promise.set_value(std::move(result));
    } catch (...) {
    }
}

} // namespace zedinfer
