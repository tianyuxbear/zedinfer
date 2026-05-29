#include "zedinfer/scheduler.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "backend/kvcache/prefix_cache.hpp"
#include "backend/kvcache/ssm_snapshot_cache.hpp"
#include "frontend/models/ssm_state_pool.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/base.hpp"
#include "utils/logging.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer {

static bool is_stop_token(int token_id, const std::vector<int>& stop_ids) {
    for (int stop_id : stop_ids) {
        if (token_id == stop_id) {
            return true;
        }
    }
    return false;
}

Scheduler::Scheduler(SchedulerConfig config) : config_(config) {}

void Scheduler::set_block_allocator(kvcache::BlockAllocator* allocator) {
    block_allocator_ = allocator;
}

void Scheduler::set_prefix_cache(kvcache::PrefixCache* cache) {
    prefix_cache_ = cache;
}

void Scheduler::set_ssm_snapshot_cache(kvcache::SSMSnapshotCache* cache) {
    ssm_snapshot_cache_ = cache;
}

void Scheduler::submit(std::unique_ptr<InferenceRequest> request) {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    if (static_cast<int>(waiting_queue_.size()) >= config_.max_queue_size) {
        throw std::runtime_error("[Scheduler] Queue full");
    }
    request->request_id = next_request_id_++;
    request->phase = RequestPhase::QUEUED;
    waiting_queue_.push_back(std::move(request));
    // Wake the serving thread if it is parked in wait_for_work(). Notifying
    // under the lock is intentional: it pairs the queue mutation with the
    // wakeup so a concurrent waiter cannot miss it.
    work_cv_.notify_one();
}

bool Scheduler::has_work() const {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    return !waiting_queue_.empty() || !active_requests_.empty();
}

int Scheduler::pending_count() const {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    return static_cast<int>(waiting_queue_.size());
}

int Scheduler::active_count() const {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    return static_cast<int>(active_requests_.size());
}

void Scheduler::wait_for_work(const std::atomic<bool>& running) {
    std::unique_lock<std::mutex> lock(submit_mutex_);
    work_cv_.wait(lock, [this, &running] {
        return !waiting_queue_.empty() || !active_requests_.empty() || !running.load();
    });
}

void Scheduler::wake_waiters() {
    // Acquire the lock so a thread between its predicate check and blocking in
    // wait_for_work() cannot miss this wakeup.
    { std::lock_guard<std::mutex> lock(submit_mutex_); }
    work_cv_.notify_all();
}

void Scheduler::cleanup_failed_requests() {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    // Clean active requests (decode phase)
    active_requests_.erase(std::remove_if(active_requests_.begin(), active_requests_.end(),
                                          [](const auto& ptr) { return ptr->phase == RequestPhase::COMPLETE; }),
                           active_requests_.end());

    // Also clean waiting queue — failed prefill requests (chunked) might still be there
    while (!waiting_queue_.empty() && waiting_queue_.front()->phase == RequestPhase::COMPLETE) {
        waiting_queue_.pop_front();
    }
}

bool Scheduler::can_admit(const InferenceRequest& req) const {
    // SSM slot check (hybrid models only): if the pool exists and the request
    // does not already hold a slot, we need at least one free slot to admit.
    if (ssm_state_pool_ != nullptr && req.ssm_slot_idx() < 0) {
        if (ssm_state_pool_->num_free_slots() < 1) {
            return false;
        }
    }

    if (!block_allocator_) {
        return true;
    }

    int bs = block_allocator_->block_size();
    int num_layers = block_allocator_->num_layers();
    int prompt_len = static_cast<int>(req.input_ids.size());

    // Multi-turn with existing blocks: only count ADDITIONAL blocks needed
    if (req.has_block_table() && req.block_table().num_layers > 0) {
        int current_blocks = static_cast<int>(req.block_table().pages[0].size());
        int total_after = req.block_table().seq_len + prompt_len + std::min(req.config.max_new_tokens, 256);
        int needed_per_layer = (total_after + bs - 1) / bs;
        int additional = std::max(0, needed_per_layer - current_blocks);
        return block_allocator_->available_blocks() >= additional * num_layers;
    }

    // New request: estimate full allocation
    int est_tokens = prompt_len + std::min(req.config.max_new_tokens, 256);
    int blocks_per_layer = (est_tokens + bs - 1) / bs;
    int blocks_needed = blocks_per_layer * num_layers;
    return block_allocator_->available_blocks() >= blocks_needed;
}

void Scheduler::set_ssm_state_pool(model::SSMStatePool* pool) {
    ssm_state_pool_ = pool;
}

void Scheduler::set_think_token_ids(int open_id, int close_id, int double_newline_id) {
    think_open_token_id_ = open_id;
    think_close_token_id_ = close_id;
    double_newline_token_id_ = double_newline_id;
}

namespace {

// Count occurrences of a particular token id in a list. Used to decide whether
// the prompt's last <think> is still open at generation start by comparing the
// number of <think> opens to </think> closes; if opens > closes, we're inside
// an unclosed thinking block.
int count_token(const std::vector<int>& ids, int target_id) {
    if (target_id < 0) {
        return 0;
    }
    int n = 0;
    for (int id : ids) {
        if (id == target_id) {
            ++n;
        }
    }
    return n;
}

} // namespace

// ============================================================================
// Batched Scheduling
// ============================================================================

void Scheduler::allocate_blocks_for_request(InferenceRequest* req) {
    // Acquire an SSM slot for hybrid models (Qwen3.5). Idempotent — re-admission
    // of a request that already holds a slot is a no-op. The slot is released
    // in complete_request().
    if (ssm_state_pool_ != nullptr && req->ssm_slot_idx() < 0) {
        const int slot = ssm_state_pool_->acquire_slot();
        ssm_state_pool_->reset_slot(slot);
        req->set_ssm_slot_idx(slot);
    }

    if (!block_allocator_) {
        return;
    }

    if (!req->has_block_table() || req->block_table().num_layers == 0) {
        // New request: try prefix cache match first.
        //
        // Hybrid model gate: KV-only prefix reuse is unsafe when the request
        // also has SSM/conv state that would be left at post-reset zero — the
        // linear-attention layers would run "prefix-blind" and diverge from a
        // cold-cache run. We only honor a PrefixCache hit when (a) the SSM
        // snapshot cache is wired, (b) it holds a snapshot keyed on the EXACT
        // full prompt (so partial-prefix KV reuse is impossible), and (c) the
        // snapshot restores cleanly into the freshly-acquired SSM slot.
        if (prefix_cache_ && !req->input_ids.empty()) {
            const bool hybrid                = (ssm_state_pool_ != nullptr);
            const bool ssm_snapshot_present  = hybrid && ssm_snapshot_cache_
                                              && ssm_snapshot_cache_->has(req->input_ids);
            const bool skip_prefix_for_hybrid = hybrid && !ssm_snapshot_present;

            if (!skip_prefix_for_hybrid) {
                kvcache::SequenceBlockTable matched;
                int cached = prefix_cache_->match_prefix(req->input_ids, block_allocator_->block_size(), matched);

                // Hybrid models accept only exact full-prompt matches.
                // Anything shorter would expose the SSM bleed bug.
                if (hybrid && cached > 0 && cached < static_cast<int>(req->input_ids.size())) {
                    block_allocator_->release_sequence(matched);
                    cached = 0;
                }

                if (hybrid && cached > 0) {
                    // SSM restore for the matched (full) prompt.
                    if (!ssm_snapshot_cache_
                        || !ssm_snapshot_cache_->try_restore(req->input_ids, req->ssm_slot_idx())) {
                        // Snapshot evicted between has() and try_restore() (LRU race) —
                        // fall back to cold prefill rather than risk a prefix-blind run.
                        block_allocator_->release_sequence(matched);
                        cached = 0;
                    }
                }

                if (cached > 0) {
                    if (req->has_block_table()) {
                        req->block_table() = std::move(matched);
                    } else {
                        req->own_block_table(std::move(matched));
                    }
                    req->prefill_progress = cached;
                }
            }
        }

        if (!req->has_block_table() || req->block_table().num_layers == 0) {
            // No prefix match — allocate from scratch
            int est = std::min(static_cast<int>(req->input_ids.size()) + 256,
                               static_cast<int>(req->input_ids.size()) + req->config.max_new_tokens);
            auto allocated = block_allocator_->allocate_sequence(est);
            if (req->has_block_table()) {
                req->block_table() = std::move(allocated);
            } else {
                req->own_block_table(std::move(allocated));
            }
        } else {
            // Prefix matched — extend for remaining tokens
            int total = static_cast<int>(req->input_ids.size()) + 256;
            block_allocator_->ensure_blocks(req->block_table(), total);
        }
    } else {
        // Multi-turn: extend blocks for new tokens
        auto& bt = req->block_table();
        int total = bt.seq_len + static_cast<int>(req->input_ids.size()) + 256;
        block_allocator_->ensure_blocks(bt, total);
    }
}

ScheduledBatch Scheduler::schedule() {
    ScheduledBatch batch;
    int token_budget = config_.max_batch_tokens;

    // 1. Decode-first: all active decode requests (1 token each)
    for (auto& req_ptr : active_requests_) {
        if (token_budget <= 0) {
            break;
        }
        if (block_allocator_) {
            block_allocator_->ensure_blocks(req_ptr->block_table(), req_ptr->block_table().seq_len + 1);
        }
        batch.decode_requests.push_back(req_ptr.get());
        token_budget--;
    }

    // 2. Admit new prefill requests
    int prefill_budget = std::min(token_budget, config_.max_prefill_tokens);
    std::lock_guard<std::mutex> lock(submit_mutex_);

    while (!waiting_queue_.empty() && prefill_budget > 0) {
        auto* req = waiting_queue_.front().get();

        if (static_cast<int>(active_requests_.size()) + static_cast<int>(batch.prefill_requests.size())
            >= config_.max_batch_requests) {
            break;
        }

        if (!can_admit(*req)) {
            LOGW << "[Scheduler] Cannot admit request " << req->request_id << ": prompt=" << req->input_ids.size()
                 << " tokens" << ", available=" << (block_allocator_ ? block_allocator_->available_blocks() : -1);
            break;
        }

        allocate_blocks_for_request(req);

        // Chunked prefill
        int remaining = static_cast<int>(req->input_ids.size()) - req->prefill_progress;
        int chunk = std::min(remaining, prefill_budget);

        batch.prefill_requests.push_back(req);
        batch.prefill_chunk_starts.push_back(req->prefill_progress);
        batch.prefill_chunk_sizes.push_back(chunk);
        prefill_budget -= chunk;

        req->prefill_progress += chunk;
        req->phase = RequestPhase::PREFILL;

        if (req->prefill_progress >= static_cast<int>(req->input_ids.size())) {
            active_requests_.push_back(std::move(waiting_queue_.front()));
            waiting_queue_.pop_front();
        } else {
            break;
        }
    }

    return batch;
}

void Scheduler::process_results(ScheduledBatch& batch, tensor_t logits, sampler::Sampler& default_sampler,
                                sampler::Sampler& argmax_sampler, sampler::GeneralSampler& general_sampler,
                                tokenizer::Tokenizer& tokenizer, const std::vector<int>& stop_token_ids) {
    int offset = 0;

    // Debug toggle resolved once per process (the env var never changes at
    // runtime). Dumps every committed token id to stderr. Resolving it here
    // keeps getenv() out of the per-token sampling hot path.
    static const bool dump_token_ids = [] {
        const char* env = std::getenv("ZEDINFER_DUMP_TOKEN_IDS");
        return env && env[0] == '1';
    }();

    // Pick a sampler honoring the request's GenerationConfig overrides. Called
    // immediately before each sample() so per-request temperature / top_p /
    // top_k / seed / repetition_penalty land on the GeneralSampler instance.
    // Single-threaded process_results means there is no concurrent setParams.
    //
    // Important: setSeed() is applied at most ONCE per request, gated by
    // req.sampler_seeded. Re-seeding on every token would collapse the RNG
    // into a deterministic single-step sequence (token 2 would resample from
    // the same state as token 1), which breaks the OpenAI seed-per-request
    // semantics ("seed initializes the run, subsequent tokens advance the RNG").
    auto pick_sampler = [&](InferenceRequest& req) -> sampler::Sampler& {
        const auto& gc = req.config;
        if (!gc.has_sampling_override) {
            return default_sampler;
        }
        if (gc.use_argmax || gc.temperature <= 0.0f) {
            return argmax_sampler;
        }
        sampler::SamplerParams p;
        p.temperature        = gc.temperature;
        p.top_k              = gc.top_k;
        p.top_p              = (gc.top_p <= 0.0f) ? 1.0f : gc.top_p;
        p.repetition_penalty = (gc.repetition_penalty <= 0.0f) ? 1.0f : gc.repetition_penalty;
        p.seed               = gc.seed;
        if (p.validate()) {
            general_sampler.setParams(p);
            if (gc.seed != 0 && !req.sampler_seeded) {
                general_sampler.setSeed(gc.seed);
                req.sampler_seeded = true;
            }
        } else {
            LOGW << "[Scheduler] Invalid sampler params (req=" << req.request_id
                 << "), keeping previous params: " << p.info();
        }
        return general_sampler;
    };

    // Inline helper: apply the "force-emit </think>" budget to a freshly
    // sampled token, then update the request's thinking state. Centralized so
    // both prefill (first generated token) and decode (subsequent tokens)
    // paths share the exact same logic.
    //
    // Rules:
    //   - If a post-think newline force-emit is pending (post_think_forced_newlines > 0),
    //     replace the sampled token with "\n\n" so the model sees the trained
    //     "</think>\n\n" pattern instead of continuing its truncated reasoning.
    //   - Else if we are currently in thinking AND the per-request budget is
    //     set AND we've used it up AND the model did NOT itself emit </think>,
    //     replace the sampled token with </think> and schedule a "\n\n"
    //     force-emit on the next step.
    //   - If the token (sampled or forced) is </think>, leave thinking.
    //   - Otherwise, if we are still in thinking, increment the budget counter.
    auto apply_think_budget = [this](InferenceRequest* req, int token) -> int {
        // Highest priority: drain the post-</think> newline budget. Carries the
        // model from "just force-closed thinking" back to a normal sampling
        // state by reproducing the trained </think>\n\n separator pattern.
        if (req->post_think_forced_newlines > 0) {
            if (double_newline_token_id_ >= 0) {
                token = double_newline_token_id_;
            }
            req->post_think_forced_newlines--;
            return token;
        }

        if (!req->in_thinking) {
            return token;
        }
        const int budget = req->config.max_think_tokens;
        if (budget > 0 && req->think_token_count >= budget && token != think_close_token_id_
            && think_close_token_id_ >= 0) {
            LOGI << "[Scheduler] Request " << req->request_id << " force-emit </think> after "
                 << req->think_token_count << " think tokens (budget=" << budget << ")";
            token = think_close_token_id_;
            // Queue the trailing "\n\n" force-emit so the model exits the
            // truncated-thinking attractor. Skip if we don't have a newline id.
            if (double_newline_token_id_ >= 0) {
                req->post_think_forced_newlines = 1;
            }
        }
        if (token == think_close_token_id_) {
            req->in_thinking = false;
        } else {
            req->think_token_count++;
        }
        return token;
    };

    // Process decode results
    for (auto* req : batch.decode_requests) {
        // Check cancellation (client disconnected)
        if (req->cancelled && req->cancelled->load()) {
            LOGI << "[Scheduler] Request " << req->request_id << " cancelled";
            complete_request(*req);
            // Skip the row(s) this req contributed.
            offset += (req->mtp_pending_draft >= 0) ? 2 : 1;
            continue;
        }

        // Stage D.1 spec-decode verify path: if a draft is pending, the
        // forward was 2-token [last_token, draft] so we have logits at
        // offset+0 (main's guess given last_token) and offset+1 (main's
        // guess given draft, only valid if draft is accepted).
        const bool is_spec = (req->mtp_pending_draft >= 0);
        const int  draft   = req->mtp_pending_draft;
        // Always clear the pending slot for the next iteration; serving_loop
        // will refill it with a new draft after MTP runs.
        req->mtp_pending_draft = -1;

        if (is_spec) {
            // logits[0] -> main's distribution p for the draft's position; an
            // accept also unlocks the t+2 bonus token from logits[1].
            //
            // Sampling mode (GeneralSampler + the MTP draft distribution q carried
            // from the draft step): TRUE rejection sampling — accept the draft d
            // with probability min(1, p(d)/q(d)), else resample the corrected
            // token from the residual normalize(max(0, p-q)). The committed token
            // is distributed exactly as the target p. Greedy/argmax mode: accept
            // iff main's argmax equals the draft (exact top-1 match).
            auto row0 = logits->slice(0, offset,     offset + 1);
            sampler::Sampler& spec_sampler = pick_sampler(*req);
            int  t0       = -1;
            bool accepted = false;
            if (auto* gs = dynamic_cast<sampler::GeneralSampler*>(&spec_sampler);
                gs != nullptr && !req->mtp_draft_q.empty()) {
                auto p_dist = gs->truncatedDist(row0, &req->output_ids);
                t0          = gs->specRejectionSample(p_dist, req->mtp_draft_q, draft, accepted);
            } else {
                t0       = spec_sampler.sample(row0, &req->output_ids);
                accepted = (t0 == draft);
            }
            req->mtp_draft_q.clear();
            t0 = apply_think_budget(req, t0);

            if (accepted) {
                // ACCEPT: commit draft + sample next from logits[1].
                auto row1 = logits->slice(0, offset + 1, offset + 2);
                int  t1   = spec_sampler.sample(row1, &req->output_ids);
                // think-budget on the second emitted token too.
                t1 = apply_think_budget(req, t1);

                if (dump_token_ids) {
                    fprintf(stderr, "[zedinfer-tok] step=%d id=%d (spec-accept)\n",
                            req->generated_count, t0);
                    fprintf(stderr, "[zedinfer-tok] step=%d id=%d (spec-accept+1)\n",
                            req->generated_count + 1, t1);
                }

                req->output_ids.push_back(t0);
                req->output_ids.push_back(t1);
                req->last_token = t1;
                req->generated_count   += 2;
                req->block_table().seq_len += 2;
                req->mtp_accept_count  += 1;
                req->mtp_last_n_committed = 2;
                // Draft accepted: forward_linear_attn_layer left the request's
                // real SSM slot at the post-last_token state and put the post-
                // draft state in the pool's temp slot. Promote temp->real so the
                // next step continues from the correct post-draft state.
                if (ssm_state_pool_ != nullptr && req->ssm_slot_idx() >= 0) {
                    ssm_state_pool_->copy_slot_state(req->ssm_slot_idx(), ssm_state_pool_->spec_temp_slot());
                }

                if (is_stop_token(t0, stop_token_ids) || is_stop_token(t1, stop_token_ids)) {
                    req->finish_reason = "stop";
                    complete_request(*req);
                } else if (req->generated_count >= req->config.max_new_tokens) {
                    req->finish_reason = "length";
                    complete_request(*req);
                } else if (req->config.stream && req->stream_callback) {
                    req->stream_callback(tokenizer.decode({t0, t1}));
                }
            } else {
                // REJECT: emit only the corrected token. The draft's recurrent
                // (GatedDeltaNet + causal-conv) update went to the pool's temp
                // slot, so the request's real SSM slot already holds the correct
                // post-last_token state — nothing to roll back. The draft's KV
                // slot at position N+1 is dirty, but the next decode step is a
                // 1-token forward at N+1 that overwrites it before any attention
                // read reaches it.
                if (dump_token_ids) {
                    fprintf(stderr, "[zedinfer-tok] step=%d id=%d (spec-reject draft=%d)\n",
                            req->generated_count, t0, draft);
                }

                req->output_ids.push_back(t0);
                req->last_token = t0;
                req->generated_count++;
                req->block_table().seq_len += 1;
                req->mtp_reject_count += 1;
                req->mtp_last_n_committed = 1;

                if (is_stop_token(t0, stop_token_ids)) {
                    req->finish_reason = "stop";
                    complete_request(*req);
                } else if (req->generated_count >= req->config.max_new_tokens) {
                    req->finish_reason = "length";
                    complete_request(*req);
                } else if (req->config.stream && req->stream_callback) {
                    req->stream_callback(tokenizer.decode({t0}));
                }
            }
            offset += 2;
        } else {
            // Normal 1-token decode (no draft pending OR spec disabled).
            auto req_logits = logits->slice(0, offset, offset + 1);
            int  token = pick_sampler(*req).sample(req_logits, &req->output_ids);
            token = apply_think_budget(req, token);

            if (dump_token_ids) {
                fprintf(stderr, "[zedinfer-tok] step=%d id=%d\n", req->generated_count, token);
            }

            req->output_ids.push_back(token);
            req->last_token = token;
            req->generated_count++;
            req->block_table().seq_len++;
            req->mtp_last_n_committed = 1;

            if (is_stop_token(token, stop_token_ids)) {
                req->finish_reason = "stop";
                complete_request(*req);
            } else if (req->generated_count >= req->config.max_new_tokens) {
                req->finish_reason = "length";
                complete_request(*req);
            } else if (req->config.stream && req->stream_callback) {
                req->stream_callback(tokenizer.decode({token}));
            }
            offset++;
        }
    }

    // Process prefill results
    for (size_t i = 0; i < batch.prefill_requests.size(); ++i) {
        auto* req = batch.prefill_requests[i];
        int chunk = batch.prefill_chunk_sizes[i];

        req->block_table().seq_len += chunk;

        if (req->phase == RequestPhase::DECODE || req->prefill_progress >= static_cast<int>(req->input_ids.size())) {
            req->phase = RequestPhase::DECODE;

            // Initialize per-request thinking state from the prompt. For Qwen3.5
            // the open-think generation_prompt ends with <think>, so input_ids
            // will contain one more <think> than </think> and we enter decode
            // already inside an unclosed thinking block. For closed-think /
            // non-reasoning models, opens == closes and in_thinking stays false.
            if (think_open_token_id_ >= 0 && think_close_token_id_ >= 0) {
                int opens = count_token(req->input_ids, think_open_token_id_);
                int closes = count_token(req->input_ids, think_close_token_id_);
                req->in_thinking = opens > closes;
                req->think_token_count = 0;
            }

            // Insert full blocks into prefix cache after prefill completes
            if (prefix_cache_ && block_allocator_) {
                prefix_cache_->insert_blocks(req->input_ids, block_allocator_->block_size(), req->block_table());
            }
            // Multimodal prefill consumed the pre-baked input_embeds in this
            // step; subsequent decode steps must NOT re-use the [N_total, H]
            // input_embeds buffer (forward would see size mismatch against the
            // N=1 decode batch). Clear it now that prefill is fully done.
            if (req->has_input_embeds()) {
                req->set_input_embeds(nullptr);
            }
            // Hybrid model: snapshot the SSM/conv state at the prefill boundary
            // so a future request with the same exact prompt can restore it
            // and skip prefill. The sample() call that immediately follows
            // does not advance SSM state (it only reads logits), so capturing
            // here is equivalent to capturing right after the last prefill
            // chunk finished. We only record on the path where prefill
            // actually ran end-to-end (`req->prefill_progress` advanced to
            // input_ids.size() through scheduler chunks); on a full prefix-
            // cache hit the SSM state is already a restored snapshot and
            // re-recording would just rewrite identical bytes.
            if (ssm_snapshot_cache_ != nullptr && ssm_state_pool_ != nullptr
                && req->ssm_slot_idx() >= 0) {
                ssm_snapshot_cache_->record(req->input_ids, req->ssm_slot_idx());
            }

            auto req_logits = logits->slice(0, offset, offset + chunk);
            // First sampled token of a request: no generated history yet.
            int token = pick_sampler(*req).sample(req_logits, &req->output_ids);
            token = apply_think_budget(req, token);

            if (dump_token_ids) {
                fprintf(stderr, "[zedinfer-tok] step=0 id=%d (prefill)\n", token);
            }

            req->output_ids.push_back(token);
            req->last_token = token;
            req->generated_count = 1;
            req->stats.prompt_tokens = req->input_ids.size();

            if (is_stop_token(token, stop_token_ids)) {
                req->finish_reason = "stop";
                complete_request(*req);
            } else if (req->generated_count >= req->config.max_new_tokens) {
                // max_new_tokens == 1 edge case: the first prefill token is
                // already the last allowed token, finish with "length".
                req->finish_reason = "length";
                complete_request(*req);
            } else if (req->config.stream && req->stream_callback) {
                req->stream_callback(tokenizer.decode({token}));
            }
        }
        offset += chunk;
    }

    // Remove completed requests. Lock submit_mutex_ for the structural mutation
    // so the HTTP-thread status queries (active_count/has_work) never read the
    // vector mid-erase.
    {
        std::lock_guard<std::mutex> lock(submit_mutex_);
        active_requests_.erase(std::remove_if(active_requests_.begin(), active_requests_.end(),
                                              [](const auto& ptr) { return ptr->phase == RequestPhase::COMPLETE; }),
                               active_requests_.end());
    }
}

void Scheduler::complete_request(InferenceRequest& req) {
    req.phase = RequestPhase::COMPLETE;

    // Release blocks for owned tables (not borrowed session tables).
    // Uses release (not free) so cached prefix blocks stay in the pool.
    if (block_allocator_ && req.owns_block_table() && req.block_table().num_layers > 0) {
        block_allocator_->release_sequence(req.block_table());
    }

    // Release SSM slot for hybrid models. Safe even if no pool / no slot was
    // ever acquired (ssm_slot_idx() == -1 short-circuits).
    if (ssm_state_pool_ != nullptr && req.ssm_slot_idx() >= 0) {
        ssm_state_pool_->release_slot(req.ssm_slot_idx());
        req.set_ssm_slot_idx(-1);
    }

    GenerationResult result;
    result.output_ids = std::move(req.output_ids);
    result.stats = req.stats;
    result.stats.generated_tokens = result.output_ids.size();
    result.stats.total_tokens = result.stats.prompt_tokens + result.stats.generated_tokens;
    result.finish_reason = req.finish_reason;

    try {
        req.result_promise.set_value(std::move(result));
    } catch (...) {}
}

} // namespace zedinfer
