#include "zedinfer/serving_loop.hpp"
#include "backend/device/runtime_api.hpp"
#include "backend/ops/ops.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/hybrid_transformer_forward.hpp"
#include "frontend/models/mtp_module.hpp"
#include "frontend/models/qwen3_5.hpp"
#include "frontend/models/qwen3_5_moe.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "utils/logging.hpp"
#include "zedinfer/engine.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <plog/Log.h>

namespace zedinfer {

ServingLoop::ServingLoop(InferenceEngine& engine, SchedulerConfig sched_config)
    : engine_(&engine), mtp_enabled_(sched_config.mtp_enabled), scheduler_(std::move(sched_config)) {
    if (engine_->block_allocator()) {
        scheduler_.set_block_allocator(engine_->block_allocator());
    }
    // Wire SSMStatePool for hybrid models (Qwen3.5 / Qwen3.5-MoE). Non-hybrid
    // models return nullptr from ssm_state_pool() and the scheduler keeps the
    // pre-existing single-pool admission semantics.
    auto* ssm_pool = engine_->ssm_state_pool();
    if (ssm_pool) {
        scheduler_.set_ssm_state_pool(ssm_pool);
    }
    // PrefixCache is now safe to wire even for hybrid models: the scheduler
    // pairs it with the SSMSnapshotCache below so that a prefix-cache hit is
    // only honored when the SSM/conv state for the exact prompt is also
    // restorable. Without that pairing, partial-prefix KV reuse would leave
    // the linear-attention layers prefix-blind and the output would diverge
    // from a cold-cache run.
    if (engine_->prefix_cache()) {
        scheduler_.set_prefix_cache(engine_->prefix_cache());
    }
    if (engine_->ssm_snapshot_cache()) {
        scheduler_.set_ssm_snapshot_cache(engine_->ssm_snapshot_cache());
        LOGI << "[ServingLoop] Hybrid model: SSM snapshot cache enabled; "
                "prefix cache reuse limited to exact full-prompt matches.";
    }
    // Pass <think> / </think> ids (Qwen3.5 family) so the scheduler can drive
    // the per-request thinking-budget force-emit. Returns -1 for models without
    // these special tokens, in which case the scheduler treats the budget as
    // disabled. The "\n\n" id (resolved via tokenizer.encode) anchors the
    // post-</think> state after a force-close.
    scheduler_.set_think_token_ids(engine_->think_open_token_id(), engine_->think_close_token_id(),
                                   engine_->double_newline_token_id());
}

// ============================================================================
// Synchronous Generation
// ============================================================================

std::string ServingLoop::generate(kvcache::SequenceBlockTable& block_table, const std::string& prompt,
                                  const GenerationConfig& config) {
    config.validate();

    if (config.verbose) {
        if (config.gen_mode == GenerationMode::PING) {
            LOG_INFO_(utils::BOTH) << "\n=== [Inference] Prompt: ===\n" << prompt;
        } else {
            LOGI << "\n=== [Inference] Prompt: ===\n" << prompt;
        }
        LOGI << "[Inference] Encoding prompt...";
    }

    auto input_ids = engine_->tokenizer().encode(prompt);

    if (config.verbose) {
        LOGI << "[Inference] Prompt tokens: " << input_ids.size();
        LOGI << "[Inference] Generating...";
    }

    auto result = generate_tokens(block_table, input_ids, config);
    std::string output = engine_->tokenizer().decode(result.output_ids);

    if (config.verbose) {
        LOGI << "[Inference] Generation complete";
        if (config.gen_mode == GenerationMode::PING) {
            LOG_INFO_(utils::BOTH) << "\n=== [Inference] Generated ===\n" << output;
        } else {
            LOGI << "\n=== [Inference] Generated ===\n" << output;
        }
    }

    if (config.print_stats) {
        LOGI << result.stats.summary();
    }

    return output;
}

GenerationResult ServingLoop::generate_tokens(kvcache::SequenceBlockTable& block_table,
                                              const std::vector<int>& input_ids, const GenerationConfig& config) {
    // Build request borrowing the session's block table
    auto request = build_request(input_ids, config);
    request->borrow_block_table(block_table);
    auto future = submit_async(std::move(request));

    while (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) { step(); }

    return future.get();
}

std::unique_ptr<InferenceRequest> ServingLoop::build_request(const std::vector<int>& input_ids,
                                                             const GenerationConfig& config) {
    auto req = std::make_unique<InferenceRequest>();
    req->input_ids = input_ids;
    req->config = config;
    req->stream_callback = config.stream ? config.stream_callback : nullptr;
    req->arrival_time = std::chrono::steady_clock::now();
    return req;
}

// ============================================================================
// Batch Mode
// ============================================================================

std::future<GenerationResult> ServingLoop::submit_async(std::unique_ptr<InferenceRequest> request) {
    auto future = request->result_promise.get_future();
    // Scheduler::submit() notifies the serving thread's wait_for_work().
    scheduler_.submit(std::move(request));
    return future;
}

bool ServingLoop::step() {
    auto batch = scheduler_.schedule();
    if (batch.empty()) {
        return false;
    }

    auto batch_ctx = batch.build_context();

    try {
        auto t0 = std::chrono::high_resolution_clock::now();

        model::PagedForwardContext ctx(batch_ctx, *engine_->block_allocator());
        // Use decode scratch for single-token decode (no prefill in batch)
        bool is_pure_decode = batch.prefill_requests.empty() && batch.decode_requests.size() == 1;
        auto* scratch = is_pure_decode ? engine_->decode_scratch() : nullptr;

        // Hybrid Qwen3.5 dispatch: SSU + paged-attn mixed path needs the
        // request's SSM slot index, image embeds, and pos_ids_thw. Single-user
        // M1 smoke assumes one request per batch — pick whichever phase is
        // present. Multi-request batched hybrid is M5 territory.
        tensor_t logits;
        tensor_t mtp_hidden_last;  // captured iff MTP active (see below)
        const model::Qwen3_5Model* mtp_owner = nullptr;
        // Three independent ways to enable MTP, in priority order:
        //   1. --mtp CLI flag (mtp_enabled_, the user-facing knob, default off).
        //   2. ZEDINFER_MTP_SPEC env var (research/legacy active spec mode).
        //   3. ZEDINFER_MTP_DEBUG env var (observer-only: log drafts, no spec).
        // mtp_spec is the "active" path (drafts get committed via 2-token verify);
        // mtp_debug just logs. The user-visible default is off — env vars exist
        // so we don't have to touch the binary's CLI surface to A/B test.
        // Resolved once per process (env vars are immutable at runtime), so the
        // per-step path pays no getenv cost.
        static const bool mtp_spec_env  = std::getenv("ZEDINFER_MTP_SPEC")  != nullptr;
        static const bool mtp_debug_env = std::getenv("ZEDINFER_MTP_DEBUG") != nullptr;
        const bool mtp_spec             = mtp_enabled_ || mtp_spec_env;
        if (auto* hybrid_model = dynamic_cast<const model::Qwen3_5Model*>(&engine_->model())) {
            InferenceRequest* req = !batch.decode_requests.empty()
                                       ? batch.decode_requests[0]
                                       : (!batch.prefill_requests.empty() ? batch.prefill_requests[0] : nullptr);
            if (!req) {
                throw std::runtime_error("[ServingLoop] hybrid model: empty batch (no request to forward)");
            }
            // MoE variant carries an ExpertPool + shared-expert config the dense
            // forward config doesn't populate; downcast first so the right
            // is_moe / num_experts / expert_pool fields are set.
            const auto* moe_model = dynamic_cast<const model::Qwen3_5MoeModel*>(hybrid_model);
            model::HybridForwardConfig hcfg = moe_model ? moe_model->hybrid_forward_config_moe()
                                                          : hybrid_model->hybrid_forward_config();
            // Stage D.1 speculative decoding gate. ZEDINFER_MTP_SPEC=1 turns
            // on the full proposal+verify path: main captures hidden_last,
            // MTPModule runs after each step to set req.mtp_pending_draft,
            // and the NEXT scheduled step submits 2 tokens [last, draft] so
            // main can verify in a single forward. ZEDINFER_MTP_DEBUG keeps
            // working as a no-op observer (just logs draft vs main argmax).
            // MTP head lives on the base Qwen3_5Model now, so dense (27B) and
            // MoE (35B-A3B) are both supported through the same accessor.
            const bool mtp_active    = (mtp_spec || mtp_debug_env)
                                       && hybrid_model->mtp_module()
                                       && hybrid_model->mtp_module()->ready();
            tensor_t* hidden_out_ptr = nullptr;
            if (mtp_active) {
                hidden_out_ptr = &mtp_hidden_last;
                mtp_owner      = hybrid_model;
                if (!batch.prefill_requests.empty()) {
                    // Lazily allocate + reset MTP K/V state on this request.
                    // Stage D.0: state is per-request now, not module-global.
                    const auto& mcfg = hybrid_model->config();
                    const size_t Hkv = mcfg.num_key_value_heads;
                    const size_t Dh  = mcfg.head_dim > 0 ? mcfg.head_dim
                                                         : (mcfg.hidden_size / mcfg.num_attention_heads);
                    model::mtp_reset_request_state(*batch.prefill_requests[0],
                                                   hybrid_model->mtp_module()->max_kv_len(),
                                                   Hkv, Dh,
                                                   engine_->exec_config());
                } else if (req->mtp_pending_draft >= 0 && req->ssm_slot_idx() >= 0) {
                    // Spec verify step: tell forward_linear_attn_layer to route
                    // the draft token's recurrent (GDN + conv) update to the
                    // pool's temp slot so the request's real SSM slot keeps the
                    // post-last_token state. On accept the scheduler promotes
                    // temp->real; on reject the real slot is already correct so
                    // it just emits the corrected token — no rollback, no redo.
                    req->mtp_spec_verify_active = true;
                }
            }
            logits = model::hybrid_transformer_forward(hcfg, ctx, *req, engine_->exec_config(), scratch,
                                                         req->input_embeds(), hidden_out_ptr);
            req->mtp_spec_verify_active = false;
        } else {
            logits = model::transformer_forward(engine_->model().forward_config(), ctx, engine_->exec_config(),
                                                  scratch);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Track timing: prefill or decode based on batch composition
        bool has_prefill = !batch.prefill_requests.empty();
        bool has_decode = !batch.decode_requests.empty();

        if (has_prefill) {
            for (auto* req : batch.prefill_requests) {
                req->stats.prefill_time_ms += step_ms;
                req->stats.total_time_ms += step_ms;
            }
        }
        if (has_decode) {
            for (auto* req : batch.decode_requests) {
                req->stats.decode_time_ms += step_ms / batch.decode_requests.size();
                req->stats.total_time_ms += step_ms / batch.decode_requests.size();
            }
        }

        scheduler_.process_results(batch, logits, engine_->sampler(), engine_->argmax_sampler(),
                                   engine_->general_sampler(), engine_->tokenizer(), engine_->stop_token_ids());

        // Stage D.1 MTP integration: after scheduler picks tokens for this
        // step, advance MTP's K/V cache and produce the next draft. Gated
        // on mtp_active (ZEDINFER_MTP_SPEC or _DEBUG + a ready MTP head).
        //
        // Three cases:
        //   - Prefill (P-token hidden, just-sampled t_P): run MTPModule::prefill
        //     across positions 0..P-1. The final logit gives the next draft.
        //   - Decode with mtp_last_n_committed=1: 1 main step, 1 MTP forward.
        //   - Decode with mtp_last_n_committed=2 (spec ACCEPT): 2 MTP forwards
        //     covering the two committed positions; last logit -> next draft.
        // Spec REJECT also yields n_committed=1, but the second row of
        // mtp_hidden_last was computed against a poisoned K/V slot and must
        // not be fed to MTP — we just use row 0 (clean).
        const bool mtp_active = (mtp_spec || mtp_debug_env) && mtp_owner;
        if (mtp_active && mtp_hidden_last) {
            try {
                InferenceRequest* req = !batch.decode_requests.empty()
                                           ? batch.decode_requests[0]
                                           : (!batch.prefill_requests.empty() ? batch.prefill_requests[0]
                                                                              : nullptr);
                if (req && req->phase != RequestPhase::COMPLETE) {
                    const auto& w = mtp_owner->weights();
                    auto embed_w  = w.has_tensor("embed_tokens.weight")
                                        ? w.get_tensor("embed_tokens.weight") : nullptr;
                    auto lmhead_w = w.has_tensor("lm_head.weight")
                                        ? w.get_tensor("lm_head.weight") : nullptr;
                    if (embed_w && lmhead_w) {
                        const size_t N = mtp_hidden_last->shape()[0];
                        tensor_t mtp_logits;
                        if (!batch.prefill_requests.empty()) {
                            // Prefill: cover all P prompt positions through MTP.
                            std::vector<int> next_tokens;
                            next_tokens.reserve(N);
                            for (size_t i = 1; i < req->input_ids.size() && next_tokens.size() < N - 1; ++i) {
                                next_tokens.push_back(req->input_ids[i]);
                            }
                            next_tokens.push_back(req->last_token);
                            mtp_logits = mtp_owner->mtp_module()->prefill(
                                *req, mtp_hidden_last, next_tokens, embed_w, lmhead_w,
                                engine_->exec_config());
                        } else {
                            // Decode: advance MTP by mtp_last_n_committed positions.
                            // For each committed slot k:
                            //   hidden_row_k = mtp_hidden_last[k]   (row 0 always clean,
                            //                                        row 1 only used on
                            //                                        spec accept)
                            //   token_k      = the k-th most-recently-emitted token,
                            //                  i.e. output_ids[end - n_committed + k]
                            const int n = std::max(1, req->mtp_last_n_committed);
                            const size_t end = req->output_ids.size();
                            // The n committed tokens were just appended to output_ids, so
                            // end >= n always holds here. Guard it explicitly: the unsigned
                            // (end - n) indexing below would wrap catastrophically if a future
                            // change ever broke that invariant.
                            if (end < static_cast<size_t>(n)) {
                                throw std::runtime_error("MTP decode: output_ids shorter than committed count");
                            }
                            for (int k = 0; k < n; ++k) {
                                auto hrow = mtp_hidden_last->slice(0, k, k + 1);
                                int  tok  = req->output_ids[end - n + k];
                                mtp_logits = mtp_owner->mtp_module()->forward(
                                    *req, hrow, tok, embed_w, lmhead_w,
                                    engine_->exec_config());
                            }
                        }

                        // Top-1 via ops::argmax; store as next draft so the
                        // NEXT scheduled step builds a 2-token verify batch.
                        const size_t vocab = mtp_logits->shape()[1];
                        auto last_view = mtp_logits->view({vocab});
                        auto idx_dev = Tensor::create({1}, ZEDINFER_DTYPE_I64,
                                                      engine_->exec_config().device_type,
                                                      engine_->exec_config().device_id);
                        auto val_dev = Tensor::create({1}, engine_->exec_config().data_type,
                                                      engine_->exec_config().device_type,
                                                      engine_->exec_config().device_id);
                        ops::argmax(idx_dev, val_dev, last_view);
                        int64_t mtp_top1 = -1;
                        auto* api = device::getRuntimeAPI(engine_->exec_config().device_type);
                        api->memcpy_sync(&mtp_top1, idx_dev->data(), sizeof(int64_t),
                                         ZEDINFER_MEMCPY_D2H);
                        if (mtp_spec) {
                            // Stage D.2: in sampling mode draw the draft from the
                            // MTP head's truncated distribution q (not argmax) and
                            // carry q so the verify step runs true rejection
                            // sampling. Greedy/argmax mode keeps the MTP top-1 as
                            // the draft (exact top-1 match acceptance).
                            if (auto* gs = dynamic_cast<sampler::GeneralSampler*>(&engine_->sampler())) {
                                auto q = gs->truncatedDist(last_view, &req->output_ids);
                                req->mtp_pending_draft = gs->sampleFromDist(q);
                                req->mtp_draft_q       = std::move(q);
                            } else {
                                req->mtp_pending_draft = static_cast<int>(mtp_top1);
                                req->mtp_draft_q.clear();
                            }
                        }
                        if (mtp_debug_env) {
                            fprintf(stderr, "[MTP-debug] main_token=%d  mtp_top1=%lld\n",
                                    req->last_token, (long long)mtp_top1);
                        }
                    }
                }
                if (req) {
                    // Consume the per-step flag so a future iteration doesn't
                    // accidentally pick a stale n.
                    req->mtp_last_n_committed = 0;
                }
            } catch (const std::exception& e) {
                LOGW << "[ServingLoop] MTP branch raised: " << e.what();
            }
        }
    } catch (const std::exception& e) {
        LOGE << "[ServingLoop] Forward pass failed: " << e.what();
        // Fail all requests in this batch gracefully instead of crashing
        fail_batch(batch, std::string("Forward pass error: ") + e.what());
    } catch (...) {
        LOGE << "[ServingLoop] Forward pass failed with unknown error";
        fail_batch(batch, "Unknown forward pass error");
    }

    return true;
}

void ServingLoop::run_loop() {
    while (scheduler_.has_work()) {
        try {
            step();
        } catch (const std::exception& e) { LOGE << "[ServingLoop] Unexpected error in run_loop: " << e.what(); }
    }
}

void ServingLoop::run_serving() {
    running_ = true;
    LOGI << "[ServingLoop] Started";

    while (running_) {
        // Process all pending work
        while (scheduler_.has_work() && running_) {
            try {
                step();
            } catch (const std::exception& e) { LOGE << "[ServingLoop] Unexpected error in run_serving: " << e.what(); }
        }

        // No work — wait for new submissions or stop signal. The wait lives in
        // the Scheduler so the wakeup predicate shares submit_mutex_ with
        // submit(); no lost wakeups, no data race on the queue.
        if (running_) {
            scheduler_.wait_for_work(running_);
        }
    }

    // Drain remaining work before exit
    while (scheduler_.has_work()) {
        try {
            step();
        } catch (...) {}
    }

    LOGI << "[ServingLoop] Stopped";
}

void ServingLoop::stop() {
    running_ = false;
    scheduler_.wake_waiters();
}

void ServingLoop::fail_batch(ScheduledBatch& batch, const std::string& error_msg) {
    // Fail all requests in the batch by setting an exception on their promises.
    // This ensures HTTP handlers get an error instead of hanging forever.
    auto fail_request = [&](InferenceRequest* req) {
        if (!req) {
            return;
        }
        req->phase = RequestPhase::COMPLETE;
        // Free owned blocks
        if (engine_->block_allocator() && req->owns_block_table() && req->block_table().num_layers > 0) {
            engine_->block_allocator()->free_sequence(req->block_table());
        }
        try {
            req->result_promise.set_exception(std::make_exception_ptr(std::runtime_error(error_msg)));
        } catch (...) {}
    };

    for (auto* req : batch.decode_requests) { fail_request(req); }
    for (auto* req : batch.prefill_requests) { fail_request(req); }

    // Remove completed requests from scheduler's active list
    // (scheduler expects process_results to clean up, but we're bypassing it)
    scheduler_.cleanup_failed_requests();
}

} // namespace zedinfer
