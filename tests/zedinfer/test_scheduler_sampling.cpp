#include "backend/kvcache/block_pool.hpp"
#include "backend/kvcache/prefix_cache.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/base.hpp"
#include "zedinfer/batch_context.hpp"
#include "zedinfer/scheduler.hpp"

#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace zedinfer;

namespace {

class FakeTokenizer : public tokenizer::Tokenizer {
public:
    std::vector<int> encode(const std::string&) override { return {}; }
    std::string decode(const std::vector<int>& tokens) override {
        return tokens.empty() ? std::string() : std::to_string(tokens.front());
    }
    int get_vocab_size() const override { return 3; }
    const tokenizer::Config& get_config() const override { return config_; }
    void load_from_file(const std::string&) override {}

private:
    tokenizer::Config config_;
};

tensor_t make_logits_2d(const std::vector<float>& data, size_t rows, size_t cols) {
    auto t = Tensor::create({rows, cols}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU, 0);
    t->load(data.data());
    return t;
}

kvcache::SequenceBlockTable make_block_table() {
    kvcache::SequenceBlockTable table;
    table.num_layers = 0;
    table.seq_len = 0;
    return table;
}

sampler::ArgmaxSampler make_argmax_sampler() {
    ExecutorConfig cfg;
    cfg.device_type = ZEDINFER_DEVICE_CPU;
    cfg.device_id = 0;
    cfg.data_type = ZEDINFER_DTYPE_F32;
    return sampler::ArgmaxSampler(cfg);
}

sampler::GeneralSampler make_general_sampler() {
    sampler::SamplerParams params;
    return sampler::GeneralSampler(params);
}

} // namespace

TEST(SchedulerSamplingTest, MixedDecodePrefillSamplesEachRequestLogitRow) {
    Scheduler scheduler;
    FakeTokenizer tokenizer;
    auto default_sampler = make_argmax_sampler();
    auto argmax_sampler = make_argmax_sampler();
    auto general_sampler = make_general_sampler();

    InferenceRequest decode_req;
    decode_req.config.max_new_tokens = 8;
    decode_req.phase = RequestPhase::DECODE;
    decode_req.own_block_table(make_block_table());

    InferenceRequest prefill_req;
    prefill_req.config.max_new_tokens = 8;
    prefill_req.prefill_progress = 2;
    prefill_req.input_ids = {10, 11};
    prefill_req.own_block_table(make_block_table());

    ScheduledBatch batch;
    batch.decode_requests.push_back(&decode_req);
    batch.prefill_requests.push_back(&prefill_req);
    batch.prefill_chunk_starts.push_back(0);
    batch.prefill_chunk_sizes.push_back(2);

    auto logits = make_logits_2d(
        {
            9.0f, 0.0f, 0.0f, // decode row: token 0
            0.0f, 8.0f, 0.0f, // prefill first token row
            0.0f, 0.0f, 7.0f, // prefill final token row: token 2
        },
        3, 3);

    scheduler.process_results(batch, logits, default_sampler, argmax_sampler, general_sampler, tokenizer, {});

    ASSERT_EQ(decode_req.output_ids.size(), 1u);
    ASSERT_EQ(prefill_req.output_ids.size(), 1u);
    EXPECT_EQ(decode_req.output_ids[0], 0);
    EXPECT_EQ(prefill_req.output_ids[0], 2);
}

TEST(SchedulerSamplingTest, OversizedQueuedRequestFailsInsteadOfSpinning) {
    SchedulerConfig config;
    config.max_batch_tokens = 16;
    config.max_prefill_tokens = 16;
    Scheduler scheduler(config);

    kvcache::BlockConfig block_config;
    block_config.block_size = 4;
    block_config.num_kv_heads = 1;
    block_config.head_dim = 1;
    block_config.dtype = ZEDINFER_DTYPE_F32;
    kvcache::BlockPool pool(block_config, 1, ZEDINFER_DEVICE_CPU, 0);
    kvcache::BlockAllocator allocator(pool, 1);
    scheduler.set_block_allocator(&allocator);

    auto req = std::make_unique<InferenceRequest>();
    req->input_ids.assign(32, 1);
    req->config.max_new_tokens = 1;
    auto future = req->result_promise.get_future();
    scheduler.submit(std::move(req));

    ScheduledBatch batch = scheduler.schedule();

    EXPECT_TRUE(batch.empty());
    EXPECT_EQ(scheduler.pending_count(), 0);
    ASSERT_EQ(future.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST(SchedulerSamplingTest, ActiveDecodeKvExhaustionFailsRequestInsteadOfSpinning) {
    SchedulerConfig config;
    config.max_batch_tokens = 16;
    config.max_prefill_tokens = 16;
    Scheduler scheduler(config);

    kvcache::BlockConfig block_config;
    block_config.block_size = 1;
    block_config.num_kv_heads = 1;
    block_config.head_dim = 1;
    block_config.dtype = ZEDINFER_DTYPE_F32;
    kvcache::BlockPool pool(block_config, 257, ZEDINFER_DEVICE_CPU, 0);
    kvcache::BlockAllocator allocator(pool, 1);
    scheduler.set_block_allocator(&allocator);

    auto req = std::make_unique<InferenceRequest>();
    req->input_ids = {1};
    req->config.max_new_tokens = 300;
    auto future = req->result_promise.get_future();
    scheduler.submit(std::move(req));

    ScheduledBatch prefill = scheduler.schedule();
    ASSERT_EQ(prefill.prefill_requests.size(), 1u);
    ASSERT_EQ(scheduler.active_count(), 1);

    auto* active = prefill.prefill_requests[0];
    active->phase = RequestPhase::DECODE;
    active->last_token = 1;
    active->block_table().seq_len = 257;

    ScheduledBatch decode;
    EXPECT_NO_THROW(decode = scheduler.schedule());

    EXPECT_TRUE(decode.empty());
    EXPECT_EQ(scheduler.active_count(), 0);
    ASSERT_EQ(future.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    EXPECT_THROW(future.get(), std::runtime_error);
}

// Hybrid (Qwen3.5-family) models run one request per forward and gate admission
// on a free SSMStatePool slot. The single-request, round-robin hybrid scheduler
// must INTERLEAVE concurrent requests: a request submitted while another is
// mid-generation has to start making progress before the first one finishes,
// rather than waiting FIFO behind a single SSM slot (the bug this guards).
TEST(SchedulerSamplingTest, HybridSchedulerInterleavesConcurrentRequests) {
    SchedulerConfig config;
    config.max_batch_tokens = 64;
    config.max_prefill_tokens = 16;
    Scheduler scheduler(config);

    kvcache::BlockConfig block_config;
    block_config.block_size = 4;
    block_config.num_kv_heads = 1;
    block_config.head_dim = 1;
    block_config.dtype = ZEDINFER_DTYPE_F32;
    kvcache::BlockPool pool(block_config, 64, ZEDINFER_DEVICE_CPU, 0);
    kvcache::BlockAllocator allocator(pool, 1);
    scheduler.set_block_allocator(&allocator);

    // CPU-backed SSM pool with 2 slots → up to 2 concurrent hybrid requests.
    model::SSMStatePoolConfig ssm_cfg;
    ssm_cfg.num_linear_layers = 1;
    ssm_cfg.num_v_heads = 1;
    ssm_cfg.value_head_dim = 1;
    ssm_cfg.d_state = 1;
    ssm_cfg.conv_kernel_dim = 2;
    ssm_cfg.qkv_dim = 1;
    ssm_cfg.max_concurrent = 2;
    ssm_cfg.state_dtype = ZEDINFER_DTYPE_F32;
    ExecutorConfig ssm_exec(ZEDINFER_DEVICE_CPU, 0, ZEDINFER_DTYPE_F32);
    model::SSMStatePool ssm_pool(ssm_cfg, ssm_exec);
    scheduler.set_ssm_state_pool(&ssm_pool);

    constexpr int kTokensEach = 4;
    auto submit_req = [&]() {
        auto req = std::make_unique<InferenceRequest>();
        req->input_ids = {1};
        req->config.max_new_tokens = kTokensEach;
        scheduler.submit(std::move(req));
    };
    submit_req(); // request_id 1 (A)
    submit_req(); // request_id 2 (B)

    // Drive schedule(); after each batch apply the minimal state update the real
    // serving loop's process_results would (prefill → DECODE + first token,
    // decode → +1 token, finish at the token budget) so the next schedule sees
    // realistic phases. Record the request id served each step.
    std::vector<uint64_t> order;
    auto apply_and_record = [&](ScheduledBatch& b) {
        for (size_t i = 0; i < b.prefill_requests.size(); ++i) {
            auto* r = b.prefill_requests[i];
            r->block_table().seq_len += b.prefill_chunk_sizes[i];
            r->phase = RequestPhase::DECODE;
            r->generated_count = 1;
            if (r->generated_count >= r->config.max_new_tokens) {
                r->phase = RequestPhase::COMPLETE;
            }
            order.push_back(r->request_id);
        }
        for (auto* r : b.decode_requests) {
            r->block_table().seq_len += 1;
            r->generated_count += 1;
            if (r->generated_count >= r->config.max_new_tokens) {
                r->phase = RequestPhase::COMPLETE;
            }
            order.push_back(r->request_id);
        }
    };

    for (int step = 0; step < 32; ++step) {
        ScheduledBatch b = scheduler.schedule();
        if (b.empty()) {
            if (!scheduler.has_work()) {
                break;
            }
            continue;
        }
        apply_and_record(b);
    }

    // Both requests are fully served (one prefill token + kTokensEach-1 decodes).
    int count_a = 0, count_b = 0;
    for (uint64_t id : order) {
        count_a += (id == 1);
        count_b += (id == 2);
    }
    EXPECT_EQ(count_a, kTokensEach);
    EXPECT_EQ(count_b, kTokensEach);

    // The fix: B (id 2) must be scheduled BEFORE A (id 1) finishes its last
    // token. Under the old single-slot serialization B would not appear until
    // every A unit had run.
    int a_completion_idx = -1, a_seen = 0, first_b_idx = -1;
    for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        if (order[i] == 1 && ++a_seen == kTokensEach) {
            a_completion_idx = i;
            break;
        }
    }
    for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        if (order[i] == 2) {
            first_b_idx = i;
            break;
        }
    }
    ASSERT_GE(a_completion_idx, 0);
    ASSERT_GE(first_b_idx, 0);
    EXPECT_LT(first_b_idx, a_completion_idx) << "request B did not interleave with A (FIFO serialization)";
}

TEST(SchedulerSamplingTest, FullPrefixCacheHitStillSchedulesNonEmptyPrefill) {
    SchedulerConfig config;
    config.max_batch_tokens = 16;
    config.max_prefill_tokens = 16;
    Scheduler scheduler(config);

    kvcache::BlockConfig block_config;
    block_config.block_size = 4;
    block_config.num_kv_heads = 1;
    block_config.head_dim = 1;
    block_config.dtype = ZEDINFER_DTYPE_F32;
    kvcache::BlockPool pool(block_config, 8, ZEDINFER_DEVICE_CPU, 0);
    kvcache::BlockAllocator allocator(pool, 1);
    kvcache::PrefixCache prefix_cache(pool, 1);

    scheduler.set_block_allocator(&allocator);
    scheduler.set_prefix_cache(&prefix_cache);

    std::vector<int> prompt = {1, 2, 3, 4};
    auto cached_table = allocator.allocate_sequence(static_cast<int>(prompt.size()));
    prefix_cache.insert_blocks(prompt, allocator.block_size(), cached_table);
    allocator.release_sequence(cached_table);

    auto req = std::make_unique<InferenceRequest>();
    req->input_ids = prompt;
    req->config.max_new_tokens = 1;
    scheduler.submit(std::move(req));

    ScheduledBatch batch = scheduler.schedule();

    ASSERT_EQ(batch.prefill_requests.size(), 1u);
    ASSERT_EQ(batch.prefill_chunk_sizes.size(), 1u);
    EXPECT_EQ(batch.prefill_chunk_starts[0], 0);
    EXPECT_EQ(batch.prefill_chunk_sizes[0], static_cast<int>(prompt.size()));
    EXPECT_EQ(batch.prefill_requests[0]->prefill_progress, static_cast<int>(prompt.size()));
}
