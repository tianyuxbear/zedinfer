#include "backend/tensor/tensor.hpp"
#include "backend/kvcache/block_pool.hpp"
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
