#pragma once

#include "backend/device/device.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "backend/kvcache/prefix_cache.hpp"
#include "backend/kvcache/ssm_snapshot_cache.hpp"
#include "frontend/models/base.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/base.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/chat_template.hpp"
#include "zedinfer/generation_types.hpp"
#include "zedinfer/profiler.hpp"
#include "zedinfer/request.hpp"
#include "zedinfer/serving_loop.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace zedinfer {

class InferenceSession;
class ChatTemplateJinja;
class MultiModalProcessor;

struct PerplexityEvalConfig {
    size_t context_window = 0;
    size_t max_length = 0;
    bool verbose = false;
};

struct PerplexitySampleStats {
    size_t input_tokens = 0;
    size_t eval_tokens = 0;
    double nll_sum = 0.0;
    double avg_nll = 0.0;
    double ppl = 0.0;

    bool skipped = false;
    std::string skip_reason;
};

struct PerplexityStats {
    uint64_t run_id = 0;
    size_t context_window = 0;
    size_t total_samples = 0;
    size_t evaluated_samples = 0;
    size_t skipped_samples = 0;
    size_t total_input_tokens = 0;
    size_t total_eval_tokens = 0;
    size_t total_chunks = 0;

    double nll_sum = 0.0;
    double avg_nll = 0.0;
    double avg_nll_se = 0.0;
    double ppl = 0.0;
    double ppl_se = 0.0;
    double elapsed_ms = 0.0;

    std::vector<PerplexitySampleStats> samples;
};

/**
 * Inference engine: resource container and factory.
 * Owns model, tokenizer, sampler, block pool.
 * Delegates serving to ServingLoop and profiling to Profiler.
 */
class InferenceEngine : public std::enable_shared_from_this<InferenceEngine> {
public:
    static std::shared_ptr<InferenceEngine> create(const std::string& model_path, device::Device device,
                                                   SchedulerConfig sched_config = {});

    std::unique_ptr<InferenceSession> create_session(const GenerationConfig& gen_config);

    PerplexityStats evaluate_perplexity(const std::vector<std::string>& samples,
                                        const PerplexityEvalConfig& config = PerplexityEvalConfig());

    // Resource accessors (used by ServingLoop, Profiler, Session)
    model::Model& model() { return *model_; }
    tokenizer::Tokenizer& tokenizer() { return *tokenizer_; }
    sampler::Sampler& sampler() { return *sampler_; }
    // Always-available argmax + general samplers, used when the scheduler needs
    // to honor per-request sampling overrides (OpenAI temperature/top_p/top_k).
    // Distinct from sampler() above, which returns whichever was selected at
    // load time from generation_config.json.
    sampler::Sampler& argmax_sampler() { return *argmax_sampler_; }
    sampler::GeneralSampler& general_sampler() { return *general_sampler_; }
    const ExecutorConfig& exec_config() const { return exec_config_; }
    const ChatTemplate& chat_template() const { return chat_template_; }
    // Optional Jinja chat template loaded once at engine init. Nullable:
    // returns nullptr only for models that ship neither a chat_template.jinja
    // file nor a "chat_template" string in tokenizer_config.json. For all
    // mainstream HuggingFace models (Qwen2/Qwen3/Qwen3.5, DeepSeek-R1, Llama,
    // Mistral, etc.) this is non-null and is the canonical prompt formatter.
    // The legacy chat_template() above remains as a last-resort fallback.
    const ChatTemplateJinja* chat_template_jinja() const { return chat_template_jinja_.get(); }
    const std::vector<int>& stop_token_ids() const { return stop_token_ids_; }

    // Token ids for the reasoning <think>/</think> markers used by Qwen3.5.
    // Engine resolves these from the tokenizer at init; return -1 when the
    // model does not have these as special tokens (in which case the
    // scheduler's force-emit-</think> path is a no-op).
    int think_open_token_id() const { return think_open_token_id_; }
    int think_close_token_id() const { return think_close_token_id_; }
    // Token id for the "\n\n" double-newline used to separate </think> from
    // the answer in Qwen3.5's chat_template. Engine resolves it from the
    // tokenizer at init (BPE: token "ĊĊ" = 271 for the Qwen3.5 vocab);
    // return -1 if the tokenizer does not have it as a single token.
    int double_newline_token_id() const { return double_newline_token_id_; }
    kvcache::BlockPool* block_pool() { return block_pool_.get(); }
    kvcache::BlockAllocator* block_allocator() { return block_allocator_.get(); }
    kvcache::PrefixCache* prefix_cache() { return prefix_cache_.get(); }
    // SSM snapshot cache — non-null only when the loaded model is hybrid
    // (Qwen3.5 family). Persists per-prompt linear-attention state so
    // prefix-cache reuse remains correct for the SSM layers.
    kvcache::SSMSnapshotCache* ssm_snapshot_cache() { return ssm_snapshot_cache_.get(); }

    // Multimodal helpers — only valid when the loaded model is a Qwen3.5
    // vision-capable variant; throw std::runtime_error otherwise.
    //
    //   encode_image_data_uri  : full base64 image URI → device-resident
    //                            [num_image_tokens, hidden_size] embedding tensor
    //                            produced by the vision tower + spatial merger.
    //   build_multimodal_input_embeds : look up text token embeddings for
    //                            input_ids and scatter the supplied image
    //                            embedding chunks (in encounter order) over
    //                            <|image_pad|> positions. Returns
    //                            [input_ids.size(), hidden_size].
    //   image_pad_token_id     : tokenizer id of <|image_pad|>, or -1.
    bool has_vision() const;
    tensor_t encode_image_data_uri(std::string_view data_uri);
    tensor_t build_multimodal_input_embeds(const std::vector<int>& input_ids,
                                           const std::vector<tensor_t>& image_embeds_chunks);
    int image_pad_token_id() const;
    model::DecodeScratch* decode_scratch() { return decode_scratch_.get(); }
    // Hybrid SSM state pool, if the loaded model owns one (Qwen3.5 / Qwen3.5-MoE).
    // Returns nullptr for non-hybrid models so Scheduler keeps single-pool semantics.
    model::SSMStatePool* ssm_state_pool();
    const std::string& model_name() const { return model_name_; }

    // Sub-component accessors (callers use these directly instead of delegation)
    ServingLoop& serving_loop() { return *serving_loop_; }
    Profiler& profiler() { return *profiler_; }

private:
    InferenceEngine(std::shared_ptr<model::Model> model, std::shared_ptr<tokenizer::Tokenizer> tokenizer,
                    std::shared_ptr<sampler::Sampler> sampler, device::Device device, ExecutorConfig exec_config,
                    ChatTemplate chat_template);

    std::shared_ptr<model::Model> model_;
    std::shared_ptr<tokenizer::Tokenizer> tokenizer_;
    std::shared_ptr<sampler::Sampler> sampler_;
    std::shared_ptr<sampler::Sampler> argmax_sampler_;
    std::shared_ptr<sampler::GeneralSampler> general_sampler_;
    std::shared_ptr<ChatTemplateJinja> chat_template_jinja_; // nullable
    device::Device device_;
    ExecutorConfig exec_config_;
    ChatTemplate chat_template_;
    std::vector<int> stop_token_ids_;
    int think_open_token_id_ = -1;
    int think_close_token_id_ = -1;
    int double_newline_token_id_ = -1;
    SchedulerConfig scheduler_config_;
    std::string model_name_;
    std::unique_ptr<kvcache::BlockPool> block_pool_;
    std::unique_ptr<kvcache::BlockAllocator> block_allocator_;
    std::unique_ptr<kvcache::PrefixCache> prefix_cache_;
    std::unique_ptr<kvcache::SSMSnapshotCache> ssm_snapshot_cache_;
    std::unique_ptr<MultiModalProcessor> mm_processor_; // lazy, non-null only for vision models
    int image_pad_token_id_ = -1;
    std::unique_ptr<model::DecodeScratch> decode_scratch_;

    // Owned sub-components
    std::unique_ptr<ServingLoop> serving_loop_;
    std::unique_ptr<Profiler> profiler_;

    void init_block_pool();
    void build_stop_token_ids();
};

} // namespace zedinfer
