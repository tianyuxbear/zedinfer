#include "zedinfer/engine.hpp"
#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "backend/kvcache/dynamic.hpp"
#include "backend/kvcache/paged.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/hf_tokenizer.hpp"
#include "utils/logging.hpp"
#include "utils/random.hpp"
#include "utils/types.hpp"
#include "zedinfer/session.hpp"

#include <chrono>
#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer {

InferenceEngine::InferenceEngine(
    std::shared_ptr<model::Model> model,
    std::shared_ptr<tokenizer::Tokenizer> tokenizer,
    std::shared_ptr<sampler::Sampler> sampler,
    device::Device device,
    ExecutorConfig exec_config,
    ChatTemplate chat_template)
    : model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      sampler_(std::move(sampler)),
      device_(device),
      exec_config_(exec_config),
      chat_template_(std::move(chat_template)) {}

std::shared_ptr<InferenceEngine> InferenceEngine::create(
    const std::string &model_path,
    device::Device device) {

    if (model_path.empty()) {
        throw std::invalid_argument("Model path cannot be empty");
    }

    LOGI << "[Engine] Loading model from: " << model_path;

    auto model = model::Model::parse(model_path, device.type());
    if (!model) {
        throw std::runtime_error("Failed to parse model from: " + model_path);
    }
    LOGI << "[Engine] Model loaded: " << model->model_type();

    auto tokenizer = tokenizer::HFTokenizer::create(model_path + "/tokenizer.json");
    if (!tokenizer) {
        throw std::runtime_error("Failed to load tokenizer");
    }

    ExecutorConfig exec_config;
    exec_config.device_type = device.type();
    exec_config.device_id = device.id();
    exec_config.data_type = utils::str_to_dtype(model->config().torch_dtype);
    exec_config.max_seq_len = tokenizer->get_config().model_max_length;

    auto sampler = sampler::createSampler(exec_config, sampler::SamplerType::ARGMAX);

    auto chat_template = ChatTemplate::load(model_path, model->model_type());

    auto engine = std::shared_ptr<InferenceEngine>(
        new InferenceEngine(
            std::move(model),
            std::move(tokenizer),
            std::move(sampler),
            device,
            exec_config,
            std::move(chat_template)));

    engine->build_stop_token_ids();

    LOGI << "[Engine] Initialization complete";

    LOG_VERBOSE_(utils::BOTH) << "[Engine] Performing warmup...";
    engine->warmup();
    LOG_VERBOSE_(utils::BOTH) << "[Engine] Ready";

    // Create block pool for paged KV cache (after warmup so VRAM is settled)
    engine->init_block_pool();

    return engine;
}

std::unique_ptr<InferenceSession> InferenceEngine::create_session(
    const GenerationConfig &config) {

    const auto &mc = model_->config();

    kvcache::KVCacheConfig base_kv_config;
    base_kv_config.num_layers = mc.num_hidden_layers;
    base_kv_config.num_kv_heads = mc.num_key_value_heads;
    base_kv_config.head_dim = mc.hidden_size / mc.num_attention_heads;
    base_kv_config.device_type = device_.type();
    base_kv_config.device_id = device_.id();
    base_kv_config.dtype = utils::str_to_dtype(mc.torch_dtype);

    kvcache::kvcache_t kv_cache;

    if (block_allocator_ && scheduler_config_.use_paged_kvcache) {
        kv_cache = std::make_unique<kvcache::PagedKVCache>(
            base_kv_config, *block_allocator_, 256);
        LOGI << "[Session] Paged KV cache initialized: "
             << "capacity=" << kv_cache->allocated_capacity()
             << ", pool_free=" << block_pool_->free_blocks() << "/" << block_pool_->total_blocks();
    } else {
        kvcache::DynamicKVCacheConfig dyn_config;
        static_cast<kvcache::KVCacheConfig &>(dyn_config) = base_kv_config;
        dyn_config.model_max_seq_len = tokenizer_->get_config().model_max_length;
        kv_cache = kvcache::DynamicKVCache::create_dynamic_kvcache(dyn_config);
        LOGI << "[Session] Dynamic KV cache initialized: "
             << "capacity=" << kv_cache->allocated_capacity()
             << ", memory=" << kv_cache->memory_usage() / (1024.0 * 1024.0) << " MB";
    }

    return std::unique_ptr<InferenceSession>(
        new InferenceSession(shared_from_this(), std::move(kv_cache), config, chat_template_));
}

std::string InferenceEngine::generate(
    kvcache::KVCache &kvcache,
    const std::string &prompt,
    const GenerationConfig &config) {

    config.validate();

    if (config.verbose) {
        if (config.gen_mode == GenerationMode::PING) {
            LOG_INFO_(utils::BOTH) << "\n=== [Inference] Prompt: ===\n" << prompt;
        } else {
            LOGI << "\n=== [Inference] Prompt: ===\n" << prompt;
        }
        LOGI << "[Inference] Encoding prompt...";
    }

    auto input_ids = tokenizer_->encode(prompt);

    if (config.verbose) {
        LOGI << "[Inference] Prompt tokens: " << input_ids.size();
        LOGI << "[Inference] Generating...";
    }

    auto result = generate_tokens(kvcache, input_ids, config);
    std::string output = tokenizer_->decode(result.output_ids);

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

GenerationResult InferenceEngine::generate_tokens(
    kvcache::KVCache &kvcache,
    const std::vector<int> &input_ids,
    const GenerationConfig &config) {

    auto request = build_request(input_ids, config);
    scheduler_.submit(std::move(request));
    return scheduler_.run_one(
        *model_, kvcache, exec_config_,
        *sampler_, *tokenizer_, stop_token_ids_);
}

std::unique_ptr<InferenceRequest> InferenceEngine::build_request(
    const std::vector<int> &input_ids,
    const GenerationConfig &config) {

    auto req = std::make_unique<InferenceRequest>();
    req->input_ids = input_ids;
    req->config = config;
    req->stream_callback = config.stream ? config.stream_callback : nullptr;
    req->arrival_time = std::chrono::steady_clock::now();
    return req;
}

void InferenceEngine::init_block_pool() {
    if (!scheduler_config_.use_paged_kvcache) {
        LOGI << "[Engine] Paged KV cache disabled, skipping block pool";
        return;
    }

    const auto &mc = model_->config();
    auto dtype = utils::str_to_dtype(mc.torch_dtype);

    // Query free device memory
    core::context().setDevice(device_.type(), device_.id());
    auto api = device::getRuntimeAPI(device_.type());
    size_t free_bytes = 0, total_bytes = 0;
    api->get_memory_info(&free_bytes, &total_bytes);

    LOGI << "[Engine] Device memory: free=" << free_bytes / (1024 * 1024)
         << " MB, total=" << total_bytes / (1024 * 1024) << " MB";

    // Calculate KV budget (vLLM-style: total * utilization - used)
    size_t used_bytes = total_bytes - free_bytes;
    size_t allowed_bytes = static_cast<size_t>(
        total_bytes * scheduler_config_.gpu_memory_utilization);
    size_t kv_budget = (allowed_bytes > used_bytes) ? (allowed_bytes - used_bytes) : 0;

    // Configure blocks
    kvcache::BlockConfig block_config;
    block_config.block_size = scheduler_config_.kv_block_size;
    block_config.num_kv_heads = mc.num_key_value_heads;
    block_config.head_dim = mc.hidden_size / mc.num_attention_heads;
    block_config.dtype = dtype;

    size_t block_bytes = block_config.block_bytes();
    int num_blocks = static_cast<int>(kv_budget / block_bytes);

    if (num_blocks <= 0) {
        LOGW << "[Engine] Not enough memory for block pool, falling back to DynamicKVCache";
        scheduler_config_.use_paged_kvcache = false;
        return;
    }

    LOGI << "[Engine] Creating block pool: " << num_blocks << " blocks x "
         << block_config.block_size << " tokens, "
         << (num_blocks * block_bytes) / (1024 * 1024) << " MB";

    block_pool_ = std::make_unique<kvcache::BlockPool>(
        block_config, num_blocks, device_.type(), device_.id());
    block_allocator_ = std::make_unique<kvcache::BlockAllocator>(
        *block_pool_, mc.num_hidden_layers);
}

void InferenceEngine::build_stop_token_ids() {
    auto add_unique = [this](int id) {
        if (id >= 0) {
            for (int existing : stop_token_ids_)
                if (existing == id) return;
            stop_token_ids_.push_back(id);
        }
    };

    // 1. Tokenizer's EOS
    add_unique(tokenizer_->get_eos_token_id());

    // 2. All model config EOS token IDs (handles array eos_token_id)
    for (int eos_id : model_->config().eos_token_ids) {
        add_unique(eos_id);
    }

    // 3. Resolve ChatTemplate eos_token string to ID (e.g., <|im_end|> for ChatML)
    if (!chat_template_.eos_token.empty()) {
        add_unique(tokenizer_->get_special_token_id(chat_template_.eos_token));
    }

    std::string ids_str;
    for (int id : stop_token_ids_) {
        if (!ids_str.empty()) ids_str += ", ";
        ids_str += std::to_string(id);
    }
    LOGI << "[Engine] Stop token IDs: [" << ids_str << "]";
}

bool InferenceEngine::should_stop(int token_id) const {
    for (int stop_id : stop_token_ids_) {
        if (token_id == stop_id) return true;
    }
    return false;
}

void InferenceEngine::warmup(size_t prefill_len, size_t decode_steps) {
    LOGI << "[Engine] Warming up with prefill_len=" << prefill_len
         << ", decode_steps=" << decode_steps;

    auto t0 = std::chrono::high_resolution_clock::now();

    kvcache::DynamicKVCacheConfig kv_config;
    const auto &mc = model_->config();

    kv_config.num_layers = mc.num_hidden_layers;
    kv_config.num_kv_heads = mc.num_key_value_heads;
    kv_config.head_dim = mc.hidden_size / mc.num_attention_heads;
    kv_config.device_type = device_.type();
    kv_config.device_id = device_.id();
    kv_config.dtype = utils::str_to_dtype(mc.torch_dtype);
    kv_config.initial_capacity = prefill_len + decode_steps;
    kv_config.model_max_seq_len = tokenizer_->get_config().model_max_length;

    auto tmp_kv = kvcache::DynamicKVCache::create_dynamic_kvcache(kv_config);

    int min_id = 100, max_id = mc.vocab_size - 100;
    std::vector<int> dummy(prefill_len);
    for (auto &t : dummy) t = utils::randint(min_id, max_id);

    int past_len = 0;
    auto logits = model_->forward(dummy, past_len, *tmp_kv, exec_config_);
    int next = sampler_->sample(logits);
    past_len += prefill_len;

    for (size_t i = 1; i < decode_steps; ++i) {
        logits = model_->forward({next}, past_len, *tmp_kv, exec_config_);
        next = sampler_->sample(logits);
        past_len++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    LOGI << "[Engine] Warmup complete in "
         << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms";
}

std::pair<double, double> InferenceEngine::profile(size_t prefill_len, size_t decode_steps) {
    LOGI << "[Engine] Profiling with prefill_len=" << prefill_len
         << ", decode_steps=" << decode_steps;

    kvcache::DynamicKVCacheConfig kv_config;
    const auto &mc = model_->config();

    kv_config.num_layers = mc.num_hidden_layers;
    kv_config.num_kv_heads = mc.num_key_value_heads;
    kv_config.head_dim = mc.hidden_size / mc.num_attention_heads;
    kv_config.device_type = device_.type();
    kv_config.device_id = device_.id();
    kv_config.dtype = utils::str_to_dtype(mc.torch_dtype);
    kv_config.initial_capacity = prefill_len + decode_steps;
    kv_config.model_max_seq_len = tokenizer_->get_config().model_max_length;

    auto tmp_kv = kvcache::DynamicKVCache::create_dynamic_kvcache(kv_config);

    int min_id = 100, max_id = mc.vocab_size - 100;
    std::vector<int> dummy(prefill_len);
    for (auto &t : dummy) t = utils::randint(min_id, max_id);

    auto p0 = std::chrono::high_resolution_clock::now();
    int past_len = 0;
    auto logits = model_->forward(dummy, past_len, *tmp_kv, exec_config_);
    int next = sampler_->sample(logits);
    auto p1 = std::chrono::high_resolution_clock::now();

    past_len += prefill_len;

    auto d0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 1; i < decode_steps; ++i) {
        logits = model_->forward({next}, past_len, *tmp_kv, exec_config_);
        next = sampler_->sample(logits);
        past_len++;
    }
    auto d1 = std::chrono::high_resolution_clock::now();

    return {
        std::chrono::duration<double, std::milli>(p1 - p0).count(),
        std::chrono::duration<double, std::milli>(d1 - d0).count()};
}

} // namespace zedinfer
