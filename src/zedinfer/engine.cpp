#include "zedinfer/engine.hpp"
#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/hf_tokenizer.hpp"
#include "utils/logging.hpp"
#include "utils/types.hpp"
#include "zedinfer/session.hpp"

#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer {

InferenceEngine::InferenceEngine(std::shared_ptr<model::Model> model, std::shared_ptr<tokenizer::Tokenizer> tokenizer,
                                 std::shared_ptr<sampler::Sampler> sampler, device::Device device,
                                 ExecutorConfig exec_config, ChatTemplate chat_template)
    : model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      sampler_(std::move(sampler)),
      device_(device),
      exec_config_(exec_config),
      chat_template_(std::move(chat_template)) {}

std::shared_ptr<InferenceEngine> InferenceEngine::create(const std::string& model_path, device::Device device,
                                                         SchedulerConfig sched_config) {
    if (model_path.empty()) {
        throw std::invalid_argument("Model path cannot be empty");
    }

    // Derive model name from directory basename
    std::string model_name = model_path;
    auto last_slash = model_name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        model_name = model_name.substr(last_slash + 1);
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

    auto engine = std::shared_ptr<InferenceEngine>(new InferenceEngine(
        std::move(model), std::move(tokenizer), std::move(sampler), device, exec_config, std::move(chat_template)));

    engine->build_stop_token_ids();

    engine->model_name_ = model_name;
    engine->scheduler_config_ = sched_config;

    LOGI << "[Engine] Initialization complete";

    // Create block pool first (warmup now uses paged path)
    engine->init_block_pool();

    // Create prefix cache (after block pool, before serving loop)
    if (engine->block_pool_ && engine->block_allocator_) {
        engine->prefix_cache_
            = std::make_unique<kvcache::PrefixCache>(*engine->block_pool_, engine->block_allocator_->num_layers());
    }

    // Create decode scratch buffers (pre-allocated for N=1 decode)
    {
        auto fwd_cfg = engine->model_->forward_config();
        engine->decode_scratch_
            = model::DecodeScratch::create(engine->model_->config(), fwd_cfg.has_qk_norm, engine->exec_config_);
    }

    // Create profiler and run warmup (exercises paged attention kernels)
    engine->profiler_ = std::make_unique<Profiler>(engine);
    LOG_VERBOSE_(utils::BOTH) << "[Engine] Performing warmup...";
    engine->profiler_->warmup();
    LOG_VERBOSE_(utils::BOTH) << "[Engine] Ready";

    // Create serving loop (after block pool)
    engine->serving_loop_ = std::make_unique<ServingLoop>(engine, engine->scheduler_config_);

    return engine;
}

// ============================================================================
// Session Management
// ============================================================================

std::unique_ptr<InferenceSession> InferenceEngine::create_session(const GenerationConfig& config) {
    if (!block_allocator_) {
        throw std::runtime_error("[Engine] Block allocator not initialized");
    }

    auto block_table = block_allocator_->allocate_sequence(256);

    LOGI << "[Session] Created with " << block_table.k_blocks[0].size()
         << " blocks/layer, pool_free=" << block_pool_->free_blocks() << "/" << block_pool_->total_blocks();

    return std::unique_ptr<InferenceSession>(new InferenceSession(shared_from_this(), std::move(block_table),
                                                                  block_allocator_.get(), config, chat_template_));
}

// ============================================================================
// Block Pool Initialization
// ============================================================================

void InferenceEngine::init_block_pool() {
    if (!scheduler_config_.use_paged_kvcache) {
        LOGI << "[Engine] Paged KV cache disabled, skipping block pool";
        return;
    }

    const auto& mc = model_->config();
    auto dtype = utils::str_to_dtype(mc.torch_dtype);

    core::context().setDevice(device_.type(), device_.id());
    auto api = device::getRuntimeAPI(device_.type());
    size_t free_bytes = 0, total_bytes = 0;
    api->get_memory_info(&free_bytes, &total_bytes);

    LOGI << "[Engine] Device memory: free=" << free_bytes / (1024 * 1024)
         << " MB, total=" << total_bytes / (1024 * 1024) << " MB";

    size_t used_bytes = total_bytes - free_bytes;
    size_t allowed_bytes = static_cast<size_t>(total_bytes * scheduler_config_.gpu_memory_utilization);
    size_t kv_budget = (allowed_bytes > used_bytes) ? (allowed_bytes - used_bytes) : 0;

    kvcache::BlockConfig block_config;
    block_config.block_size = scheduler_config_.kv_block_size;
    block_config.num_kv_heads = mc.num_key_value_heads;
    block_config.head_dim = mc.hidden_size / mc.num_attention_heads;
    block_config.dtype = dtype;

    size_t block_bytes = block_config.block_bytes();
    int num_blocks = static_cast<int>(kv_budget / block_bytes);

    if (num_blocks <= 0) {
        LOGW << "[Engine] Not enough memory for block pool";
        scheduler_config_.use_paged_kvcache = false;
        return;
    }

    LOGI << "[Engine] Creating block pool: " << num_blocks << " blocks x " << block_config.block_size << " tokens, "
         << (num_blocks * block_bytes) / (1024 * 1024) << " MB";

    block_pool_ = std::make_unique<kvcache::BlockPool>(block_config, num_blocks, device_.type(), device_.id());
    block_allocator_ = std::make_unique<kvcache::BlockAllocator>(*block_pool_, mc.num_hidden_layers);
}

// ============================================================================
// Stop Tokens
// ============================================================================

void InferenceEngine::build_stop_token_ids() {
    auto add_unique = [this](int id) {
        if (id >= 0) {
            for (int existing : stop_token_ids_) {
                if (existing == id) {
                    return;
                }
            }
            stop_token_ids_.push_back(id);
        }
    };

    add_unique(tokenizer_->get_eos_token_id());
    for (int eos_id : model_->config().eos_token_ids) { add_unique(eos_id); }
    if (!chat_template_.eos_token.empty()) {
        add_unique(tokenizer_->get_special_token_id(chat_template_.eos_token));
    }

    std::string ids_str;
    for (int id : stop_token_ids_) {
        if (!ids_str.empty()) {
            ids_str += ", ";
        }
        ids_str += std::to_string(id);
    }
    LOGI << "[Engine] Stop token IDs: [" << ids_str << "]";
}

} // namespace zedinfer
