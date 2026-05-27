#include "zedinfer/engine.hpp"
#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "frontend/models/qwen3_5.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/hf_tokenizer.hpp"
#include "utils/logging.hpp"
#include "utils/types.hpp"
#include "zedinfer/session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <plog/Log.h>
#include <sstream>
#include <stdexcept>

namespace zedinfer {

namespace {

// Construct a sampler from the model's generation_config.json (HuggingFace
// convention). When `do_sample=true` and the parsed params validate, return a
// GeneralSampler so generations follow the model author's recommended decoding
// (temperature / top_k / top_p). Otherwise fall back to ARGMAX greedy.
//
// Greedy decoding on reasoning-style models is prone to deterministic loops
// once probability mass is concentrated on a small set of near-tied tokens
// (e.g. CoT "Wait, ..." patterns); honoring the model's own do_sample=true is
// the canonical fix.
std::shared_ptr<sampler::Sampler> create_sampler_from_generation_config(const std::string& model_path,
                                                                       const ExecutorConfig& exec_config) {
    namespace fs = std::filesystem;
    // Debug override: if ZEDINFER_FORCE_ARGMAX is set, ignore generation_config
    // and use deterministic greedy sampling. Useful for reproducible debugging
    // of forward-path correctness without sampler RNG noise.
    if (const char* force_argmax = std::getenv("ZEDINFER_FORCE_ARGMAX");
        force_argmax != nullptr && *force_argmax != '\0' && std::string(force_argmax) != "0") {
        LOGI << "[Sampler] ZEDINFER_FORCE_ARGMAX set; using ARGMAX regardless of generation_config.json";
        return sampler::createSampler(exec_config, sampler::SamplerType::ARGMAX);
    }
    fs::path gen_cfg_path = fs::path(model_path) / "generation_config.json";
    if (!fs::exists(gen_cfg_path)) {
        LOGI << "[Sampler] No generation_config.json at " << gen_cfg_path.string()
             << "; defaulting to ARGMAX (greedy)";
        return sampler::createSampler(exec_config, sampler::SamplerType::ARGMAX);
    }
    try {
        std::ifstream file(gen_cfg_path);
        nlohmann::json j;
        file >> j;

        const bool do_sample = j.value("do_sample", false);
        if (!do_sample) {
            LOGI << "[Sampler] generation_config.json has do_sample=false; using ARGMAX (greedy)";
            return sampler::createSampler(exec_config, sampler::SamplerType::ARGMAX);
        }

        sampler::SamplerParams params;
        params.temperature = j.value("temperature", params.temperature);
        params.top_k = j.value("top_k", params.top_k);
        params.top_p = j.value("top_p", params.top_p);
        // Repetition penalty: honor generation_config.json if specified, else
        // default to 1.0 (off) to match HF transformers and vLLM. A previous
        // version defaulted to 1.1 to suppress "Wait, the user is asking ..."
        // attractor loops in long open-<think> generation, but 1.1 is too
        // aggressive for arithmetic / code prompts: it demotes already-emitted
        // tokens that the answer LEGITIMATELY needs to repeat (digits like
        // "2", operators "+"/"=" in math; identifiers/keywords in code). The
        // first-token sample is unaffected (output_ids is empty), but step 2+
        // pulls the nucleus off the correct continuation, so a "What is 2+2?"
        // prompt would non-deterministically answer "2 and 3 is 5" or
        // "2+4=10" instead of "2+2=4". The thinking-loop drift is now
        // addressed by Scheduler::max_think_tokens force-emit, which doesn't
        // perturb non-thinking sampling.
        params.repetition_penalty = j.value("repetition_penalty", 1.0f);
        // Debug override: ZEDINFER_REPETITION_PENALTY=<float> bypasses the
        // generation_config value entirely. Useful when validating whether
        // the default 1.1 (introduced to suppress thinking-loop attractors)
        // is interfering with non-reasoning prompts.
        if (const char* env_rp = std::getenv("ZEDINFER_REPETITION_PENALTY");
            env_rp != nullptr && *env_rp != '\0') {
            try {
                params.repetition_penalty = std::stof(env_rp);
                LOGI.printf("[Sampler] ZEDINFER_REPETITION_PENALTY=%.3f overriding generation_config",
                            params.repetition_penalty);
            } catch (const std::exception& e) {
                LOGW << "[Sampler] ZEDINFER_REPETITION_PENALTY parse failed: " << e.what()
                     << "; using generation_config value " << params.repetition_penalty;
            }
        }
        // generation_config rarely sets a fixed seed; honor it if present, else 0
        // (createSampler will seed from std::random_device).
        params.seed = j.value("seed", 0u);

        if (!params.validate()) {
            LOGW << "[Sampler] generation_config.json params failed validation ("
                 << params.info() << "); falling back to ARGMAX";
            return sampler::createSampler(exec_config, sampler::SamplerType::ARGMAX);
        }

        LOGI.printf("[Sampler] Using GeneralSampler from generation_config.json: "
                    "temperature=%.3f top_k=%d top_p=%.3f repetition_penalty=%.3f",
                    params.temperature, params.top_k, params.top_p, params.repetition_penalty);
        return sampler::createSampler(exec_config, sampler::SamplerType::GENERAL, params);
    } catch (const std::exception& e) {
        LOGW << "[Sampler] Failed to parse " << gen_cfg_path.string() << ": " << e.what()
             << "; defaulting to ARGMAX";
        return sampler::createSampler(exec_config, sampler::SamplerType::ARGMAX);
    }
}

double read_tensor_scalar(const std::byte* data, zedinferDataType_t dtype, size_t index) {
    switch (dtype) {
        case ZEDINFER_DTYPE_F16:
            return utils::cast<float>(reinterpret_cast<const fp16_t*>(data)[index]);
        case ZEDINFER_DTYPE_BF16:
            return utils::cast<float>(reinterpret_cast<const bf16_t*>(data)[index]);
        case ZEDINFER_DTYPE_F32:
            return reinterpret_cast<const float*>(data)[index];
        case ZEDINFER_DTYPE_F64:
            return reinterpret_cast<const double*>(data)[index];
        case ZEDINFER_DTYPE_I8:
            return reinterpret_cast<const int8_t*>(data)[index];
        case ZEDINFER_DTYPE_I16:
            return reinterpret_cast<const int16_t*>(data)[index];
        case ZEDINFER_DTYPE_I32:
            return reinterpret_cast<const int32_t*>(data)[index];
        case ZEDINFER_DTYPE_I64:
            return static_cast<double>(reinterpret_cast<const int64_t*>(data)[index]);
        case ZEDINFER_DTYPE_U8:
            return reinterpret_cast<const uint8_t*>(data)[index];
        case ZEDINFER_DTYPE_U16:
            return reinterpret_cast<const uint16_t*>(data)[index];
        case ZEDINFER_DTYPE_U32:
            return reinterpret_cast<const uint32_t*>(data)[index];
        case ZEDINFER_DTYPE_U64:
            return static_cast<double>(reinterpret_cast<const uint64_t*>(data)[index]);
        default:
            throw std::runtime_error("Unsupported logits dtype for perplexity evaluation");
    }
}

double compute_row_nll(const std::byte* data, zedinferDataType_t dtype, size_t row_idx, int target_token,
                       size_t vocab_size, ptrdiff_t row_stride, ptrdiff_t col_stride) {
    if (target_token < 0 || static_cast<size_t>(target_token) >= vocab_size) {
        throw std::runtime_error("Target token is out of vocabulary range");
    }

    double max_logit = -std::numeric_limits<double>::infinity();
    for (size_t v = 0; v < vocab_size; ++v) {
        size_t idx = static_cast<size_t>(row_idx * row_stride + static_cast<ptrdiff_t>(v) * col_stride);
        max_logit = std::max(max_logit, read_tensor_scalar(data, dtype, idx));
    }

    double exp_sum = 0.0;
    for (size_t v = 0; v < vocab_size; ++v) {
        size_t idx = static_cast<size_t>(row_idx * row_stride + static_cast<ptrdiff_t>(v) * col_stride);
        exp_sum += std::exp(read_tensor_scalar(data, dtype, idx) - max_logit);
    }

    size_t target_idx = static_cast<size_t>(row_idx * row_stride + static_cast<ptrdiff_t>(target_token) * col_stride);
    double target_logit = read_tensor_scalar(data, dtype, target_idx);
    double log_denom = max_logit + std::log(exp_sum);
    return log_denom - target_logit;
}

} // namespace

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

    auto model = model::Model::parse(model_path, device.type(), sched_config.gpu_memory_utilization);
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

    auto sampler = create_sampler_from_generation_config(model_path, exec_config);
    auto chat_template = ChatTemplate::load(model_path, model->model_type());

    auto engine = std::shared_ptr<InferenceEngine>(new InferenceEngine(
        std::move(model), std::move(tokenizer), std::move(sampler), device, exec_config, std::move(chat_template)));

    engine->build_stop_token_ids();

    // Resolve <think>/</think> token ids from the tokenizer once at engine
    // init. Reasoning models in the Qwen3.5 family carry these as special
    // tokens (248068 / 248069); models that do not have them get -1 here and
    // the scheduler's force-emit-</think> path becomes a no-op.
    engine->think_open_token_id_ = engine->tokenizer_->get_special_token_id("<think>");
    engine->think_close_token_id_ = engine->tokenizer_->get_special_token_id("</think>");
    // Resolve "\n\n" token id for the post-</think> separator. The BPE
    // tokenizer encodes "\n\n" as a single token (id 271 for Qwen3.5's
    // vocab); if for some reason it splits, take the first id. We need this
    // so the scheduler can mirror the natural </think>\n\n pattern from the
    // training chat_template after force-closing thinking.
    {
        auto nl_ids = engine->tokenizer_->encode("\n\n");
        engine->double_newline_token_id_ = nl_ids.empty() ? -1 : nl_ids.back();
    }
    LOGI << "[Engine] Think tokens: <think>=" << engine->think_open_token_id_
         << " </think>=" << engine->think_close_token_id_
         << " \\n\\n=" << engine->double_newline_token_id_;

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
        engine->decode_scratch_ = model::DecodeScratch::create(engine->model_->config(), fwd_cfg, engine->exec_config_);
    }

    // Create profiler and run warmup (exercises paged attention kernels).
    // Profiler takes a non-owning reference — it must not co-own the engine, or we
    // rebuild the shared_ptr cycle we just broke.
    engine->profiler_ = std::make_unique<Profiler>(*engine);
    // Local debugging/profiling can skip engine warmup to isolate model correctness from the
    // startup benchmark pass. Normal runs keep warmup enabled.
    //
    // Hybrid Qwen3.5 models also skip warmup automatically: the warmup path drives
    // transformer_forward (Qwen2/Qwen3 path) which can't dispatch the hybrid
    // SSM+attention forward without a real InferenceRequest (for the SSM slot).
    // Real generation goes through ServingLoop which acquires SSM slots and
    // routes to hybrid_transformer_forward.
    const bool is_hybrid_qwen3_5 = (engine->model_->model_type() == "qwen3_5"
                                    || engine->model_->model_type() == "qwen3_5_moe");
    if (std::getenv("ZEDINFER_DISABLE_WARMUP") == nullptr && !is_hybrid_qwen3_5) {
        LOG_VERBOSE_(utils::BOTH) << "[Engine] Performing warmup...";
        engine->profiler_->warmup();
        LOG_VERBOSE_(utils::BOTH) << "[Engine] Ready";
    } else if (is_hybrid_qwen3_5) {
        LOG_VERBOSE_(utils::BOTH) << "[Engine] Warmup skipped (Qwen3.5 hybrid path requires per-request SSM slot)";
        LOG_VERBOSE_(utils::BOTH) << "[Engine] Ready";
    } else {
        LOG_VERBOSE_(utils::BOTH) << "[Engine] Warmup skipped by ZEDINFER_DISABLE_WARMUP";
        LOG_VERBOSE_(utils::BOTH) << "[Engine] Ready";
    }

    // Create serving loop (after block pool). Same non-owning contract as Profiler.
    engine->serving_loop_ = std::make_unique<ServingLoop>(*engine, engine->scheduler_config_);

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

    LOGI << "[Session] Created with " << block_table.pages[0].size()
         << " pages/layer, pool_free=" << block_pool_->free_blocks() << "/" << block_pool_->total_blocks();

    return std::unique_ptr<InferenceSession>(new InferenceSession(shared_from_this(), std::move(block_table),
                                                                  block_allocator_.get(), config, chat_template_));
}

PerplexityStats InferenceEngine::evaluate_perplexity(const std::vector<std::string>& samples,
                                                     const PerplexityEvalConfig& config) {
    if (!block_allocator_ || !block_pool_) {
        throw std::runtime_error("[PPL] Paged KV cache/block pool is unavailable");
    }

    PerplexityStats result;
    result.total_samples = samples.size();
    result.samples.reserve(samples.size());

    auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    result.run_id = static_cast<uint64_t>(now.time_since_epoch().count());

    const size_t model_max_len
        = std::min(exec_config_.max_seq_len, static_cast<size_t>(tokenizer_->get_config().model_max_length));
    const size_t context_window = (config.context_window == 0) ? model_max_len : config.context_window;
    if (context_window < 2) {
        throw std::invalid_argument("context_window must be >= 2");
    }
    if (context_window > model_max_len) {
        throw std::invalid_argument("context_window exceeds tokenizer/model max length");
    }
    result.context_window = context_window;

    const size_t calc_chunk = context_window;
    const size_t eval_from = context_window / 2;

    size_t chunk_count = 0;
    double chunk_nll_mean = 0.0;
    double chunk_nll_m2 = 0.0;
    double chunk_ppl_mean = 0.0;
    double chunk_ppl_m2 = 0.0;

    size_t eval_block_size = 16;
    auto fwd_cfg = model_->forward_config();

    if (config.verbose) {
        LOGI << "[PPL] Start evaluation" << ", run_id=" << result.run_id << ", samples=" << result.total_samples
             << ", context_window=" << context_window << ", calc_chunk=" << calc_chunk << ", eval_from=" << eval_from
             << ", max_length=" << config.max_length;
    }

    auto eval_start = std::chrono::high_resolution_clock::now();

    for (size_t sample_idx = 0; sample_idx < samples.size(); ++sample_idx) {
        const auto& text = samples[sample_idx];
        PerplexitySampleStats sample_stats;
        auto sample_start = std::chrono::high_resolution_clock::now();

        auto tokens = tokenizer_->encode(text);
        if (config.max_length > 0 && tokens.size() > config.max_length) {
            tokens.resize(config.max_length);
        }
        sample_stats.input_tokens = tokens.size();
        result.total_input_tokens += tokens.size();

        if (config.verbose) {
            LOGI << "[PPL] Sample " << (sample_idx + 1) << "/" << samples.size() << " begin"
                 << ", chars=" << text.size() << ", tokens=" << tokens.size();
        }

        if (tokens.size() < 2 * calc_chunk) {
            sample_stats.skipped = true;
            sample_stats.skip_reason = "token_count < 2 * context_window";
            result.skipped_samples++;

            if (config.verbose) {
                LOGI << "[PPL] Sample " << (sample_idx + 1) << " skipped: " << sample_stats.skip_reason;
            }

            result.samples.push_back(std::move(sample_stats));
            continue;
        }

        const size_t n_chunk = tokens.size() / calc_chunk;
        if (config.verbose) {
            LOGI << "[PPL] Sample " << (sample_idx + 1) << ": calculating perplexity over " << n_chunk
                 << " chunks, n_ctx=" << context_window;
        }

        auto block_table = block_allocator_->allocate_sequence(static_cast<int>(context_window));
        const int bos_token_id = tokenizer_->get_bos_token_id();

        try {
            for (size_t chunk_idx = 0; chunk_idx < n_chunk; ++chunk_idx) {
                size_t begin = chunk_idx * calc_chunk;
                size_t end = begin + calc_chunk;

                std::vector<int> chunk(tokens.begin() + static_cast<ptrdiff_t>(begin),
                                       tokens.begin() + static_cast<ptrdiff_t>(end));

                if (bos_token_id >= 0 && !chunk.empty()) {
                    chunk[0] = bos_token_id;
                }

                block_table.seq_len = 0;
                const size_t eval_begin = begin + eval_from + 1;
                double window_nll_sum = 0.0;
                size_t window_eval_tokens = 0;

                size_t local_begin = 0;
                while (local_begin + 1 < chunk.size()) {
                    size_t local_end = std::min(local_begin + eval_block_size, chunk.size() - 1);
                    std::vector<int> input_block(chunk.begin() + static_cast<ptrdiff_t>(local_begin),
                                                 chunk.begin() + static_cast<ptrdiff_t>(local_end));

                    tensor_t logits;
                    try {
                        block_allocator_->ensure_blocks(block_table, static_cast<int>(local_end));
                        model::PagedForwardContext ctx(input_block, static_cast<int>(local_begin), block_table,
                                                       *block_pool_);
                        logits = model::transformer_forward(fwd_cfg, ctx, exec_config_);
                    } catch (const std::runtime_error&) {
                        if (eval_block_size > 1) {
                            eval_block_size = std::max<size_t>(1, eval_block_size / 2);
                            if (config.verbose) {
                                LOGI << "[PPL] Reducing eval block size to " << eval_block_size
                                     << " due to allocation/runtime pressure";
                            }
                            continue;
                        }
                        throw;
                    }

                    block_table.seq_len = static_cast<int>(local_end);

                    if (logits->deviceType() != ZEDINFER_DEVICE_CPU) {
                        logits = logits->to(ZEDINFER_DEVICE_CPU, 0);
                    }

                    const auto& shape = logits->shape();
                    const auto& strides = logits->strides();
                    if (shape.size() != 2 || shape[0] != input_block.size() || shape[1] == 0) {
                        throw std::runtime_error("Expected logits shape [block, vocab_size] for perplexity eval");
                    }

                    const size_t vocab_size = shape[1];
                    const ptrdiff_t row_stride = strides[0];
                    const ptrdiff_t col_stride = strides[1];
                    const std::byte* data = logits->data();
                    const zedinferDataType_t dtype = logits->dtype();

                    for (size_t i = 0; i < input_block.size(); ++i) {
                        size_t global_target_idx = begin + local_begin + i + 1;
                        if (global_target_idx < eval_begin) {
                            continue;
                        }

                        int target_token = chunk[local_begin + i + 1];
                        double nll = compute_row_nll(data, dtype, i, target_token, vocab_size, row_stride, col_stride);

                        sample_stats.nll_sum += nll;
                        sample_stats.eval_tokens++;
                        window_nll_sum += nll;
                        window_eval_tokens++;
                    }

                    local_begin = local_end;
                }

                if (window_eval_tokens > 0) {
                    double chunk_avg_nll = window_nll_sum / static_cast<double>(window_eval_tokens);
                    double chunk_ppl = std::exp(chunk_avg_nll);

                    chunk_count++;

                    double delta_nll = chunk_avg_nll - chunk_nll_mean;
                    chunk_nll_mean += delta_nll / static_cast<double>(chunk_count);
                    chunk_nll_m2 += delta_nll * (chunk_avg_nll - chunk_nll_mean);

                    double delta_ppl = chunk_ppl - chunk_ppl_mean;
                    chunk_ppl_mean += delta_ppl / static_cast<double>(chunk_count);
                    chunk_ppl_m2 += delta_ppl * (chunk_ppl - chunk_ppl_mean);

                    if (config.verbose) {
                        std::cout << "[" << chunk_count << "]" << std::fixed << std::setprecision(4) << chunk_ppl
                                  << ",";
                        std::cout.flush();
                    }
                }

                if (config.verbose && (chunk_idx == 0 || ((chunk_idx + 1) % 16 == 0) || (chunk_idx + 1 == n_chunk))) {
                    LOGI << "[PPL] Sample " << (sample_idx + 1) << " chunk " << (chunk_idx + 1) << "/" << n_chunk
                         << ", token_range=[" << begin << ", " << end << ")"
                         << ", eval_tokens=" << sample_stats.eval_tokens;
                }
            }
        } catch (...) {
            block_allocator_->free_sequence(block_table);
            throw;
        }

        block_allocator_->free_sequence(block_table);

        if (sample_stats.eval_tokens == 0) {
            sample_stats.skipped = true;
            sample_stats.skip_reason = "no evaluable tokens";
            result.skipped_samples++;

            if (config.verbose) {
                LOGI << "[PPL] Sample " << (sample_idx + 1) << " skipped: " << sample_stats.skip_reason;
            }

            result.samples.push_back(std::move(sample_stats));
            continue;
        }

        sample_stats.avg_nll = sample_stats.nll_sum / static_cast<double>(sample_stats.eval_tokens);
        sample_stats.ppl = std::exp(sample_stats.avg_nll);

        result.evaluated_samples++;
        result.total_eval_tokens += sample_stats.eval_tokens;
        result.nll_sum += sample_stats.nll_sum;

        if (config.verbose) {
            auto sample_end = std::chrono::high_resolution_clock::now();
            double sample_ms = std::chrono::duration<double, std::milli>(sample_end - sample_start).count();
            double sample_tok_per_s = sample_ms > 0.0 ? (sample_stats.eval_tokens * 1000.0 / sample_ms) : 0.0;
            LOGI << "[PPL] Sample " << (sample_idx + 1) << " done" << ", eval_tokens=" << sample_stats.eval_tokens
                 << ", avg_nll=" << sample_stats.avg_nll << ", ppl=" << sample_stats.ppl << ", elapsed_ms=" << sample_ms
                 << ", tok_per_s=" << sample_tok_per_s;

            if (((sample_idx + 1) % 10 == 0 || (sample_idx + 1) == samples.size()) && result.total_eval_tokens > 0) {
                double running_avg_nll = result.nll_sum / static_cast<double>(result.total_eval_tokens);
                double running_ppl = std::exp(running_avg_nll);
                LOGI << "[PPL] Progress " << (sample_idx + 1) << "/" << samples.size()
                     << ", evaluated_samples=" << result.evaluated_samples
                     << ", skipped_samples=" << result.skipped_samples
                     << ", total_eval_tokens=" << result.total_eval_tokens << ", running_avg_nll=" << running_avg_nll
                     << ", running_ppl=" << running_ppl;
            }
        }

        result.samples.push_back(std::move(sample_stats));
    }

    if (result.total_eval_tokens > 0) {
        result.avg_nll = result.nll_sum / static_cast<double>(result.total_eval_tokens);
        result.ppl = std::exp(result.avg_nll);
    }

    result.total_chunks = chunk_count;
    if (chunk_count > 1) {
        double n = static_cast<double>(chunk_count);
        double nll_std = std::sqrt(chunk_nll_m2 / (n - 1.0));
        double ppl_std = std::sqrt(chunk_ppl_m2 / (n - 1.0));
        result.avg_nll_se = nll_std / std::sqrt(n);
        result.ppl_se = ppl_std / std::sqrt(n);
    }

    auto eval_end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(eval_end - eval_start).count();

    if (config.verbose) {
        if (result.total_chunks > 0) {
            std::cout << std::endl;
        }
        LOGI << "[PPL] run_id=" << result.run_id << ", samples=" << result.total_samples
             << ", evaluated=" << result.evaluated_samples << ", skipped=" << result.skipped_samples
             << ", chunks=" << result.total_chunks << ", context_window=" << result.context_window
             << ", eval_tokens=" << result.total_eval_tokens << ", avg_nll=" << result.avg_nll
             << ", avg_nll_se=" << result.avg_nll_se << ", ppl=" << result.ppl << ", ppl_se=" << result.ppl_se
             << ", elapsed_ms=" << result.elapsed_ms;

        LOGI << "[PPL] Final estimate: PPL = " << std::fixed << std::setprecision(4) << result.ppl << " +/- "
             << result.ppl_se;
    }

    return result;
}

// ============================================================================
// Block Pool Initialization
// ============================================================================

model::SSMStatePool* InferenceEngine::ssm_state_pool() {
    // Currently only Qwen3.5 dense and Qwen3.5-MoE own an SSMStatePool. Both
    // derive from Qwen3_5Model, so a single downcast resolves either. Other
    // models return nullptr — the scheduler keeps its single-pool behavior.
    if (auto* q35 = dynamic_cast<model::Qwen3_5Model*>(model_.get())) {
        return &q35->ssm_state_pool();
    }
    return nullptr;
}

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
    // Use explicit head_dim from config when available (Qwen3-MoE has head_dim=128
    // while hidden_size/num_attention_heads=64). Fallback to computed value.
    block_config.head_dim = mc.head_dim > 0 ? mc.head_dim : (mc.hidden_size / mc.num_attention_heads);
    block_config.dtype = dtype;

    size_t block_bytes = block_config.block_bytes();
    int num_blocks = static_cast<int>(kv_budget / (2 * block_bytes));

    auto fail = [&](const std::string& detail) {
        std::ostringstream oss;
        oss << "[Engine] Cannot allocate KV page pool: " << detail << ". " << "total=" << total_bytes / (1024 * 1024)
            << " MB, " << "free=" << free_bytes / (1024 * 1024) << " MB, " << "used=" << used_bytes / (1024 * 1024)
            << " MB, " << "allowed=" << allowed_bytes / (1024 * 1024) << " MB "
            << "(gpu_memory_utilization=" << scheduler_config_.gpu_memory_utilization << "), "
            << "budget=" << kv_budget / (1024 * 1024) << " MB, " << "per-block=" << (2 * block_bytes) / 1024 << " KB. "
            << "Retry with a larger --gpu-memory-utilization or free VRAM before launch.";
        throw std::runtime_error(oss.str());
    };

    if (num_blocks <= 0) {
        fail("budget is smaller than a single block");
    }

    LOGI << "[Engine] Creating KV page pool: " << num_blocks << " shared pages x " << block_config.block_size
         << " tokens, " << (2 * num_blocks * block_bytes) / (1024 * 1024) << " MB";

    int try_blocks = num_blocks;
    while (try_blocks > 0) {
        try {
            block_pool_ = std::make_unique<kvcache::BlockPool>(block_config, try_blocks, device_.type(), device_.id());
            break;
        } catch (const std::exception& e) {
            LOGW << "[Engine] KV pool allocation failed for " << try_blocks << " blocks: " << e.what();
            int next_try = try_blocks * 3 / 4;
            if (next_try >= try_blocks) {
                next_try = try_blocks - 1;
            }
            try_blocks = next_try;
        }
    }

    if (!block_pool_) {
        fail("all allocation retries exhausted");
    }

    if (try_blocks != num_blocks) {
        LOGW << "[Engine] KV page pool reduced to " << try_blocks << " blocks after allocation retries";
    }

    // Number of KV-bearing layers: for hybrid models (Qwen3.5) only the full-attention
    // layers contribute KV blocks; linear-attention layers carry SSM state in
    // SSMStatePool instead and must NOT consume KV slots. For non-hybrid models all
    // hidden layers carry KV → fall through to num_hidden_layers.
    size_t num_kv_layers = mc.num_hidden_layers;
    if (!mc.layer_types.empty()) {
        size_t full_count = 0;
        for (const auto& t : mc.layer_types) {
            if (t == "full_attention") {
                ++full_count;
            }
        }
        if (full_count > 0 && full_count < mc.num_hidden_layers) {
            num_kv_layers = full_count;
            LOGI << "[Engine] Hybrid model: KV pool sized for " << full_count << " full-attention layers (of "
                 << mc.num_hidden_layers << " total)";
        }
    }
    block_allocator_ = std::make_unique<kvcache::BlockAllocator>(*block_pool_, num_kv_layers);
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
