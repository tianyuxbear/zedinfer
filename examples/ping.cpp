#include "backend/device/device.hpp"
#include "utils/logging.hpp"
#include "utils/system_info.hpp"
#include "zedinfer.h"
#include "zedinfer/chat_template_jinja.hpp"
#include "zedinfer/engine.hpp"
#include "zedinfer/multimodal_processor.hpp"
#include "zedinfer/request.hpp"
#include "zedinfer/scheduler.hpp"
#include "zedinfer/serving_loop.hpp"
#include "zedinfer/session.hpp"

#include "zedinfer/version.hpp"
#include <argparse/argparse.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

using namespace zedinfer;

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("ZedInfer Ping", std::string("zedinfer ") + ZEDINFER_VERSION + " (build "
                                                          + ZEDINFER_GIT_HASH + ", " + ZEDINFER_BUILD_DATE + ")");

    program.add_argument("model_path").help("Path to the model directory");

    program.add_argument("--nvidia").help("Use NVIDIA GPU backend").default_value(false).implicit_value(true);

    program.add_argument("--gpu-memory-utilization")
        .help("Fraction of GPU memory for KV cache (0.0-1.0)")
        .default_value(0.9f)
        .scan<'g', float>();

    program.add_argument("--prompt")
        .help("Prompt text for single-turn generation")
        .default_value(std::string("Who are you?"));

    program.add_argument("--max-new-tokens")
        .help("Maximum number of new tokens to generate")
        .default_value(512)
        .scan<'i', int>();

    program.add_argument("--thinking", "--enable-thinking")
        .help("Enable the reasoning <think> block (Qwen3.5). Off by default because the "
              "open-<think> branch is fragile on GPTQ-Int4 weights and produces hallucinated "
              "prompt content. Opt in if the weights handle empty-thinking-start cleanly.")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--max-think-tokens")
        .help("Force </think> after this many tokens inside an open <think> block (0 = disabled)")
        .default_value(0)
        .scan<'i', int>();

    program.add_argument("--stream")
        .help("Stream tokens to stdout as they are generated")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--mtp")
        .help("Enable Qwen3.5 MTP speculative decoding (off by default). "
              "Only effective when the loaded model ships an MTP head.")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--image")
        .help("Path to a local image file (PNG/JPEG) to send alongside --prompt. "
              "Requires a vision-capable model (Qwen3.5-VL family). Mutually "
              "exclusive with multi-turn session use — vision input always runs "
              "stateless full prefill.")
        .default_value(std::string(""));

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    utils::initLoggerWithOverwrite(plog::verbose, "logs/ping.log");
    LOG_VERBOSE_(utils::BOTH) << utils::get_runtime_info();

    auto model_path = program.get<std::string>("model_path");
    bool use_nvidia = program.get<bool>("--nvidia");
    int max_think_tokens = program.get<int>("--max-think-tokens");
    if (max_think_tokens < 0) {
        std::cerr << "--max-think-tokens must be >= 0" << std::endl;
        return 1;
    }

    zedinferDeviceType_t device_type = use_nvidia ? ZEDINFER_DEVICE_NVIDIA : ZEDINFER_DEVICE_CPU;
    device::Device device(device_type, 0);

    SchedulerConfig sched_config;
    sched_config.gpu_memory_utilization = program.get<float>("--gpu-memory-utilization");
    sched_config.mtp_enabled = program.get<bool>("--mtp");
    if (const char* env = std::getenv("ZEDINFER_KV_BLOCK_SIZE")) {
        sched_config.kv_block_size = std::atoi(env);
        printf("[ping] Using kv_block_size=%d from env\n", sched_config.kv_block_size);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    std::shared_ptr<InferenceEngine> engine;
    try {
        engine = InferenceEngine::create(model_path, device, sched_config);
    } catch (const std::exception& err) {
        std::string msg = err.what() ? err.what() : "";
        // Qwen3.5 currently throws "not implemented until M1" from forward_config(),
        // which engine init invokes for KV/scratch sizing. Treat this as a clean WIP exit.
        if (msg.find("not implemented until M1") != std::string::npos) {
            std::cout << "[ping] Qwen3.5 forward path is M1 work-in-progress; exiting clean.\n"
                      << "       Detail: " << msg << "\n";
            return 0;
        }
        std::cerr << "[ping] engine init failed: " << msg << "\n";
        std::cerr << "       Hint: pinned-host memory or VRAM may be insufficient; try --gpu-memory-utilization 0.5 or "
                     "a smaller model.\n";
        return 3;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    printf("Engine initialized in %ld ms (%s)\n", init_ms, use_nvidia ? "NVIDIA" : "CPU");

    GenerationConfig gen_config;
    gen_config.gen_mode = GenerationMode::PING;
    gen_config.max_new_tokens = program.get<int>("--max-new-tokens");
    gen_config.enable_thinking = program.get<bool>("--thinking");
    gen_config.max_think_tokens = max_think_tokens;
    gen_config.verbose = true;
    gen_config.print_stats = true;
    if (program.get<bool>("--stream")) {
        gen_config.stream = true;
        gen_config.stream_callback = [](const std::string& tok) { std::cout << tok << std::flush; };
    }

    auto prompt = program.get<std::string>("--prompt");
    auto image_path = program.get<std::string>("--image");

    // Multimodal path: bypass InferenceSession (it does not understand
    // pre-built input_embeds) and submit a stateless multimodal request
    // straight to the serving loop. This mirrors the HTTP /v1/chat/completions
    // multimodal flow so a future user can compare CLI vs HTTP outputs.
    if (!image_path.empty()) {
        if (!engine->has_vision()) {
            std::cerr << "[ping] --image was passed but the loaded model has no vision tower. "
                         "Re-run with a Qwen3.5-VL checkpoint.\n";
            return 4;
        }
        const auto* jinja = engine->chat_template_jinja();
        if (!jinja) {
            std::cerr << "[ping] --image requires a Jinja chat template in the model directory; "
                         "none found.\n";
            return 4;
        }

        // 1. Read image bytes from disk and base64-encode them into a data URI.
        std::ifstream f(image_path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "[ping] failed to open image: " << image_path << "\n";
            return 4;
        }
        std::ostringstream oss;
        oss << f.rdbuf();
        std::string raw = oss.str();
        // Minimal in-house base64 encoder — avoids dragging another dep.
        static const char b64alpha[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string b64;
        b64.reserve(((raw.size() + 2) / 3) * 4);
        for (size_t i = 0; i < raw.size(); i += 3) {
            const uint8_t b0 = static_cast<uint8_t>(raw[i]);
            const uint8_t b1 = (i + 1 < raw.size()) ? static_cast<uint8_t>(raw[i + 1]) : 0;
            const uint8_t b2 = (i + 2 < raw.size()) ? static_cast<uint8_t>(raw[i + 2]) : 0;
            const uint32_t bits = (uint32_t(b0) << 16) | (uint32_t(b1) << 8) | b2;
            b64.push_back(b64alpha[(bits >> 18) & 0x3F]);
            b64.push_back(b64alpha[(bits >> 12) & 0x3F]);
            b64.push_back(i + 1 < raw.size() ? b64alpha[(bits >> 6) & 0x3F] : '=');
            b64.push_back(i + 2 < raw.size() ? b64alpha[bits & 0x3F] : '=');
        }
        std::string data_uri = "data:image/png;base64," + b64;

        // 2. Encode the image with the vision tower.
        tensor_t image_embed;
        try {
            image_embed = engine->encode_image_data_uri(data_uri);
        } catch (const std::exception& e) {
            std::cerr << "[ping] vision encode failed: " << e.what() << "\n";
            return 4;
        }
        const int n_image_tokens = static_cast<int>(image_embed->dim(0));

        // 3. Render the Jinja prompt with one (image, text) content pair.
        std::vector<ChatMessageMM> mm_messages(1);
        mm_messages[0].role = "user";
        std::vector<ContentPart> parts;
        parts.push_back(ImagePart{data_uri});
        parts.push_back(TextPart{prompt});
        mm_messages[0].content = std::move(parts);
        std::string rendered = jinja->render(mm_messages, /*add_generation_prompt=*/true, gen_config.enable_thinking);

        // 4. Expand the single <|image_pad|> placeholder to N copies so the
        //    tokenized input_ids matches the vision tower output row count.
        const std::string pad = "<|image_pad|>";
        std::string expanded;
        expanded.reserve(rendered.size() + n_image_tokens * pad.size());
        size_t pos = rendered.find(pad);
        if (pos == std::string::npos) {
            std::cerr << "[ping] rendered prompt has no <|image_pad|> placeholder; chat template mismatch?\n";
            return 4;
        }
        expanded.append(rendered, 0, pos);
        for (int k = 0; k < n_image_tokens; ++k) { expanded.append(pad); }
        expanded.append(rendered, pos + pad.size(), rendered.size() - pos - pad.size());

        std::vector<int> input_ids = engine->tokenizer().encode(expanded);
        tensor_t input_embeds = engine->build_multimodal_input_embeds(input_ids, {image_embed});

        // 5. Submit the request directly to the serving loop.
        auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
        auto req = std::make_unique<InferenceRequest>();
        req->input_ids = std::move(input_ids);
        req->config = gen_config;
        req->cancelled = cancel_flag;
        if (gen_config.stream) {
            req->stream_callback = gen_config.stream_callback;
        }
        req->set_input_embeds(input_embeds);
        req->arrival_time = std::chrono::steady_clock::now();

        // Drive the serving loop on this thread; the engine was created with
        // its own thread for ServingLoop but ping does not start it. Mirror
        // the serve binary's pattern.
        std::thread serving_thread([&engine] { engine->serving_loop().run_serving(); });
        auto future = engine->serving_loop().submit_async(std::move(req));
        try {
            auto result = future.get();
            std::string out = engine->tokenizer().decode(result.output_ids);
            if (!gen_config.stream) {
                std::cout << out << std::endl;
            } else {
                std::cout << std::endl;
            }
        } catch (const std::exception& e) { std::cerr << "[ping] generation failed: " << e.what() << "\n"; }
        engine->serving_loop().stop();
        serving_thread.join();
        return 0;
    }

    try {
        auto session = engine->create_session(gen_config);
        session->chat(prompt);
    } catch (const std::exception& err) {
        std::string msg = err.what() ? err.what() : "";
        if (msg.find("not implemented until M1") != std::string::npos) {
            std::cout << "[ping] Qwen3.5 forward path is M1 work-in-progress; exiting clean.\n"
                      << "       Detail: " << msg << "\n";
            return 0;
        }
        std::cerr << "[ping] generation failed: " << msg << "\n";
        return 2;
    }

    return 0;
}
