#include "backend/device/device.hpp"
#include "plog/Severity.h"
#include "utils/logging.hpp"
#include "utils/system_info.hpp"
#include "zedinfer.h"
#include "zedinfer/engine.hpp"
#include "zedinfer/scheduler.hpp"

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace zedinfer;
namespace fs = std::filesystem;

namespace {

std::string to_lower(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return s;
}

bool is_json_file(const fs::path& path) {
    const std::string ext = to_lower(path.extension().string());
    return ext == ".json";
}

bool is_jsonl_file(const fs::path& path) {
    const std::string ext = to_lower(path.extension().string());
    return ext == ".jsonl";
}

std::string extract_text_field(const nlohmann::json& node, const std::string& field) {
    if (node.is_string()) {
        return node.get<std::string>();
    }

    if (node.is_object()) {
        auto it = node.find(field);
        if (it != node.end() && it->is_string()) {
            return it->get<std::string>();
        }
    }

    return "";
}

std::vector<std::string> load_dataset(const std::string& dataset_path, const std::string& json_field,
                                      size_t max_samples) {
    std::vector<std::string> samples;
    fs::path path(dataset_path);

    if (!fs::exists(path)) {
        throw std::runtime_error("Dataset file not found: " + dataset_path);
    }

    if (is_jsonl_file(path)) {
        std::ifstream fin(dataset_path);
        if (!fin.is_open()) {
            throw std::runtime_error("Failed to open dataset: " + dataset_path);
        }

        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) {
                continue;
            }

            nlohmann::json obj = nlohmann::json::parse(line);
            std::string text = extract_text_field(obj, json_field);
            if (!text.empty()) {
                samples.push_back(std::move(text));
            }
            if (max_samples > 0 && samples.size() >= max_samples) {
                break;
            }
        }

        return samples;
    }

    if (is_json_file(path)) {
        std::ifstream fin(dataset_path);
        if (!fin.is_open()) {
            throw std::runtime_error("Failed to open dataset: " + dataset_path);
        }

        nlohmann::json doc;
        fin >> doc;

        if (doc.is_array()) {
            for (const auto& node : doc) {
                std::string text = extract_text_field(node, json_field);
                if (!text.empty()) {
                    samples.push_back(std::move(text));
                }
                if (max_samples > 0 && samples.size() >= max_samples) {
                    break;
                }
            }
        } else {
            std::string text = extract_text_field(doc, json_field);
            if (!text.empty()) {
                samples.push_back(std::move(text));
            }
        }

        return samples;
    }

    std::ifstream fin(dataset_path);
    if (!fin.is_open()) {
        throw std::runtime_error("Failed to open dataset: " + dataset_path);
    }

    std::string content((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    if (!content.empty()) {
        samples.push_back(std::move(content));
    }

    return samples;
}

std::string current_time_string() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
#if defined(_WIN32)
    localtime_s(&local_tm, &now_c);
#else
    localtime_r(&now_c, &local_tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

void maybe_write_csv(const std::string& csv_path, const std::string& model_tag, const std::string& device_name,
                     const std::string& dataset, const PerplexityStats& stats) {
    if (csv_path.empty()) {
        return;
    }

    const bool write_header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;

    std::ofstream fout(csv_path, std::ios::app);
    if (!fout.is_open()) {
        throw std::runtime_error("Failed to open csv path: " + csv_path);
    }

    if (write_header) {
        fout << "run_id,timestamp,model_tag,device,dataset,context_window,total_samples,"
                "evaluated_samples,skipped_samples,total_input_tokens,total_eval_tokens,"
                "total_chunks,avg_nll,avg_nll_se,ppl,ppl_se,elapsed_ms\n";
    }

    fout << stats.run_id << "," << current_time_string() << "," << model_tag << "," << device_name << "," << dataset
         << "," << stats.context_window << "," << stats.total_samples << "," << stats.evaluated_samples << ","
         << stats.skipped_samples << "," << stats.total_input_tokens << "," << stats.total_eval_tokens << ","
         << stats.total_chunks << "," << std::setprecision(10) << stats.avg_nll << "," << std::setprecision(10)
         << stats.avg_nll_se << "," << std::setprecision(10) << stats.ppl << "," << std::setprecision(10)
         << stats.ppl_se << "," << std::setprecision(10) << stats.elapsed_ms << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    utils::initLoggerWithOverwrite(plog::verbose, "logs/ppl.log");
    LOG_VERBOSE_(utils::BOTH) << utils::get_runtime_info();

    argparse::ArgumentParser program("ZedInfer PPL");

    program.add_argument("model_path").help("Path to model directory");

    program.add_argument("dataset_path").help("Path to dataset file (.txt/.json/.jsonl)");

    program.add_argument("--json-field")
        .help("Field name for JSON/JSONL text extraction")
        .default_value(std::string("text"));

    program.add_argument("--max-samples")
        .help("Maximum number of samples to evaluate (0 means all)")
        .default_value(0)
        .scan<'i', int>();

    program.add_argument("--max-length")
        .help("Maximum number of tokens per sample after tokenization (0 means no extra truncation)")
        .default_value(0)
        .scan<'i', int>();

    program.add_argument("--context-window")
        .help("Context window size for perplexity evaluation (0 means model/tokenizer max)")
        .default_value(0)
        .scan<'i', int>();

    program.add_argument("--csv-out").help("Optional CSV output path").default_value(std::string(""));

    program.add_argument("--nvidia").help("Use NVIDIA GPU backend").default_value(false).implicit_value(true);

    program.add_argument("--gpu-memory-utilization")
        .help("Fraction of GPU memory for KV cache (0.0-1.0)")
        .default_value(0.9f)
        .scan<'g', float>();

    program.add_argument("--enable-thinking")
        .help("Accepted for CLI parity; perplexity uses raw tokenized dataset text")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--max-think-tokens")
        .help("Accepted for CLI parity; perplexity uses raw tokenized dataset text")
        .default_value(0)
        .scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const auto model_path = program.get<std::string>("model_path");
    const auto dataset_path = program.get<std::string>("dataset_path");
    const auto json_field = program.get<std::string>("--json-field");
    const int max_samples_i = program.get<int>("--max-samples");
    const int max_length_i = program.get<int>("--max-length");
    const int context_window_i = program.get<int>("--context-window");
    const auto csv_out = program.get<std::string>("--csv-out");
    const bool use_nvidia = program.get<bool>("--nvidia");
    const bool enable_thinking = program.get<bool>("--enable-thinking");
    const int max_think_tokens = program.get<int>("--max-think-tokens");

    if (max_samples_i < 0 || max_length_i < 0 || context_window_i < 0 || max_think_tokens < 0) {
        std::cerr << "max-samples, max-length, context-window and max-think-tokens must be >= 0" << std::endl;
        return 1;
    }

    const size_t max_samples = static_cast<size_t>(max_samples_i);
    const size_t max_length = static_cast<size_t>(max_length_i);
    const size_t context_window = static_cast<size_t>(context_window_i);

    zedinferDeviceType_t device_type = use_nvidia ? ZEDINFER_DEVICE_NVIDIA : ZEDINFER_DEVICE_CPU;
    device::Device device(device_type, 0);
    const std::string device_name = use_nvidia ? "nvidia" : "cpu";

    std::vector<std::string> samples;
    try {
        samples = load_dataset(dataset_path, json_field, max_samples);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    if (samples.empty()) {
        std::cerr << "No valid samples loaded from dataset." << std::endl;
        return 1;
    }

    SchedulerConfig sched_config;
    sched_config.gpu_memory_utilization = program.get<float>("--gpu-memory-utilization");
    if (enable_thinking || max_think_tokens > 0) {
        LOGW
            << "--enable-thinking/--max-think-tokens have no effect in ppl; perplexity uses raw tokenized dataset text";
    }

    LOGI << "Loaded samples: " << samples.size();
    LOGI << "Initializing engine on device: " << device_name;

    auto engine = InferenceEngine::create(model_path, device, sched_config);

    PerplexityEvalConfig eval_config;
    eval_config.context_window = context_window;
    eval_config.max_length = max_length;
    eval_config.verbose = true;

    auto stats = engine->evaluate_perplexity(samples, eval_config);

    std::string model_tag = engine->model_name();
    if (model_tag.empty()) {
        model_tag = fs::path(model_path).filename().string();
        if (model_tag.empty()) {
            model_tag = model_path;
        }
    }

    printf("\n==================== PPL Report ====================\n");
    printf("run_id          : %llu\n", static_cast<unsigned long long>(stats.run_id));
    printf("model           : %s\n", model_tag.c_str());
    printf("device          : %s\n", device_name.c_str());
    printf("dataset         : %s\n", dataset_path.c_str());
    printf("context_window  : %zu\n", stats.context_window);
    printf("samples         : %zu (evaluated=%zu, skipped=%zu)\n", stats.total_samples, stats.evaluated_samples,
           stats.skipped_samples);
    printf("tokens          : input=%zu, eval=%zu\n", stats.total_input_tokens, stats.total_eval_tokens);
    printf("chunks          : %zu\n", stats.total_chunks);
    printf("avg_nll         : %.8f\n", stats.avg_nll);
    printf("avg_nll_se      : %.8f\n", stats.avg_nll_se);
    printf("ppl             : %.8f\n", stats.ppl);
    printf("ppl_se          : %.8f\n", stats.ppl_se);
    printf("final_estimate  : PPL = %.4f +/- %.4f\n", stats.ppl, stats.ppl_se);
    printf("elapsed_ms      : %.2f\n", stats.elapsed_ms);
    printf("====================================================\n");

    maybe_write_csv(csv_out, model_tag, device_name, dataset_path, stats);
    if (!csv_out.empty()) {
        LOGI << "CSV written to: " << csv_out;
    }

    return 0;
}
