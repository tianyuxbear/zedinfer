#include "zedinfer/chat_template_jinja.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <plog/Log.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

// minja uses nlohmann::ordered_json under the json alias. Include the chat
// template header *before* declaring our own ordered_json users so the alias
// resolves to the same type used by minja::chat_template_inputs.
#include <minja/chat-template.hpp>

namespace zedinfer {

namespace {

// Helper: read a whole file into a std::string. Throws on open failure.
std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("[ChatTemplateJinja] failed to open template file: " + path);
    }
    std::string contents((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return contents;
}

// Convert a ChatMessageMM into the ordered_json shape expected by minja.
// std::string content -> {"role": ..., "content": "<text>"}
// vector<ContentPart>  -> {"role": ..., "content": [{"type": "text", "text": ...} | {"type": "image", "image": ...}]}
nlohmann::ordered_json to_json_message(const ChatMessageMM& msg) {
    nlohmann::ordered_json out;
    out["role"] = msg.role;

    std::visit(
        [&out](const auto& payload) {
            using T = std::decay_t<decltype(payload)>;
            if constexpr (std::is_same_v<T, std::string>) {
                out["content"] = payload;
            } else {
                // Multimodal: render each part as a typed object. The Qwen3.5
                // render_content macro checks `'image' in item or item.type == 'image'`
                // and `'text' in item`, so both shapes work.
                nlohmann::ordered_json arr = nlohmann::ordered_json::array();
                for (const auto& part : payload) {
                    std::visit(
                        [&arr](const auto& concrete) {
                            using P = std::decay_t<decltype(concrete)>;
                            nlohmann::ordered_json item;
                            if constexpr (std::is_same_v<P, TextPart>) {
                                item["type"] = "text";
                                item["text"] = concrete.text;
                            } else if constexpr (std::is_same_v<P, ImagePart>) {
                                item["type"] = "image";
                                item["image"] = concrete.data_uri;
                            }
                            arr.push_back(std::move(item));
                        },
                        part);
                }
                out["content"] = std::move(arr);
            }
        },
        msg.content);

    return out;
}

} // namespace

struct ChatTemplateJinja::Impl {
    // Owned compiled Jinja AST. minja::chat_template performs capability probes
    // in its constructor (which can throw on malformed templates), so we
    // construct via emplace inside load().
    std::unique_ptr<minja::chat_template> tpl;
};

ChatTemplateJinja ChatTemplateJinja::load(const std::string& template_file) {
    auto impl = std::make_shared<Impl>();
    std::string source = read_file(template_file);
    if (source.empty()) {
        throw std::runtime_error("[ChatTemplateJinja] template file is empty: " + template_file);
    }

    // BOS/EOS not used by the Qwen3.5 jinja; pass "" to match the T2 smoke test.
    // If a future template references them via {{ bos_token }}, the caller can
    // surface them through extra_context until load() learns to read
    // tokenizer_config.json.
    impl->tpl = std::make_unique<minja::chat_template>(source, /*bos_token=*/"", /*eos_token=*/"");

    LOGI << "[ChatTemplateJinja] Compiled chat template from " << template_file
         << " (" << source.size() << " bytes)";

    ChatTemplateJinja out;
    out.impl_ = std::move(impl);
    return out;
}

ChatTemplateJinja ChatTemplateJinja::load_from_source(const std::string& source) {
    auto impl = std::make_shared<Impl>();
    if (source.empty()) {
        throw std::runtime_error("[ChatTemplateJinja] empty inline chat template source");
    }
    impl->tpl = std::make_unique<minja::chat_template>(source, /*bos_token=*/"", /*eos_token=*/"");
    LOGI << "[ChatTemplateJinja] Compiled chat template from inline source (" << source.size() << " bytes)";
    ChatTemplateJinja out;
    out.impl_ = std::move(impl);
    return out;
}

std::string ChatTemplateJinja::render(const std::vector<ChatMessageMM>& messages, bool add_generation_prompt,
                                      bool enable_thinking, const nlohmann::ordered_json* tools) const {
    if (!impl_ || !impl_->tpl) {
        throw std::runtime_error("[ChatTemplateJinja] render() called on an uninitialised template");
    }

    nlohmann::ordered_json msgs_json = nlohmann::ordered_json::array();
    msgs_json.get_ptr<nlohmann::ordered_json::array_t*>()->reserve(messages.size());
    for (const auto& m : messages) {
        msgs_json.push_back(to_json_message(m));
    }

    minja::chat_template_inputs inputs;
    inputs.messages = std::move(msgs_json);
    inputs.add_generation_prompt = add_generation_prompt;
    if (tools != nullptr && !tools->is_null()) {
        inputs.tools = *tools;
    }
    // The Qwen3.5 template reads `enable_thinking` from its evaluation context.
    // Pass it via extra_context — minja's apply() forwards extra_context entries
    // as top-level variables to the rendered template.
    inputs.extra_context = nlohmann::ordered_json{{"enable_thinking", enable_thinking}};

    minja::chat_template_options opts; // defaults (apply_polyfills=true) match upstream expectations
    return impl_->tpl->apply(inputs, opts);
}

} // namespace zedinfer
