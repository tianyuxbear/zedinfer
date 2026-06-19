#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <variant>
#include <vector>

namespace zedinfer {

// A textual content fragment for a multimodal chat message.
struct TextPart {
    std::string text;
};

// An image fragment carried as a data URI (e.g. data:image/png;base64,...).
// The renderer forwards the URI to the Jinja template under the "image" key,
// matching Qwen3.5's render_content macro which dispatches on
//   'image' in item or 'image_url' in item or item.type == 'image'.
struct ImagePart {
    std::string data_uri;
};

using ContentPart = std::variant<TextPart, ImagePart>;

// OpenAI-style chat message with either a plain-text or multimodal payload.
// Kept separate from the existing zedinfer::ChatTemplate's pair<role, string>
// API so the Jinja path can coexist without disturbing the data-driven
// templates used by Qwen2/Qwen3/DeepSeek-R1.
struct ChatMessageMM {
    std::string role;
    std::variant<std::string, std::vector<ContentPart>> content;
    nlohmann::ordered_json tool_calls;
    std::string reasoning_content;
    std::string name;
    std::string tool_call_id;
};

// minja-backed chat template loader/renderer for Qwen3.5 and other models
// that ship a `chat_template.jinja` file. Compilation is performed once in
// load() and stored inside an Impl held by shared_ptr so the compiled AST is
// shared across copies (and across Qwen3_5Model -> Qwen3_5MoeModel via
// inheritance).
class ChatTemplateJinja {
public:
    // Read `template_file` from disk and compile it via minja. Throws on I/O
    // or parse errors. The bos/eos tokens are not read here because the
    // Qwen3.5 jinja template does not reference them; pass empty strings,
    // matching the T2 smoke test.
    static ChatTemplateJinja load(const std::string& template_file);

    // Compile an inline Jinja source string (without going through disk). Used
    // when the template ships embedded inside tokenizer_config.json under the
    // "chat_template" key — the convention for most HuggingFace models that
    // predate the standalone chat_template.jinja file (DeepSeek-R1, Qwen2, etc.).
    static ChatTemplateJinja load_from_source(const std::string& source);

    // Render a list of messages.
    //   add_generation_prompt: append the `<|im_start|>assistant\n` opener.
    //   enable_thinking:       toggle Qwen3.5's `<think>` block. When false,
    //                          the template emits an empty `<think>\n\n</think>\n\n`
    //                          stub instead of the open `<think>\n` tag.
    //   tools:                 OpenAI-style tools array (each entry shaped as
    //                          `{"type":"function","function":{...}}`) forwarded
    //                          to minja's chat_template_inputs.tools so the
    //                          Jinja template can format function definitions
    //                          into the prompt. Empty / null disables tool use.
    std::string render(const std::vector<ChatMessageMM>& messages, bool add_generation_prompt = true,
                       bool enable_thinking = true, const nlohmann::ordered_json* tools = nullptr) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace zedinfer
