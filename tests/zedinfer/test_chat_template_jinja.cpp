#include "zedinfer/chat_template_jinja.hpp"

#include <cstdlib>
#include <gtest/gtest.h>
#include <string>
#include <vector>

TEST(ChatTemplateJinja, RendersQwen3_5UserPrompt) {
    const char* model_path = std::getenv("ZEDINFER_TEST_MODEL_PATH");
    if (!model_path) {
        GTEST_SKIP() << "ZEDINFER_TEST_MODEL_PATH not set; "
                        "set it to a Qwen3.5 model dir to enable this test.";
    }

    auto tpl = zedinfer::ChatTemplateJinja::load(std::string(model_path) + "/chat_template.jinja");

    std::vector<zedinfer::ChatMessageMM> msgs;
    zedinfer::ChatMessageMM m;
    m.role = "user";
    m.content = std::string("Who are you?");
    msgs.push_back(m);

    std::string rendered = tpl.render(msgs, /*add_generation_prompt=*/true);
    EXPECT_NE(rendered.find("<|im_start|>user"), std::string::npos);
    EXPECT_NE(rendered.find("Who are you?"), std::string::npos);
    EXPECT_NE(rendered.find("<|im_start|>assistant"), std::string::npos);
}

TEST(ChatTemplateJinja, FoldsTypedTextAndPreservesToolHistory) {
    auto tpl = zedinfer::ChatTemplateJinja::load_from_source(
        "{%- for message in messages %}"
        "{%- if message.role == 'user' %}user:{{ message.content }}\n"
        "{%- elif message.role == 'assistant' %}assistant:{{ message.content }}"
        "{%- if message.tool_calls %}"
        "{%- for call in message.tool_calls %}<tool_call>{{ call.function.name }}:{{ call.function.arguments "
        "}}</tool_call>{%- endfor %}"
        "{%- endif %}\n"
        "{%- elif message.role == 'tool' %}tool:{{ message.tool_call_id }}={{ message.content }}\n"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}");

    zedinfer::ChatMessageMM user;
    user.role = "user";
    user.content = std::vector<zedinfer::ContentPart>{zedinfer::TextPart{"List files."}};

    zedinfer::ChatMessageMM assistant;
    assistant.role = "assistant";
    assistant.content = std::string();
    assistant.tool_calls = nlohmann::ordered_json::array(
        {{{"id", "call_unique"},
          {"type", "function"},
          {"function", {{"name", "exec_command"}, {"arguments", "{\"cmd\":\"ls\"}"}}}}});

    zedinfer::ChatMessageMM tool;
    tool.role = "tool";
    tool.tool_call_id = "call_unique";
    tool.content = std::vector<zedinfer::ContentPart>{zedinfer::TextPart{"main.cpp\nREADME.md"}};

    std::string rendered = tpl.render({user, assistant, tool}, /*add_generation_prompt=*/true);

    EXPECT_NE(rendered.find("user:List files."), std::string::npos);
    EXPECT_NE(rendered.find("<tool_call>exec_command:{\"cmd\":\"ls\"}</tool_call>"), std::string::npos);
    EXPECT_NE(rendered.find("tool:call_unique=main.cpp\nREADME.md"), std::string::npos);
    EXPECT_NE(rendered.find("<|im_start|>assistant"), std::string::npos);
}
