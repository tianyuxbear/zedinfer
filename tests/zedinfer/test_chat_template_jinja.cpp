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
