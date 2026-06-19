// Smoke test: verify minja can parse the Qwen3.5 chat_template.jinja file.
//
// This is the vendoring smoke test for third_party/minja added in P1-T2 of the
// Qwen3.5 implementation plan. It loads the actual chat_template.jinja shipped
// with the Qwen3.5 model and confirms that minja::chat_template can compile it
// without throwing. No rendering is performed here; only template parsing.

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

// Resolves Qwen3.5 chat_template.jinja from ZEDINFER_TEST_MODEL_PATH so the
// test stays portable across machines, matching the convention used by
// test-tokenizer / test-loader (see CLAUDE.md Build & Test section).
TEST(MinjaSmoke, ParsesQwen3_5ChatTemplate) {
    const char* model_path = std::getenv("ZEDINFER_TEST_MODEL_PATH");
    if (!model_path) {
        GTEST_SKIP() << "ZEDINFER_TEST_MODEL_PATH not set; "
                        "set it to a Qwen3.5 model dir to enable this test.";
    }
    const std::string path = std::string(model_path) + "/chat_template.jinja";
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open()) << "failed to open " << path;
    std::string tpl((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(tpl.empty()) << "chat_template.jinja is empty";

    EXPECT_NO_THROW({
        minja::chat_template compiled(tpl, /*bos=*/"", /*eos=*/"");
        (void)compiled;
    });
}
