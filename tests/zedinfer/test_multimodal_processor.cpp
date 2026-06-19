#include "zedinfer/multimodal_processor.hpp"

#include <gtest/gtest.h>

using zedinfer::MultiModalProcessor;
using zedinfer::model::VisionConfig;

namespace {

VisionConfig qwen_vision_config() {
    VisionConfig cfg;
    cfg.patch_size = 16;
    cfg.spatial_merge_size = 2;
    cfg.temporal_patch_size = 2;
    cfg.min_pixels = 65536;   // 64 merged image tokens.
    cfg.max_pixels = 1048576; // 1024 merged image tokens.
    return cfg;
}

} // namespace

TEST(MultiModalProcessorTest, CapsLargeImagesToMaxImageTokens) {
    MultiModalProcessor processor(qwen_vision_config());

    EXPECT_EQ(processor.num_image_tokens_for(4096, 4096), 1024);
}

TEST(MultiModalProcessorTest, UpscalesTinyImagesToMinImageTokens) {
    MultiModalProcessor processor(qwen_vision_config());

    EXPECT_EQ(processor.num_image_tokens_for(16, 16), 64);
}

TEST(MultiModalProcessorTest, PreservesMediumImageAspectAndAlignment) {
    MultiModalProcessor processor(qwen_vision_config());

    // 1024x512 is already aligned to patch_size * spatial_merge_size.
    EXPECT_EQ(processor.num_image_tokens_for(512, 1024), 512);
}

TEST(MultiModalProcessorTest, ReportsMergedImageTokenGrid) {
    MultiModalProcessor processor(qwen_vision_config());

    auto grid = processor.image_token_grid_for(512, 1024);
    EXPECT_EQ(grid.t, 1);
    EXPECT_EQ(grid.h, 16);
    EXPECT_EQ(grid.w, 32);
    EXPECT_EQ(grid.num_tokens(), 512);
}
