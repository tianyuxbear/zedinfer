#include "zedinfer/multimodal_positions.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using zedinfer::build_multimodal_position_ids;
using zedinfer::expand_multimodal_input_ids;
using zedinfer::ImageTokenGrid;

TEST(MultimodalPositionsTest, ExpandsImagePadPlaceholdersAtTokenLevel) {
    constexpr int image_token = 99;
    const std::vector<int> input_ids = {1, image_token, 2};

    const auto got = expand_multimodal_input_ids(input_ids, image_token, {4});

    const std::vector<int> expected = {1, image_token, image_token, image_token, image_token, 2};
    EXPECT_EQ(got, expected);
}

TEST(MultimodalPositionsTest, ExpandsMultipleImagesInEncounterOrder) {
    constexpr int image_token = 99;
    const std::vector<int> input_ids = {1, image_token, 2, image_token, 3};

    const auto got = expand_multimodal_input_ids(input_ids, image_token, {2, 3});

    const std::vector<int> expected = {1, image_token, image_token, 2, image_token, image_token, image_token, 3};
    EXPECT_EQ(got, expected);
}

TEST(MultimodalPositionsTest, RejectsImagePlaceholderMismatchDuringExpansion) {
    constexpr int image_token = 99;

    EXPECT_THROW((void)expand_multimodal_input_ids({1, image_token}, image_token, {2, 2}), std::runtime_error);
    EXPECT_THROW((void)expand_multimodal_input_ids({1, image_token, image_token}, image_token, {2}),
                 std::runtime_error);
}

TEST(MultimodalPositionsTest, BuildsImageGridAndResumesTextAfterGridMax) {
    constexpr int image_token = 99;
    const std::vector<int> input_ids = {1, 2, image_token, image_token, image_token, image_token, 3, 4};
    const std::vector<ImageTokenGrid> grids = {{1, 2, 2}};

    auto got = build_multimodal_position_ids(input_ids, image_token, grids);

    const std::vector<int32_t> expected_t = {0, 1, 2, 2, 2, 2, 4, 5};
    const std::vector<int32_t> expected_h = {0, 1, 2, 2, 3, 3, 4, 5};
    const std::vector<int32_t> expected_w = {0, 1, 2, 3, 2, 3, 4, 5};

    ASSERT_EQ(got.pos_ids_thw.size(), input_ids.size() * 3);
    for (size_t i = 0; i < input_ids.size(); ++i) {
        EXPECT_EQ(got.pos_ids_thw[0 * input_ids.size() + i], expected_t[i]) << i;
        EXPECT_EQ(got.pos_ids_thw[1 * input_ids.size() + i], expected_h[i]) << i;
        EXPECT_EQ(got.pos_ids_thw[2 * input_ids.size() + i], expected_w[i]) << i;
    }
    EXPECT_EQ(got.mrope_position_delta, -2);
}

TEST(MultimodalPositionsTest, SupportsMultipleImagesInEncounterOrder) {
    constexpr int image_token = 99;
    const std::vector<int> input_ids = {1, image_token, image_token, 2, image_token, image_token, image_token, 3};
    const std::vector<ImageTokenGrid> grids = {{1, 1, 2}, {1, 1, 3}};

    auto got = build_multimodal_position_ids(input_ids, image_token, grids);

    const std::vector<int32_t> expected_t = {0, 1, 1, 3, 4, 4, 4, 7};
    const std::vector<int32_t> expected_h = {0, 1, 1, 3, 4, 4, 4, 7};
    const std::vector<int32_t> expected_w = {0, 1, 2, 3, 4, 5, 6, 7};
    for (size_t i = 0; i < input_ids.size(); ++i) {
        EXPECT_EQ(got.pos_ids_thw[0 * input_ids.size() + i], expected_t[i]) << i;
        EXPECT_EQ(got.pos_ids_thw[1 * input_ids.size() + i], expected_h[i]) << i;
        EXPECT_EQ(got.pos_ids_thw[2 * input_ids.size() + i], expected_w[i]) << i;
    }
    EXPECT_EQ(got.mrope_position_delta, 0);
}

TEST(MultimodalPositionsTest, RejectsImagePadRunLengthMismatch) {
    constexpr int image_token = 99;
    const std::vector<int> input_ids = {1, image_token, image_token, 2};
    const std::vector<ImageTokenGrid> grids = {{1, 2, 2}};

    EXPECT_THROW((void)build_multimodal_position_ids(input_ids, image_token, grids), std::runtime_error);
}

TEST(MultimodalPositionsTest, RejectsExtraImagePadRuns) {
    constexpr int image_token = 99;
    const std::vector<int> input_ids = {1, image_token, image_token, image_token};
    const std::vector<ImageTokenGrid> grids = {{1, 1, 2}};

    EXPECT_THROW((void)build_multimodal_position_ids(input_ids, image_token, grids), std::runtime_error);
}
