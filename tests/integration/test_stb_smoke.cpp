// Smoke test: stb_image decodes a fixture PNG end-to-end.
//
// Verifies third_party/stb vendoring landed correctly. No image preprocessing
// logic here; this just confirms the stb_image translation unit links and
// can read a real PNG from disk.

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <gtest/gtest.h>

TEST(StbSmoke, DecodesTinyFixturePng) {
    int w = 0, h = 0, ch = 0;
    unsigned char* pixels =
        stbi_load("tests/fixtures/tiny_4x4.png", &w, &h, &ch, /*desired_channels=*/3);
    ASSERT_NE(pixels, nullptr) << "stbi_load: " << stbi_failure_reason();
    EXPECT_EQ(w, 4);
    EXPECT_EQ(h, 4);
    // First pixel of an all-red 4x4 RGB PNG should be (255, 0, 0).
    EXPECT_EQ(pixels[0], 255);
    EXPECT_EQ(pixels[1], 0);
    EXPECT_EQ(pixels[2], 0);
    stbi_image_free(pixels);
}
