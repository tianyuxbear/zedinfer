#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/base.hpp"
#include "zedinfer/activation.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace zedinfer {

// Raw decoded image: H × W × 3 RGB bytes in row-major order.
struct ImagePayload {
    int                  height = 0;
    int                  width  = 0;
    std::vector<uint8_t> rgb_pixels;
};

// Output of MultiModalProcessor::process. `patches` is the BF16 tensor fed
// into VisionTower; `pos_ids_thw` carries the (t, h, w) coordinates for 3D
// MRoPE inside the ViT. `num_image_tokens` is the number of LLM-side
// <|image_pad|> tokens that VisionTower will produce (after spatial merge).
struct ProcessedImage {
    tensor_t patches;             // [N_patches, in_channels * T_patch * H_patch * W_patch]
    tensor_t pos_ids_thw;         // [N_patches, 3] int32
    int      num_image_tokens = 0;
    int      grid_t           = 0;
    int      grid_h           = 0;
    int      grid_w           = 0;
};

// Decodes OpenAI-style image_url data URIs into raw RGB and runs the
// Qwen3.5-VL preprocessor (resize → normalize → patchify). Mirrors HF's
// Qwen3VLProcessor at the patch tensor level so VisionTower input matches.
class MultiModalProcessor {
public:
    explicit MultiModalProcessor(const model::VisionConfig& cfg);

    // End-to-end: image bytes → patches + pos_ids_thw on the target device.
    ProcessedImage process(const ImagePayload& img, const ExecutorConfig& exec) const;

    // Decode a "data:image/...;base64,XXX" data URI into raw RGB pixels via
    // stb_image. Throws on malformed base64 or unsupported image format.
    static ImagePayload decode_data_uri(std::string_view data_uri);

    // Number of LLM-side image tokens produced for an image of given pixel
    // height / width. Useful when constructing the prompt before the vision
    // tower has run (to size <|image_pad|> placeholders).
    int num_image_tokens_for(int h, int w) const;

private:
    model::VisionConfig cfg_;
};

} // namespace zedinfer
