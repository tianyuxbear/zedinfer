#include "zedinfer/multimodal_processor.hpp"
#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "utils/types.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer {

namespace {

// constexpr base64 decode table built at compile-time — fully thread-safe
// without runtime initialization races. Mapped via a consteval-style lambda.
constexpr std::array<int8_t, 256> make_base64_table() {
    std::array<int8_t, 256> t{};
    for (int i = 0; i < 256; ++i) { t[i] = -1; }
    for (int i = 'A'; i <= 'Z'; ++i) { t[i] = i - 'A'; }
    for (int i = 'a'; i <= 'z'; ++i) { t[i] = 26 + (i - 'a'); }
    for (int i = '0'; i <= '9'; ++i) { t[i] = 52 + (i - '0'); }
    t[static_cast<unsigned char>('+')] = 62;
    t[static_cast<unsigned char>('/')] = 63;
    return t;
}
constexpr std::array<int8_t, 256> kBase64Table = make_base64_table();

// Standard base64 decoder. Returns raw bytes; ignores whitespace; tolerates
// optional padding.
std::vector<uint8_t> base64_decode(std::string_view s) {
    std::vector<uint8_t> out;
    out.reserve((s.size() * 3) / 4);
    int val = 0;
    int bits = -8;
    for (char c : s) {
        if (c == '=') {
            break;
        }
        const int v = kBase64Table[static_cast<unsigned char>(c)];
        if (v < 0) {
            continue; // skip whitespace / unknown chars
        }
        val = (val << 6) + v;
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

int round_by_factor(int value, int factor) {
    return static_cast<int>(std::round(static_cast<double>(value) / factor)) * factor;
}

int ceil_by_factor(double value, int factor) {
    return static_cast<int>(std::ceil(value / factor)) * factor;
}

int floor_by_factor(double value, int factor) {
    return static_cast<int>(std::floor(value / factor)) * factor;
}

} // namespace

ImagePayload MultiModalProcessor::decode_data_uri(std::string_view data_uri) {
    // Strip optional "data:image/...;base64," prefix.
    auto comma = data_uri.find(',');
    std::string_view b64 = (comma != std::string_view::npos) ? data_uri.substr(comma + 1) : data_uri;

    auto bytes = base64_decode(b64);
    if (bytes.empty()) {
        throw std::runtime_error("MultiModalProcessor: empty image bytes after base64 decode");
    }

    int w = 0, h = 0, ch = 0;
    auto* pixels = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &ch, /*desired=*/3);
    if (!pixels) {
        throw std::runtime_error(std::string("MultiModalProcessor: stb_image: ") + stbi_failure_reason());
    }
    ImagePayload pl;
    pl.width = w;
    pl.height = h;
    pl.rgb_pixels.assign(pixels, pixels + static_cast<size_t>(w) * h * 3);
    stbi_image_free(pixels);
    return pl;
}

MultiModalProcessor::MultiModalProcessor(const model::VisionConfig& cfg) : cfg_(cfg) {}

std::pair<int, int> MultiModalProcessor::target_size_for(int h, int w) const {
    if (h <= 0 || w <= 0) {
        throw std::runtime_error("MultiModalProcessor::target_size_for: invalid image size");
    }

    const int align = cfg_.patch_size * cfg_.spatial_merge_size;
    if (align <= 0) {
        throw std::runtime_error("MultiModalProcessor::target_size_for: invalid patch/merge alignment");
    }

    int target_h = std::max(align, round_by_factor(h, align));
    int target_w = std::max(align, round_by_factor(w, align));

    const double original_pixels = static_cast<double>(h) * static_cast<double>(w);
    const int max_pixels = cfg_.max_pixels;
    if (max_pixels > 0 && static_cast<long long>(target_h) * target_w > max_pixels) {
        const double beta = std::sqrt(original_pixels / static_cast<double>(max_pixels));
        target_h = std::max(align, floor_by_factor(static_cast<double>(h) / beta, align));
        target_w = std::max(align, floor_by_factor(static_cast<double>(w) / beta, align));
    }

    const int min_pixels = cfg_.min_pixels;
    if (min_pixels > 0 && static_cast<long long>(target_h) * target_w < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) / original_pixels);
        target_h = std::max(align, ceil_by_factor(static_cast<double>(h) * beta, align));
        target_w = std::max(align, ceil_by_factor(static_cast<double>(w) * beta, align));
    }

    return {target_h, target_w};
}

ImageTokenGrid MultiModalProcessor::image_token_grid_for(int h, int w) const {
    const auto [target_h, target_w] = target_size_for(h, w);
    const int patches_h = target_h / cfg_.patch_size;
    const int patches_w = target_w / cfg_.patch_size;
    const int merged_h = patches_h / cfg_.spatial_merge_size;
    const int merged_w = patches_w / cfg_.spatial_merge_size;
    return ImageTokenGrid{1, merged_h, merged_w};
}

int MultiModalProcessor::num_image_tokens_for(int h, int w) const {
    return static_cast<int>(image_token_grid_for(h, w).num_tokens());
}

ProcessedImage MultiModalProcessor::process(const ImagePayload& img, const ExecutorConfig& exec) const {
    if (img.rgb_pixels.empty() || img.width <= 0 || img.height <= 0) {
        throw std::runtime_error("MultiModalProcessor::process: empty / invalid image payload");
    }

    // 1. Smart resize. This mirrors Qwen-VL preprocessing: preserve aspect
    // ratio, align to patch_size * spatial_merge_size, and keep the area within
    // min/max pixel limits. The max limit is especially important because the
    // current vision attention is O(N^2) in pre-merge patch tokens.
    const auto [new_h, new_w] = target_size_for(img.height, img.width);

    // Use Catmull-Rom (cubic interpolating spline) — closest commonly-available
    // approximation to PIL's BICUBIC used by HF's Qwen3VL image preprocessor.
    // Falls back to linear if cubic fails (size 1 edge cases).
    std::vector<uint8_t> resized(static_cast<size_t>(new_h) * new_w * 3);
    {
        void* ok = stbir_resize(img.rgb_pixels.data(), img.width, img.height, /*input_stride=*/0, resized.data(), new_w,
                                new_h, /*output_stride=*/0, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP,
                                STBIR_FILTER_CATMULLROM);
        if (!ok) {
            // Last-ditch fallback: linear bilinear (matches the previous behavior).
            stbir_resize_uint8_linear(img.rgb_pixels.data(), img.width, img.height, 0, resized.data(), new_w, new_h, 0,
                                      STBIR_RGB);
        }
    }

    // 2. Normalize: (x/255 - mean) / std with mean=std=0.5 (Qwen3.5-VL).
    const float inv_127_5 = 1.0f / 127.5f;
    std::vector<float> normed(resized.size());
    for (size_t i = 0; i < resized.size(); ++i) { normed[i] = static_cast<float>(resized[i]) * inv_127_5 - 1.0f; }

    // 3. Patchify into [N_patches, 3*tps*ps*ps]. For static images we replicate
    //    across the temporal axis (tps slots).
    const int ps = cfg_.patch_size;
    const int tps = cfg_.temporal_patch_size;
    const int patches_h = new_h / ps;
    const int patches_w = new_w / ps;
    const int n_patches = patches_h * patches_w;
    const int patch_dim = 3 * tps * ps * ps;

    std::vector<float> patches(static_cast<size_t>(n_patches) * patch_dim);
    for (int ph = 0; ph < patches_h; ++ph) {
        for (int pw = 0; pw < patches_w; ++pw) {
            const int patch_idx = ph * patches_w + pw;
            for (int t = 0; t < tps; ++t) {
                for (int c = 0; c < 3; ++c) {
                    for (int y = 0; y < ps; ++y) {
                        for (int x = 0; x < ps; ++x) {
                            const int src = ((ph * ps + y) * new_w + (pw * ps + x)) * 3 + c;
                            const int dst = patch_idx * patch_dim + ((c * tps + t) * ps + y) * ps + x;
                            patches[dst] = normed[src];
                        }
                    }
                }
            }
        }
    }

    // 4. FP32 → BF16 host-side, then H2D.
    std::vector<zedinfer::bf16_t> bf16_buf(patches.size());
    for (size_t i = 0; i < patches.size(); ++i) { bf16_buf[i] = zedinfer::utils::cast<zedinfer::bf16_t>(patches[i]); }
    auto pt = Tensor::create({static_cast<size_t>(n_patches), static_cast<size_t>(patch_dim)}, ZEDINFER_DTYPE_BF16,
                             exec.device_type, exec.device_id);
    {
        auto* api = core::context().runtime().api();
        api->memcpy_sync(pt->data(), bf16_buf.data(), bf16_buf.size() * sizeof(zedinfer::bf16_t),
                         exec.device_type == ZEDINFER_DEVICE_CPU ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_H2D);
    }

    // 5. pos_ids_thw: (t=0, h=ph, w=pw) for each patch, row-major over (ph, pw).
    auto pos
        = Tensor::create({static_cast<size_t>(n_patches), 3}, ZEDINFER_DTYPE_I32, exec.device_type, exec.device_id);
    std::vector<int32_t> pos_host(static_cast<size_t>(n_patches) * 3);
    for (int ph = 0; ph < patches_h; ++ph) {
        for (int pw = 0; pw < patches_w; ++pw) {
            const int i = ph * patches_w + pw;
            pos_host[i * 3 + 0] = 0;
            pos_host[i * 3 + 1] = ph;
            pos_host[i * 3 + 2] = pw;
        }
    }
    {
        auto* api = core::context().runtime().api();
        api->memcpy_sync(pos->data(), pos_host.data(), pos_host.size() * sizeof(int32_t),
                         exec.device_type == ZEDINFER_DEVICE_CPU ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_H2D);
    }

    ProcessedImage out;
    out.patches = pt;
    out.pos_ids_thw = pos;
    out.grid_t = 1;
    out.grid_h = patches_h;
    out.grid_w = patches_w;
    const int sms = cfg_.spatial_merge_size;
    out.num_image_tokens = (patches_h / sms) * (patches_w / sms);
    LOGI.printf("[MultiModalProcessor] %dx%d -> %dx%d, grid %dx%d, %d patches, %d image tokens", img.width, img.height,
                new_w, new_h, patches_h, patches_w, n_patches, out.num_image_tokens);
    return out;
}

} // namespace zedinfer
