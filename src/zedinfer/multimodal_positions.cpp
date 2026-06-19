#include "zedinfer/multimodal_positions.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace zedinfer {

namespace {

int32_t checked_i32(int64_t value, const char* what) {
    if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("multimodal position out of int32 range: ") + what);
    }
    return static_cast<int32_t>(value);
}

void fill_text_positions(std::vector<int32_t>& out, size_t total_tokens, size_t begin, size_t end, int64_t base) {
    for (size_t i = begin; i < end; ++i) {
        const int32_t p = checked_i32(base + static_cast<int64_t>(i - begin), "text");
        out[0 * total_tokens + i] = p;
        out[1 * total_tokens + i] = p;
        out[2 * total_tokens + i] = p;
    }
}

} // namespace

size_t ImageTokenGrid::num_tokens() const {
    if (t <= 0 || h <= 0 || w <= 0) {
        return 0;
    }
    return static_cast<size_t>(t) * static_cast<size_t>(h) * static_cast<size_t>(w);
}

std::vector<int> expand_multimodal_input_ids(const std::vector<int>& input_ids, int image_token_id,
                                             const std::vector<size_t>& image_token_counts) {
    if (image_token_id < 0) {
        throw std::runtime_error("expand_multimodal_input_ids: invalid image token id");
    }

    size_t extra_tokens = 0;
    for (size_t count : image_token_counts) {
        if (count == 0) {
            throw std::runtime_error("expand_multimodal_input_ids: image token count must be positive");
        }
        extra_tokens += count - 1;
    }

    std::vector<int> expanded;
    expanded.reserve(input_ids.size() + extra_tokens);
    size_t image_idx = 0;
    for (int id : input_ids) {
        if (id != image_token_id) {
            expanded.push_back(id);
            continue;
        }
        if (image_idx >= image_token_counts.size()) {
            throw std::runtime_error("expand_multimodal_input_ids: more <|image_pad|> placeholders than images");
        }
        const size_t count = image_token_counts[image_idx++];
        expanded.insert(expanded.end(), count, image_token_id);
    }
    if (image_idx != image_token_counts.size()) {
        throw std::runtime_error("expand_multimodal_input_ids: fewer <|image_pad|> placeholders than images");
    }
    return expanded;
}

MultimodalPositionIds build_multimodal_position_ids(const std::vector<int>& input_ids, int image_token_id,
                                                    const std::vector<ImageTokenGrid>& image_grids) {
    if (image_token_id < 0) {
        throw std::runtime_error("build_multimodal_position_ids: invalid image token id");
    }

    const size_t total_tokens = input_ids.size();
    MultimodalPositionIds result;
    result.pos_ids_thw.assign(3 * total_tokens, 0);

    size_t cursor = 0;
    int64_t logical_pos = 0;

    for (size_t image_idx = 0; image_idx < image_grids.size(); ++image_idx) {
        const auto& grid = image_grids[image_idx];
        const size_t expected_image_tokens = grid.num_tokens();
        if (expected_image_tokens == 0) {
            throw std::runtime_error("build_multimodal_position_ids: empty image token grid");
        }

        size_t run_begin = cursor;
        while (run_begin < total_tokens && input_ids[run_begin] != image_token_id) { ++run_begin; }
        if (run_begin == total_tokens) {
            throw std::runtime_error("build_multimodal_position_ids: fewer <|image_pad|> runs than images");
        }

        fill_text_positions(result.pos_ids_thw, total_tokens, cursor, run_begin, logical_pos);
        logical_pos += static_cast<int64_t>(run_begin - cursor);

        size_t run_end = run_begin;
        while (run_end < total_tokens && input_ids[run_end] == image_token_id) { ++run_end; }
        const size_t actual_image_tokens = run_end - run_begin;
        if (actual_image_tokens != expected_image_tokens) {
            throw std::runtime_error("build_multimodal_position_ids: <|image_pad|> run has "
                                     + std::to_string(actual_image_tokens) + " token(s), expected "
                                     + std::to_string(expected_image_tokens));
        }

        size_t out_idx = run_begin;
        const int64_t image_base = logical_pos;
        for (int t = 0; t < grid.t; ++t) {
            for (int h = 0; h < grid.h; ++h) {
                for (int w = 0; w < grid.w; ++w) {
                    result.pos_ids_thw[0 * total_tokens + out_idx] = checked_i32(image_base + t, "image_t");
                    result.pos_ids_thw[1 * total_tokens + out_idx] = checked_i32(image_base + h, "image_h");
                    result.pos_ids_thw[2 * total_tokens + out_idx] = checked_i32(image_base + w, "image_w");
                    ++out_idx;
                }
            }
        }

        logical_pos = image_base + std::max({grid.t, grid.h, grid.w});
        cursor = run_end;
    }

    for (size_t i = cursor; i < total_tokens; ++i) {
        if (input_ids[i] == image_token_id) {
            throw std::runtime_error("build_multimodal_position_ids: more <|image_pad|> runs than images");
        }
    }

    fill_text_positions(result.pos_ids_thw, total_tokens, cursor, total_tokens, logical_pos);
    logical_pos += static_cast<int64_t>(total_tokens - cursor);
    result.mrope_position_delta = checked_i32(logical_pos - static_cast<int64_t>(total_tokens), "delta");
    return result;
}

} // namespace zedinfer
