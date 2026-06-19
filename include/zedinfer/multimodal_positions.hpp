#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zedinfer {

struct ImageTokenGrid {
    int t = 1;
    int h = 0;
    int w = 0;

    size_t num_tokens() const;
};

struct MultimodalPositionIds {
    // Row-major [3, N]: all t positions, then h positions, then w positions.
    std::vector<int32_t> pos_ids_thw;
    int32_t mrope_position_delta = 0;
};

MultimodalPositionIds build_multimodal_position_ids(const std::vector<int>& input_ids, int image_token_id,
                                                    const std::vector<ImageTokenGrid>& image_grids);

std::vector<int> expand_multimodal_input_ids(const std::vector<int>& input_ids, int image_token_id,
                                             const std::vector<size_t>& image_token_counts);

} // namespace zedinfer
