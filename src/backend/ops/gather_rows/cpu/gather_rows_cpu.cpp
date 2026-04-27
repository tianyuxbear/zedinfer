#include "backend/ops/gather_rows/cpu/gather_rows_cpu.hpp"

#include <cstring>

namespace zedinfer::ops::cpu {

void gather_rows(std::byte* dst, const std::byte* src, const std::int32_t* indices, std::size_t num_rows,
                 std::size_t row_bytes) {
    for (std::size_t r = 0; r < num_rows; ++r) {
        std::memcpy(dst + r * row_bytes, src + static_cast<std::size_t>(indices[r]) * row_bytes, row_bytes);
    }
}

} // namespace zedinfer::ops::cpu
